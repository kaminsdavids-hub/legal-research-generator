"""Construct a :class:`DialecticChat` from application settings.

This is the only place the dialectic module touches the host application's
configuration and LLM clients; the rest of the package stays transport-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any

from .engine import DialecticChat
from .models import NOT_OPERATIVE
from .roles import RoleSpec, detect_family
from .verification import CourtListenerClient, RateBudget

logger = logging.getLogger(__name__)

#: The four roles this module serves, in the order a reader expects them.
ROLES = ("thesis", "antithesis", "synthesis", "nli")


class _ClientAdapter:
    """Adapt the project's ``LLMClient`` to the dialectic ``ChatClient`` protocol.

    The engine passes plain ``{"role": ..., "content": ...}`` dicts; the project's
    clients expect ``ChatMessage`` dataclasses. ``name`` is the *model* string
    rather than the role, because family detection reads it.
    """

    def __init__(self, model: str, client: Any) -> None:
        self.name = model
        self._client = client

    def chat(self, messages: list[Any], config: Any | None = None) -> str:
        from legal_research.llm.base import ChatMessage, DecodingPolicy, GenerationConfig

        converted = [
            ChatMessage(role=m["role"], content=m["content"]) if isinstance(m, dict) else m
            for m in messages
        ]
        if isinstance(config, dict):
            resolved = GenerationConfig(
                temperature=float(config.get("temperature", 0.0)),
                top_p=float(config.get("top_p", 1.0)),
                max_tokens=int(config.get("max_tokens", 1024)),
                seed=config.get("seed"),
            )
        else:
            resolved = config or DecodingPolicy.COLD.config
        return str(self._client.chat(converted, resolved))


class _CorpusCiteRetriever:
    """Adapt the project's corpus retriever to the ``CiteRetriever`` protocol.

    Searches the corpus with the court hint and proposition, then formats each
    hit's record into a normalized reporter citation. Only ``case`` records with
    a complete volume/reporter/page triple can produce one; statutes, regulations
    and secondary sources are skipped, because CourtListener's citation-lookup
    endpoint resolves case citations.

    The adapter lives here, not in ``retrieval.py``, so the package stays
    transport-agnostic: this is the only module allowed to import
    ``legal_research.*``. Nothing under ``src/legal_research/`` is modified —
    this wraps ``citations/retriever.py`` and ``citations/corpus.py`` as they are.
    """

    def __init__(self, retriever: Any, corpus: Any, k: int = 5) -> None:
        self._retriever = retriever
        self._corpus = corpus
        self._k = k

    @staticmethod
    def _format(record: Any) -> str:
        """Normalized citation for a corpus record, or "" when unformattable.

        Cases render as ``"<volume> <reporter> <page>"``, which CourtListener's
        citation-lookup endpoint can adjudicate. Statutes and regulations render
        as ``"<code> <section>"``: CourtListener cannot verify those, but they
        must still be retrievable, or an authority like the EAR published
        exclusion could never be proposed at all and a rule's operative status
        could never be put in issue.
        """
        volume = getattr(record, "volume", None)
        page = getattr(record, "page", None)
        reporter = str(getattr(record, "reporter", "") or "").strip()
        if volume is not None and page is not None and reporter:
            return f"{volume} {reporter} {page}"

        code = str(getattr(record, "code", "") or "").strip()
        section = str(getattr(record, "section", "") or "").strip()
        if code and section:
            return f"{code} {section}"
        return ""

    def annotate(self, cite: str) -> str:
        """Status note for a cite whose corpus record is not in force.

        Returns "" for in-force authority, so the common case adds nothing to
        the slot. See :class:`modules.dialectic.retrieval.CiteAnnotator`.
        """
        for record in getattr(self._corpus, "records", []):
            if self._format(record) != cite:
                continue
            status = getattr(record, "status", None)
            value = getattr(status, "value", str(status or ""))
            if value and value != "in_force":
                note = str(getattr(record, "status_note", "") or "").strip()
                return f"{NOT_OPERATIVE} ({value}): {note}" if note else f"{NOT_OPERATIVE} ({value})"
            return ""
        return ""

    def propose(self, court_hint: str, proposition: str) -> list[str]:
        query = f"{court_hint} {proposition}".strip()
        if not query:
            return []
        try:
            hits = self._retriever.search(query, k=self._k)
        except Exception:  # noqa: BLE001 - a retrieval miss must not void the turn
            return []

        cites: list[str] = []
        for hit in hits:
            record = self._corpus.get(getattr(hit, "record_id", ""))
            if record is None:
                continue
            cite = self._format(record)
            if cite and cite not in cites:
                cites.append(cite)
        return cites


def _build_retriever(settings: Any) -> Any | None:
    """Build the corpus-backed retriever, or ``None`` if the corpus is absent.

    Returning ``None`` is meaningful: the engine then reports every slot as
    structurally unverifiable rather than leaving it in a bare ``pending``.
    """
    try:
        from legal_research.citations.corpus import load_corpus
        from legal_research.citations.retriever import build_retriever

        corpus = load_corpus(settings.corpus_path)
        return _CorpusCiteRetriever(build_retriever(settings, corpus), corpus)
    except Exception:  # noqa: BLE001 - no corpus configured is a normal deployment
        return None


def is_local_endpoint(base_url: str) -> bool:
    """Whether the endpoint is the same machine as the backend.

    Delegates to the client's own rule rather than re-deriving it: the same
    answer now decides which request fields are sent, and a second copy that
    disagreed would send a payload the provider rejects.
    """

    from legal_research.llm.base import is_local_endpoint as _canonical

    return _canonical(base_url)


def resolve_role(settings: Any, role: str) -> RoleSpec:
    """Endpoint and model for one debate role, honouring per-role overrides.

    One shared base URL made a mixed lineup impossible: a role could not be
    pointed at a hosted API while the others stayed on Ollama, so comparing this
    pipeline against a frontier model could not be configured at all. Empty
    overrides mean "use ``dialectic_base_url``", so existing single-endpoint
    setups resolve exactly as before.
    """

    model = str(getattr(settings, f"dialectic_{role}_model", "") or "").strip()
    base_url = str(getattr(settings, f"dialectic_{role}_base_url", "") or "").strip()
    return RoleSpec(
        role=role,
        model=model,
        family=detect_family(model),
        base_url=base_url or str(settings.dialectic_base_url or "").strip(),
    )


def resolve_api_key(settings: Any, spec: RoleSpec) -> str:
    """Key for a role's endpoint, refusing to guess one for a remote host.

    A per-role endpoint pointing off this machine with no per-role key would
    otherwise be sent ``llm_api_key`` — which defaults to the literal
    ``"local-not-secret"`` — and the run would fail at the provider with an auth
    error that reads like a network problem, or worse, succeed against something
    that ignores the header. Naming the missing variable is the whole fix.
    """

    key = str(getattr(settings, f"dialectic_{spec.role}_api_key", "") or "").strip()
    if key:
        return key
    if not is_local_endpoint(spec.base_url):
        raise ValueError(
            f"dialectic role {spec.role!r} points at {spec.base_url!r}, which is not "
            f"this machine, but LRG_DIALECTIC_{spec.role.upper()}_API_KEY is empty. "
            "Set it, or point the role back at the local endpoint — a remote debate "
            "role must not fall back to the local placeholder key."
        )
    return str(settings.llm_api_key or "")


def build_dialectic_chat(settings: Any) -> DialecticChat:
    """Build a dialectic chat engine wired to the configured models."""

    from legal_research.llm.mock import MockLLM
    from legal_research.llm.openai_compat import OpenAICompatLLM

    def make(role: str) -> _ClientAdapter:
        spec = resolve_role(settings, role)
        if settings.llm_mode == "mock":
            return _ClientAdapter(spec.model, MockLLM(spec.model))
        if not is_local_endpoint(spec.base_url):
            logger.warning(
                "dialectic role %s (%s) is served from %s: prompt text for this "
                "role leaves the local machine.",
                role,
                spec.model,
                spec.base_url,
            )
        return _ClientAdapter(
            spec.model,
            OpenAICompatLLM(
                name=spec.model,
                base_url=spec.base_url,
                model=spec.model,
                api_key=resolve_api_key(settings, spec),
                timeout=max(5.0, float(settings.dialectic_timeout_seconds)),
            ),
        )

    courtlistener = None
    token = str(getattr(settings, "courtlistener_token", "") or "").strip()
    if token:
        courtlistener = CourtListenerClient(token=token, budget=RateBudget())

    return DialecticChat(
        thesis_client=make("thesis"),
        antithesis_client=make("antithesis"),
        synthesis_client=make("synthesis"),
        nli_client=make("nli"),
        courtlistener=courtlistener,
        retriever=_build_retriever(settings),
    )
