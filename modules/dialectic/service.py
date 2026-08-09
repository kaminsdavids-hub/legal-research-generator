"""Construct a :class:`DialecticChat` from application settings.

This is the only place the dialectic module touches the host application's
configuration and LLM clients; the rest of the package stays transport-agnostic.
"""

from __future__ import annotations

from typing import Any

from .engine import DialecticChat
from .verification import CourtListenerClient, RateBudget


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


def build_dialectic_chat(settings: Any) -> DialecticChat:
    """Build a dialectic chat engine wired to the configured local models."""

    from legal_research.llm.mock import MockLLM
    from legal_research.llm.openai_compat import OpenAICompatLLM

    def make(model: str) -> _ClientAdapter:
        if settings.llm_mode == "mock":
            return _ClientAdapter(model, MockLLM(model))
        return _ClientAdapter(
            model,
            OpenAICompatLLM(
                name=model,
                base_url=settings.dialectic_base_url,
                model=model,
                api_key=settings.llm_api_key,
                timeout=max(5.0, float(settings.dialectic_timeout_seconds)),
            ),
        )

    courtlistener = None
    token = str(getattr(settings, "courtlistener_token", "") or "").strip()
    if token:
        courtlistener = CourtListenerClient(token=token, budget=RateBudget())

    return DialecticChat(
        thesis_client=make(settings.dialectic_thesis_model),
        antithesis_client=make(settings.dialectic_antithesis_model),
        synthesis_client=make(settings.dialectic_synthesis_model),
        nli_client=make(settings.dialectic_nli_model),
        courtlistener=courtlistener,
        retriever=_build_retriever(settings),
    )
