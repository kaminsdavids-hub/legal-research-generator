"""Wiring the maieutic gates to the application's corpus and clients.

The only module here that may import ``legal_research.*``, mirroring
``modules/dialectic/service.py``. Everything else stays transport-agnostic so
the gates can be exercised offline.
"""

from __future__ import annotations

from typing import Any

from .grounding import CitationVerifier


class CorpusCitationVerifier(CitationVerifier):
    """Resolve citations against the corpus, and check they support the claim.

    Two stages, because they fail differently. Resolution asks whether the
    authority exists — a fabricated citation dies here. Support asks whether it
    says what the node claims it says; a real case cited for something it does
    not hold survives resolution and must not survive this.

    Non-case authority is resolved by the corpus alone. CourtListener indexes
    case law and cannot adjudicate a C.F.R. section however correct it is, which
    cost three eval runs before it was understood (REMEDIATION §11.9a).
    """

    def __init__(self, corpus: Any, support_scorer: Any | None = None) -> None:
        self._corpus = corpus
        self._support = support_scorer
        self._by_citation = self._index(corpus)

    @staticmethod
    def _index(corpus: Any) -> dict[str, Any]:
        from modules.dialectic.service import _CorpusCiteRetriever

        index: dict[str, Any] = {}
        for record in getattr(corpus, "records", []):
            cite = _CorpusCiteRetriever._format(record)
            if cite:
                index[cite] = record
        return index

    def resolve(self, citation: str) -> tuple[bool, str]:
        record = self._by_citation.get(citation.strip())
        if record is None:
            return False, "no corpus record carries this citation"

        status = getattr(getattr(record, "status", None), "value", "in_force")
        if status != "in_force":
            # Resolving is not the same as being good law. The node may still
            # merge, but the caller must be able to see this.
            return True, f"resolved, but NOT CURRENTLY OPERATIVE ({status})"
        return True, f"resolved to {record.title}"

    def supports(self, citation: str, claim: str) -> tuple[bool, str]:
        record = self._by_citation.get(citation.strip())
        if record is None:
            return False, "citation does not resolve"

        passages = list(getattr(record, "passages", []))
        if not passages:
            return False, "the corpus record carries no text to check the claim against"

        if self._support is None:
            # Refusing to guess. A quote-to-claim check needs a scorer, and
            # asserting support without one would make this gate decorative
            # exactly where fabrication is subtlest.
            return False, (
                "no support scorer configured; a citation cannot be taken to "
                "support a claim on trust"
            )

        # `score` takes one passage at a time; the best-supporting passage is
        # what matters, since a record supports a claim if any of its text does.
        try:
            best = max(float(self._support.score(claim, p)) for p in passages)
        except Exception as exc:  # noqa: BLE001 - a broken scorer is a failure, not a pass
            return False, f"support scorer failed: {exc}"

        # The scorer carries the threshold for its own scale -- lexical overlap
        # and NLI entailment probabilities are not comparable, and inventing a
        # cut here would be wrong for at least one of them.
        threshold = float(getattr(self._support, "threshold", 0.0))
        if best < threshold:
            return False, (
                f"retrieved text does not support the claim "
                f"(score {best:.2f} < {threshold:.2f})"
            )
        return True, f"supported by {record.title} (score {best:.2f})"


#: What the exchange is asked to cite for.
#:
#: ``claim`` is the original framing: argue the author's claim, cite as you go.
#: ``premise`` exists because that framing produced authorities that could not be
#: grounded, for a structural reason rather than a fixable one — an authority
#: attached to the paper's *novel* claim cannot be supported by the corpus,
#: because a claim the corpus supports would be COMMONPLACE and the paper would
#: have nothing to argue (REMEDIATION §14). This asks the models to cite for the
#: settled propositions the claim builds from instead.
CITE_FOR = ("claim", "premise")

_CLAIM_PROMPT = (
    "Is the following claim correct? {claim}\n\n"
    "It was written in answer to: {question}"
)

_PREMISE_PROMPT = (
    "An author has made this claim, which is their own contribution and is NOT "
    "stated in any existing source: {claim}\n\n"
    "It was written in answer to: {question}\n\n"
    "Do not argue for or against the claim itself, and do not attach authority "
    "to it. Instead identify the ESTABLISHED propositions it depends on — the "
    "settled law and undisputed facts a reader must already accept before the "
    "claim can even be considered — and argue about those. Every proposition you "
    "offer should be one an existing source actually states, not an extension of "
    "one."
)


def debate_prompt(question: str, answer: str, cite_for: str = "claim") -> str:
    """Frame what the two sides argue over.

    The Socratic question is context; the *answer* is what the exchange is about.
    Sending the question instead would have the models debate a topic and hand
    back material about the subject in general, when what the loop needs is
    pressure on the specific thing the author just committed to.

    ``cite_for`` chooses whether they argue the claim or the ground beneath it.
    """
    if cite_for not in CITE_FOR:
        raise ValueError(f"cite_for must be one of {CITE_FOR}, got {cite_for!r}")
    template = _CLAIM_PROMPT if cite_for == "claim" else _PREMISE_PROMPT
    return template.format(
        claim=" ".join(answer.split()), question=" ".join(question.split())
    )


def build_turn_provider(settings: Any, cite_for: str = "claim") -> Any:
    """A `TurnProvider` running the real dialectic engine.

    Built once and reused, so the model clients and the citation cache are not
    reconstructed per answer. Exceptions are deliberately *not* caught here —
    `Session.answer` owns that, so a model being down costs the author the
    machine's pressure and never their own answer.
    """
    from modules.dialectic.service import build_dialectic_chat

    chat = build_dialectic_chat(settings)

    def provide(question: str, answer: str) -> Any:
        return chat.chat(debate_prompt(question, answer, cite_for))

    return provide


def build_grounding_gate(settings: Any) -> Any:
    """Build a grounding gate wired to the configured corpus."""
    from legal_research.citations.corpus import load_corpus

    from .grounding import GroundingGate

    try:
        corpus = load_corpus(settings.corpus_path)
    except Exception:  # noqa: BLE001 - no corpus is a real deployment state
        return GroundingGate(verifier=None)

    scorer = None
    try:
        from legal_research.citations.support import build_support_scorer

        scorer = build_support_scorer(settings)
    except Exception:  # noqa: BLE001 - scorer is optional; the gate fails closed without it
        scorer = None
    return GroundingGate(verifier=CorpusCitationVerifier(corpus, scorer))
