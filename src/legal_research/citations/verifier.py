"""Grounding guard + adversarial verifier (spec §2.9, §4).

The :class:`CitationGuard` enforces the hard rule at *write* time: a proposition may
only be cited if the retriever returns a supporting corpus passage. Anything a model
tries to assert from its own weights is blocked.

The :class:`CitationVerifier` is the gatekeeper at *ship* time: for every citation it
confirms (a) the record resolves in the corpus and (b) the record actually supports
the proposition (semantic overlap) with exact quotes and plausible pin-cites.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..config import Settings, get_settings
from ..models import Citation, CiteStatus, VerificationResult
from .corpus import Corpus, load_corpus
from .retriever import Retriever
from .support import (
    LexicalSupportScorer,
    SupportScorer,
    best_support,
    build_support_scorer,
    lexical_support,
)

SUPPORT_THRESHOLD = 0.34
GROUND_THRESHOLD = 0.05


class GroundingError(RuntimeError):
    """Raised when a proposition cannot be grounded in a retrieved source."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


# Backwards-compatible alias: the lexical scorer now lives in ``support``.
support_score = lexical_support


def quote_is_exact(quote: str, passages: list[str]) -> bool:
    norm_quote = _normalize(quote)
    return any(norm_quote and norm_quote in _normalize(p) for p in passages)


class CitationGuard:
    """Grounds propositions through the retriever; blocks ungrounded assertions."""

    def __init__(self, retriever: Retriever, new_id: Callable[[], str]) -> None:
        self._retriever = retriever
        self._new_id = new_id

    def ground(self, proposition: str, quote: str | None = None) -> Citation:
        hits = self._retriever.search(proposition, k=3)
        if not hits or hits[0].score < GROUND_THRESHOLD:
            raise GroundingError(
                f"no corpus source supports proposition: {proposition!r}"
            )
        best = hits[0]
        return Citation(
            id=self._new_id(),
            record_id=best.record_id,
            proposition=proposition,
            quote=quote,
            # No synthetic pin-cite: the corpus carries no internal pagination, and a
            # fabricated page number would itself be a citation error.
            pin_cite=None,
            supporting_passage=best.text,
            from_retrieval=True,
            status=CiteStatus.PENDING,
        )

    @staticmethod
    def assert_grounded(citation: Citation) -> None:
        """Enforce the hard rule: cites must originate from retrieval."""

        if not citation.from_retrieval:
            raise GroundingError(
                f"citation {citation.id} was not produced by the retriever and is blocked"
            )


class CitationVerifier:
    """Adversarial verifier. Never trusts a citation until the corpus confirms it."""

    def __init__(
        self,
        corpus: Corpus,
        support_threshold: float = SUPPORT_THRESHOLD,
        scorer: SupportScorer | None = None,
    ) -> None:
        self._corpus = corpus
        self._scorer: SupportScorer = scorer or LexicalSupportScorer(support_threshold)
        self._threshold = self._scorer.threshold

    def verify(self, citation: Citation) -> VerificationResult:
        # Rule 1: authority must originate from retrieval.
        if not citation.from_retrieval:
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.REMOVED,
                reason="authority did not originate from the retriever (possible hallucination)",
            )

        # Rule 2: the record must resolve in the corpus.
        record = self._corpus.get(citation.record_id)
        if record is None:
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.REMOVED,
                reason="citation does not resolve to any corpus record",
            )

        # Rule 3: quotes must be exact.
        if citation.quote and not quote_is_exact(citation.quote, record.passages):
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.REMOVED,
                reason="quoted language does not appear verbatim in the source",
            )

        # Rule 4: the source must actually support the proposition. The scorer is
        # pluggable — semantic NLI by default in the live profile, with lexical
        # fallback if semantic dependencies are unavailable.
        #
        # Scoring is clause-level: a proposition that joins two holdings from two
        # different authorities is not entailed by either passage on its own, and
        # scoring only the conjunction rejected every citation under NLI.
        best, best_passage = best_support(
            self._scorer,
            citation.proposition,
            record.passages,
            threshold=self._threshold,
        )

        if best < self._threshold:
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.REMOVED,
                reason=(
                    f"source does not support the proposition (support={best:.2f} "
                    f"< {self._threshold:.2f}); possible misattributed holding"
                ),
                supporting_passage=best_passage,
            )

        return VerificationResult(
            citation_id=citation.id,
            record_id=citation.record_id,
            status=CiteStatus.VERIFIED,
            reason=f"resolves and supports the proposition (support={best:.2f})",
            supporting_passage=best_passage,
        )

    def verify_all(self, citations: list[Citation]) -> list[VerificationResult]:
        results = [self.verify(c) for c in citations]
        by_id = {r.citation_id: r for r in results}
        for c in citations:
            r = by_id[c.id]
            c.status = r.status
            if r.supporting_passage:
                c.supporting_passage = r.supporting_passage
            if r.status != CiteStatus.VERIFIED:
                c.note = r.reason
        return results


def build_verifier(
    settings: Settings | None = None,
    corpus: Corpus | None = None,
) -> CitationVerifier:
    """Construct a verifier whose support test honors ``LRG_SUPPORT_SCORER``.

    Defaults to semantic NLI in the live profile; lexical remains available for
    deterministic runs and as a fallback when semantic dependencies are missing.
    """

    s = settings or get_settings()
    c = corpus if corpus is not None else load_corpus(s.corpus_path)
    return CitationVerifier(c, scorer=build_support_scorer(s))
