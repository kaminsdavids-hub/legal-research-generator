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
from typing import Any

from ..config import Settings, get_settings
from ..models import Citation, CiteStatus, VerificationResult
from .corpus import Corpus, load_corpus
from .retriever import Retriever
from .support import (
    LexicalSupportScorer,
    SupportRelation,
    SupportScorer,
    assess_support,
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


#: How many hits the guard considers before choosing. Wider than the old k=3
#: because the passage that actually supported the paper's thesis sat at rank 4.
GROUND_CANDIDATES = 6


class CitationGuard:
    """Grounds propositions through the retriever; blocks ungrounded assertions."""

    def __init__(
        self,
        retriever: Retriever,
        new_id: Callable[[], str],
        scorer: SupportScorer | None = None,
        candidates: int = GROUND_CANDIDATES,
    ) -> None:
        self._retriever = retriever
        self._new_id = new_id
        #: Optional. Without it the guard cites the top-ranked hit, which is the
        #: behaviour every run before this one had.
        self._scorer = scorer
        self._candidates = candidates

    def ground(self, proposition: str, quote: str | None = None) -> Citation:
        hits = self._retriever.search(proposition, k=self._candidates)
        if not hits or hits[0].score < GROUND_THRESHOLD:
            raise GroundingError(
                f"no corpus source supports proposition: {proposition!r}"
            )
        best = self._best_supported(proposition, hits)
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

    def _best_supported(self, proposition: str, hits: list[Any]) -> Any:
        """The retrieved passage that best supports the proposition.

        Retrieval ranks by semantic similarity, which is topicality, not
        support. Those differ in a way that decided every citation in nine live
        runs: for the thesis "publishing open model weights is protected
        expression, and the EAR may not treat that publication as a deemed
        export", the ranking was

            1. 0.504  Framework for AI Diffusion         (recites a fact)
            2. 0.350  IEEPA informational materials      (recites a fact)
            3. 0.340  Published information and software (recites a fact)
            4. 0.313  Published information and software (STATES THE RULE)

        Rank 4 is the EAR's published-information exclusion — the authority the
        claim rests on. The guard took rank 1 and never reconsidered, so the
        citation was bound to a fact about 4E091 at write time and the verifier
        was left removing a cite that was wrong before it ever saw it. The
        corpus had the answer, retrieval surfaced it, and the guard cited
        something else.

        Entailment wins over rule support, and rule support over neither; ties
        go to the better-ranked hit, so retrieval still breaks the tie. Scoring
        stops at the first entailment, because nothing below it can win.

        Without a scorer this returns the top hit, exactly as before. That is
        also the honest fallback: choosing on support requires measuring
        support, and the lexical scorer cannot measure the second relation.
        """

        if self._scorer is None:
            return hits[0]

        fallback = None
        for hit in hits:
            assessment = assess_support(
                self._scorer, proposition, [hit.text], retrieved_passage=hit.text
            )
            if assessment.relation is SupportRelation.ENTAILED:
                return hit
            if assessment.relation is SupportRelation.RULE_SUPPORT and fallback is None:
                fallback = hit
        return fallback or hits[0]

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
        rule_relation: bool = True,
    ) -> None:
        self._corpus = corpus
        self._scorer: SupportScorer = scorer or LexicalSupportScorer(support_threshold)
        self._threshold = self._scorer.threshold
        #: Accept a passage that states the rule a proposition applies, as a
        #: separately-labelled and weaker relation. Pass ``False`` for
        #: entailment only -- the behaviour before the relation existed, and the
        #: right setting for anyone who wants the strict gate back.
        self._rule_relation = rule_relation

    @property
    def scorer(self) -> SupportScorer:
        """The support test this verifier applies.

        Exposed so the write-time guard can choose passages by the same standard
        the ship-time verifier judges them by. Two different scorers would mean
        grounding a citation on a test the verifier never applies, and then
        removing it for failing one the guard never ran.
        """

        return self._scorer

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
        assessment = assess_support(
            self._scorer,
            citation.proposition,
            record.passages,
            threshold=self._threshold,
            rule_relation=self._rule_relation,
            # The passage the retriever returned when it grounded this
            # proposition: the topicality evidence, already paid for at write
            # time and recorded on the citation.
            retrieved_passage=citation.supporting_passage,
        )

        if assessment.relation is SupportRelation.NONE:
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.REMOVED,
                reason=(
                    f"source does not support the proposition (support="
                    f"{assessment.score:.2f} < {self._threshold:.2f}); possible "
                    f"misattributed holding"
                    + (f"; {assessment.note}" if assessment.note else "")
                ),
                supporting_passage=assessment.passage,
                relation=assessment.relation.value,
            )

        if assessment.relation is SupportRelation.RULE_SUPPORT:
            # Kept, and marked for a human. The passage is the rule the claim
            # applies; whether it *reaches* these facts is the argument the paper
            # is making, and this verifier has no way to check an argument.
            return VerificationResult(
                citation_id=citation.id,
                record_id=citation.record_id,
                status=CiteStatus.VERIFIED,
                reason=(
                    # Not "topicality": that is the retriever's judgement and is
                    # not a number here. This is the entailment probability,
                    # reported because a reader should see how far short of
                    # entailment the passage fell -- 0.04 and 0.50 are both rule
                    # support and are not the same finding.
                    f"states the rule the proposition applies (entailment="
                    f"{assessment.score:.2f}, contradiction="
                    f"{assessment.contradiction:.2f}); retrieved for this "
                    f"proposition; the application to these facts is the "
                    f"author's and is NOT verified"
                ),
                supporting_passage=assessment.passage,
                relation=assessment.relation.value,
            )

        return VerificationResult(
            citation_id=citation.id,
            record_id=citation.record_id,
            status=CiteStatus.VERIFIED,
            reason=f"resolves and supports the proposition (support={assessment.score:.2f})",
            supporting_passage=assessment.passage,
            relation=assessment.relation.value,
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
