"""Pluggable support scoring for the adversarial Verifier (spec §4).

The lexical scorer must remain stable (deterministic CI), and the Verifier must
honor whatever scorer is injected — this is the seam the semantic NLI /
embedding scorers plug into on the Spark.
"""

from __future__ import annotations

from typing import Any

from legal_research.citations.corpus import Corpus
from legal_research.citations.support import (
    LexicalSupportScorer,
    build_support_scorer,
    lexical_support,
)
from legal_research.citations.verifier import CitationVerifier, support_score
from legal_research.config import Settings
from legal_research.models import Citation, CiteStatus


class ConstantScorer:
    """A stub :class:`SupportScorer` returning a fixed value; makes Rule 4 testable."""

    def __init__(self, value: float, threshold: float = 0.5) -> None:
        self.value = value
        self.threshold = threshold

    def score(self, proposition: str, passage: str) -> float:
        return self.value


def _cite(**kw: Any) -> Citation:
    base: dict[str, Any] = dict(
        id="c",
        record_id="us-425-185",
        proposition="Section 10(b) and Rule 10b-5 require a showing of scienter.",
        from_retrieval=True,
        status=CiteStatus.PENDING,
    )
    base.update(kw)
    return Citation(**base)


def test_lexical_scorer_matches_legacy_support_score() -> None:
    prop = "Section 10(b) requires a showing of scienter."
    passage = "Section 10(b) and Rule 10b-5 require a showing of scienter, an intent to deceive."
    scorer = LexicalSupportScorer()
    assert scorer.score(prop, passage) == lexical_support(prop, passage)
    assert scorer.score(prop, passage) == support_score(prop, passage)


def test_factory_defaults_to_lexical() -> None:
    assert isinstance(build_support_scorer(Settings(support_scorer="lexical")), LexicalSupportScorer)
    # Unknown / unavailable modes must fall back to lexical, never crash.
    assert isinstance(build_support_scorer(Settings(support_scorer="nonsense")), LexicalSupportScorer)


def test_factory_threshold_override() -> None:
    scorer = build_support_scorer(Settings(support_scorer="lexical", support_threshold=0.5))
    assert scorer.threshold == 0.5


def test_verifier_rejects_when_injected_scorer_is_low(corpus: Corpus) -> None:
    verifier = CitationVerifier(corpus, scorer=ConstantScorer(0.1, threshold=0.5))
    result = verifier.verify(_cite())
    assert result.status is CiteStatus.REMOVED
    assert "does not support" in result.reason


def test_verifier_accepts_when_injected_scorer_is_high(corpus: Corpus) -> None:
    verifier = CitationVerifier(corpus, scorer=ConstantScorer(0.9, threshold=0.5))
    result = verifier.verify(_cite())
    assert result.status is CiteStatus.VERIFIED
