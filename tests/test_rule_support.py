"""The rule-support relation: what it admits, and what it must still refuse.

Motivated by a probe of a live run. The paper's central claim retrieved the
EAR's published-information exclusion — the authority it rests on — and both
scorers rejected it: NLI 0.043 against 0.55, lexical 0.250 against 0.34. The
passage states a rule; the proposition applies it. Entailment is the wrong
question for that, and asking it was rejecting the right answer.

The risk of a second, looser relation is that it admits a *contrary* authority
as though it agreed, so that case is tested first.
"""

from __future__ import annotations

import pytest

from legal_research.citations.support import (
    MAX_CONTRADICTION,
    LexicalSupportScorer,
    SupportRelation,
    assess_support,
)

# The real passage, from data/corpus/openweights.jsonl.
EAR_RULE = (
    "Information is published, and so is not subject to the EAR, when it has been "
    "made available to the public without restrictions upon its further dissemination."
)
THESIS = (
    "Publishing open model weights is protected expression, and the Export "
    "Administration Regulations may not treat that publication as a deemed export."
)
DEEMED_EXPORT_RULE = (
    "A deemed export may require a license even though no item physically leaves "
    "the United States."
)
FACTS = (
    "The parties stipulated that the model was released on a public repository in "
    "March and that no license was sought."
)


class _Scorer:
    """A scripted NLI scorer: entailment and contradiction per passage."""

    threshold = 0.55

    def __init__(self, entail: dict[str, float], contra: dict[str, float] | None = None) -> None:
        self._entail = entail
        self._contra = contra or {}

    def score(self, proposition: str, passage: str) -> float:
        return self._entail.get(passage, 0.0)

    def contradiction(self, proposition: str, passage: str) -> float:
        return self._contra.get(passage, 0.0)


# --------------------------------------------------------------------------- #
# What it must refuse
# --------------------------------------------------------------------------- #
def test_a_contrary_rule_is_not_support() -> None:
    """The deemed-export rule states what the thesis argues against. This is the
    case that decides whether the relation is safe — a live run once "verified"
    exactly this pairing."""

    scorer = _Scorer({DEEMED_EXPORT_RULE: 0.04}, {DEEMED_EXPORT_RULE: 0.87})
    assessment = assess_support(scorer, THESIS, [DEEMED_EXPORT_RULE], retrieved_passage=DEEMED_EXPORT_RULE)

    assert assessment.relation is SupportRelation.NONE


def test_the_contradiction_bar_is_strict() -> None:
    just_over = _Scorer({EAR_RULE: 0.04}, {EAR_RULE: MAX_CONTRADICTION + 0.01})
    just_under = _Scorer({EAR_RULE: 0.04}, {EAR_RULE: MAX_CONTRADICTION - 0.01})

    assert assess_support(just_over, THESIS, [EAR_RULE], retrieved_passage=EAR_RULE).relation is SupportRelation.NONE
    assert (
        assess_support(just_under, THESIS, [EAR_RULE], retrieved_passage=EAR_RULE).relation is SupportRelation.RULE_SUPPORT
    )


def test_recited_facts_are_not_authority() -> None:
    """Topical, uncontradicted, and states no rule."""

    assessment = assess_support(_Scorer({FACTS: 0.05}), THESIS, [FACTS], retrieved_passage=FACTS)

    assert assessment.relation is SupportRelation.NONE


def test_a_rule_the_retriever_never_returned_is_not_tested() -> None:
    """Topicality is the retriever's judgement: a rule sitting in the same record
    that retrieval did not surface for this proposition is not the rule it
    applies."""

    unrelated = (
        "A trademark applicant must file a statement of use within six months of "
        "the notice of allowance, and the Director shall refuse late filings."
    )
    assessment = assess_support(_Scorer({unrelated: 0.01}), THESIS, [unrelated])

    assert assessment.relation is SupportRelation.NONE


def test_a_scorer_without_contradiction_cannot_offer_the_relation() -> None:
    """Lexical has no contradiction signal, so it says so instead of guessing."""

    assessment = assess_support(LexicalSupportScorer(), THESIS, [EAR_RULE], retrieved_passage=EAR_RULE)

    assert assessment.relation is SupportRelation.NONE
    assert "cannot report contradiction" in assessment.note


# --------------------------------------------------------------------------- #
# What it admits
# --------------------------------------------------------------------------- #
def test_the_rule_the_claim_applies_is_support() -> None:
    """The case this exists for: entailment 0.043, and it is still the authority."""

    scorer = _Scorer({EAR_RULE: 0.043}, {EAR_RULE: 0.02})
    assessment = assess_support(scorer, THESIS, [EAR_RULE], retrieved_passage=EAR_RULE)

    assert assessment.relation is SupportRelation.RULE_SUPPORT
    assert assessment.passage == EAR_RULE
    assert "not verified here" in assessment.note


def test_entailment_wins_when_both_hold() -> None:
    """The stronger finding must never be reported as the weaker one."""

    scorer = _Scorer({EAR_RULE: 0.91}, {EAR_RULE: 0.01})
    assessment = assess_support(scorer, THESIS, [EAR_RULE], retrieved_passage=EAR_RULE)

    assert assessment.relation is SupportRelation.ENTAILED
    assert assessment.score == pytest.approx(0.91)


def test_the_relation_can_be_switched_off() -> None:
    scorer = _Scorer({EAR_RULE: 0.043}, {EAR_RULE: 0.02})

    assessment = assess_support(scorer, THESIS, [EAR_RULE], rule_relation=False, retrieved_passage=EAR_RULE)

    assert assessment.relation is SupportRelation.NONE


# --------------------------------------------------------------------------- #
# Through the verifier: the label must reach the reader
# --------------------------------------------------------------------------- #
def _citation(proposition: str, record_id: str, passage: str = ""):
    from legal_research.models import Citation, CiteStatus

    return Citation(
        id="cite-001",
        record_id=record_id,
        proposition=proposition,
        supporting_passage=passage,
        from_retrieval=True,
        status=CiteStatus.PENDING,
    )


def _corpus_with(passage: str):
    from legal_research.citations.corpus import Corpus, CorpusRecord
    from legal_research.models import SourceType

    record = CorpusRecord(
        id="rec-1", type=SourceType.REGULATION, title="Published information and software",
        code="15 C.F.R.", section="734.7", passages=[passage],
    )
    return Corpus(records=[record])


def test_a_rule_supported_citation_is_verified_but_labelled() -> None:
    from legal_research.citations.verifier import CitationVerifier
    from legal_research.models import CiteStatus

    verifier = CitationVerifier(
        _corpus_with(EAR_RULE), scorer=_Scorer({EAR_RULE: 0.043}, {EAR_RULE: 0.02})
    )
    result = verifier.verify(_citation(THESIS, "rec-1", EAR_RULE))

    assert result.status is CiteStatus.VERIFIED
    assert result.relation == "rule_support"
    assert "is NOT verified" in result.reason


def test_an_entailed_citation_is_labelled_differently() -> None:
    from legal_research.citations.verifier import CitationVerifier

    verifier = CitationVerifier(
        _corpus_with(EAR_RULE), scorer=_Scorer({EAR_RULE: 0.91}, {EAR_RULE: 0.0})
    )
    result = verifier.verify(_citation(THESIS, "rec-1", EAR_RULE))

    assert result.relation == "entailed"
    assert "NOT verified" not in result.reason


def test_a_contrary_authority_is_still_removed_by_the_verifier() -> None:
    from legal_research.citations.verifier import CitationVerifier
    from legal_research.models import CiteStatus

    verifier = CitationVerifier(
        _corpus_with(DEEMED_EXPORT_RULE),
        scorer=_Scorer({DEEMED_EXPORT_RULE: 0.04}, {DEEMED_EXPORT_RULE: 0.87}),
    )
    result = verifier.verify(_citation(THESIS, "rec-1", DEEMED_EXPORT_RULE))

    assert result.status is CiteStatus.REMOVED
    assert result.relation == "none"


# --------------------------------------------------------------------------- #
# The guard chooses by support, not by rank
# --------------------------------------------------------------------------- #
class _Hit:
    def __init__(self, record_id: str, text: str, score: float) -> None:
        self.record_id, self.text, self.score = record_id, text, score


class _Retriever:
    """Returns the live ranking that decided nine runs: facts first, rule at 4."""

    def __init__(self, hits: list[_Hit]) -> None:
        self._hits = hits

    def search(self, query: str, k: int = 3):  # noqa: ANN201 - test double
        return self._hits[:k]


FACT_HIT = (
    "Model weights of open-weight models published to the public were not controlled "
    "under 4E091; the control reached closed weights."
)


def _guard(hits, scorer=None):
    import itertools

    from legal_research.citations.verifier import CitationGuard

    counter = itertools.count(1)
    return CitationGuard(
        _Retriever(hits), lambda: f"cite-{next(counter):03d}", scorer=scorer
    )


def test_the_guard_reaches_past_rank_one_for_the_rule() -> None:
    """Retrieval ranks by topicality; the authority the claim rests on was rank 4,
    and the guard cited rank 1 in every run before this."""

    hits = [
        _Hit("framework", FACT_HIT, 0.504),
        _Hit("ear", EAR_RULE, 0.313),
    ]
    scorer = _Scorer({EAR_RULE: 0.043, FACT_HIT: 0.02}, {EAR_RULE: 0.02, FACT_HIT: 0.0})

    citation = _guard(hits, scorer).ground(THESIS)

    assert citation.record_id == "ear"
    assert citation.supporting_passage == EAR_RULE


def test_entailment_outranks_rule_support() -> None:
    hits = [
        _Hit("ear", EAR_RULE, 0.50),
        _Hit("entails", "Model weights published openly are published information.", 0.31),
    ]
    entailing = hits[1].text
    scorer = _Scorer({EAR_RULE: 0.043, entailing: 0.88}, {EAR_RULE: 0.02, entailing: 0.0})

    assert _guard(hits, scorer).ground(THESIS).record_id == "entails"


def test_without_a_scorer_the_guard_keeps_the_top_hit() -> None:
    """The behaviour every run before this had, and the honest fallback."""

    hits = [_Hit("framework", FACT_HIT, 0.504), _Hit("ear", EAR_RULE, 0.313)]

    assert _guard(hits).ground(THESIS).record_id == "framework"


def test_nothing_supported_still_grounds_on_the_top_hit() -> None:
    """The guard's job is to refuse ungrounded assertions, not to refuse weak
    ones — the verifier is the ship-time gate and reports why."""

    hits = [_Hit("framework", FACT_HIT, 0.504)]
    scorer = _Scorer({FACT_HIT: 0.01}, {FACT_HIT: 0.0})

    assert _guard(hits, scorer).ground(THESIS).record_id == "framework"


def test_an_unretrievable_proposition_is_still_blocked() -> None:
    from legal_research.citations.verifier import GroundingError

    with pytest.raises(GroundingError):
        _guard([]).ground(THESIS)
