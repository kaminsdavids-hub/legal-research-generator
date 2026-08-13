"""A citation must be verified against a claim, not a topic. REMEDIATION §22."""

from __future__ import annotations

from legal_research.agents.legal_researcher import is_assertable

#: Verbatim from a live pipeline run whose 122 citations were all removed.
OBSERVED_TITLES = [
    "Analyzing The Impact of Open Model Weights on Regulatory Compliance Under "
    "the U.S. Export Administration Regulation (EAR): A Case Study On Deep Learning",
    "Investigating Potential Security Threats Associated With Sharing AI Models "
    "Across International Borders: Implications For Open Model Weight Release Policy",
]

OBSERVED_CLAIMS = [
    "Publishing open model weights is not a deemed export under the EAR, because "
    "the published-information exclusion removes material already in the public domain.",
    "The published-information exclusion applies to material already in the public domain.",
]


def test_the_titles_that_removed_122_citations_are_not_assertable() -> None:
    for title in OBSERVED_TITLES:
        assert not is_assertable(title), title[:60]


def test_real_claims_are_assertable() -> None:
    for claim in OBSERVED_CLAIMS:
        assert is_assertable(claim), claim[:60]


def test_a_gerund_opener_marks_a_topic() -> None:
    assert not is_assertable("Exploring the boundaries of the exclusion")
    assert not is_assertable("Towards a theory of published information")


def test_a_case_study_suffix_marks_a_topic_even_without_a_gerund() -> None:
    assert not is_assertable("Open weights and the EAR: a case study")


def test_empty_text_is_not_assertable() -> None:
    assert not is_assertable("")
    assert not is_assertable("   ")


def test_a_claim_that_merely_mentions_analysis_is_still_a_claim() -> None:
    """Conservative by design: rejecting real claims costs coverage, and the
    check only rejects the shapes the Ideator actually produces.
    """
    assert is_assertable("Courts have rejected the analysis the agency proposes.")


#: Verbatim from the run that "verified" 9 citations. All nine were this one
#: proposition, and it is a title -- it passed only because the opener list held
#: the gerund "analyzing" and not the imperative "analyze".
OBSERVED_IMPERATIVE_TITLE = (
    "Analyze how AI-generated model weights and training data could be treated "
    "under existing law"
)


def test_the_imperative_title_that_faked_nine_verifications_is_rejected() -> None:
    assert not is_assertable(OBSERVED_IMPERATIVE_TITLE)


def test_imperative_and_gerund_forms_are_both_topics() -> None:
    for stem in ("Analyze", "Analyzing", "Investigate", "Examine", "Evaluate", "Review"):
        assert not is_assertable(f"{stem} the scope of the exclusion"), stem


def test_an_interrogative_opener_is_a_topic_not_a_claim() -> None:
    for opener in ("Whether", "How", "Why", "What"):
        assert not is_assertable(f"{opener} the exclusion reaches model weights")


def test_a_claim_containing_a_topic_word_is_still_a_claim() -> None:
    """Only the *opening* marks a topic; the check must not reject prose that
    merely uses the vocabulary.
    """
    assert is_assertable("Courts have rejected the analysis the agency proposes.")
    assert is_assertable("The record does not support review of that question.")
