"""Citation grounding for the Model Jury chat.

These tests pin the behaviour that stops the chat panel from inventing authority:
extraction of citation forms from prose, resolution against the corpus, support
scoring, and the repair/annotation path in :class:`MultiModelChat`.
"""

from __future__ import annotations

import pytest

from legal_research.chat_grounding import (
    ChatCitationAuditor,
    CitationKind,
    GroundingReport,
    GroundingStatus,
    extract_citations,
    format_grounding_notice,
)
from legal_research.citations.corpus import CorpusRecord, corpus_from_records
from legal_research.citations.retriever import MockRetriever
from legal_research.citations.support import LexicalSupportScorer, candidate_propositions
from legal_research.models import SourceType


@pytest.fixture()
def corpus():
    return corpus_from_records(
        [
            CorpusRecord(
                id="us-425-185",
                type=SourceType.CASE,
                title="Ernst & Ernst v. Hochfelder",
                reporter="U.S.",
                volume=425,
                page=185,
                court="U.S.",
                year=1976,
                passages=[
                    "Section 10(b) and Rule 10b-5 require a showing of scienter, that is, "
                    "an intent to deceive, manipulate, or defraud.",
                ],
            ),
            CorpusRecord(
                id="usc-15-78j-b",
                type=SourceType.STATUTE,
                title="Securities Exchange Act of 1934",
                code="15 U.S.C.",
                section="78j(b)",
                year=2018,
                passages=[
                    "It shall be unlawful for any person to use or employ, in connection "
                    "with the purchase or sale of any security, any manipulative or "
                    "deceptive device or contrivance.",
                ],
            ),
            CorpusRecord(
                id="cfr-17-240-10b-5",
                type=SourceType.REGULATION,
                title="Employment of manipulative and deceptive devices",
                code="17 C.F.R.",
                section="240.10b-5",
                year=2018,
                passages=[
                    "It shall be unlawful for any person to make any untrue statement of "
                    "a material fact in connection with the purchase or sale of any security.",
                ],
            ),
        ]
    )


@pytest.fixture()
def auditor(corpus):
    return ChatCitationAuditor(
        corpus,
        MockRetriever(corpus),
        LexicalSupportScorer(threshold=0.2),
    )


# --- extraction ---------------------------------------------------------------------


def test_extract_reporter_citation() -> None:
    found = extract_citations("The Court held in 425 U.S. 185 that scienter is required.")
    reporters = [c for c in found if c.kind is CitationKind.REPORTER]
    assert len(reporters) == 1
    assert reporters[0].text == "425 U.S. 185"
    assert reporters[0].key == "425|us|185"


def test_extract_statute_and_regulation() -> None:
    text = "Liability arises under 15 U.S.C. § 78j(b) and 17 C.F.R. § 240.10b-5."
    found = {c.kind: c for c in extract_citations(text)}
    assert found[CitationKind.STATUTE].key == "15usc|78jb"
    assert found[CitationKind.REGULATION].key == "17cfr|24010b5"


def test_extract_case_name_handles_markdown_emphasis() -> None:
    found = extract_citations("As held in *Ernst & Ernst v. Hochfelder*, scienter is required.")
    names = [c for c in found if c.kind is CitationKind.CASE_NAME]
    assert names
    assert "Hochfelder" in names[0].text


def test_case_name_inside_full_citation_is_not_double_counted() -> None:
    # "Chiarella v. United States" overlaps nothing, but the reporter span must not
    # also yield a redundant bare case-name finding for the same characters.
    found = extract_citations("See Chiarella v. United States, 445 U.S. 222 (1980).")
    kinds = [c.kind for c in found]
    assert CitationKind.REPORTER in kinds
    assert kinds.count(CitationKind.REPORTER) == 1


def test_extraction_is_deduplicated() -> None:
    text = "425 U.S. 185 is controlling. Again, 425 U.S. 185 controls."
    reporters = [c for c in extract_citations(text) if c.kind is CitationKind.REPORTER]
    assert len(reporters) == 1


def test_extract_returns_empty_for_uncited_prose() -> None:
    assert extract_citations("The controlling standard requires intent to deceive.") == []


def test_extraction_captures_surrounding_sentence() -> None:
    text = "Scienter is required. The holding in 425 U.S. 185 governs this dispute."
    reporter = next(c for c in extract_citations(text) if c.kind is CitationKind.REPORTER)
    assert "governs this dispute" in reporter.sentence


# --- resolution and auditing ----------------------------------------------------------


def test_audit_flags_fabricated_reporter_citation(auditor) -> None:
    report = auditor.audit("The Court so held in 471 U.S. 477 (1985).")
    assert report.fabricated
    assert report.fabricated[0].status is GroundingStatus.NOT_IN_CORPUS
    assert not report.clean


def test_audit_flags_fabricated_case_name(auditor) -> None:
    report = auditor.audit("See Texas Gulf Sulphur v. Wisenberg for the governing rule.")
    assert any(f.status is GroundingStatus.NOT_IN_CORPUS for f in report.findings)


def test_audit_accepts_real_citation_supporting_the_proposition(auditor) -> None:
    text = (
        "Section 10(b) and Rule 10b-5 require a showing of scienter, an intent to "
        "deceive, manipulate, or defraud, 425 U.S. 185."
    )
    report = auditor.audit(text)
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.SUPPORTED
    assert reporter.record_id == "us-425-185"


def test_audit_flags_real_citation_used_for_unrelated_proposition(auditor) -> None:
    text = "Municipal zoning variances follow a rational basis standard, 425 U.S. 185."
    report = auditor.audit(text)
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.MISATTRIBUTED
    assert reporter.record_id == "us-425-185"


def test_audit_resolves_statute_citation(auditor) -> None:
    text = (
        "It shall be unlawful for any person to use or employ any manipulative or "
        "deceptive device in connection with the purchase or sale of any security, "
        "15 U.S.C. § 78j(b)."
    )
    report = auditor.audit(text)
    statute = next(f for f in report.findings if f.kind is CitationKind.STATUTE)
    assert statute.status is GroundingStatus.SUPPORTED
    assert statute.record_id == "usc-15-78j-b"


def test_audit_resolves_case_name_by_fuzzy_title_match(auditor) -> None:
    text = (
        "Ernst & Ernst v. Hochfelder held that section 10(b) and Rule 10b-5 require a "
        "showing of scienter, an intent to deceive, manipulate, or defraud."
    )
    report = auditor.audit(text)
    case = next(f for f in report.findings if f.kind is CitationKind.CASE_NAME)
    assert case.record_id == "us-425-185"


def test_audit_of_uncited_prose_is_clean(auditor) -> None:
    report = auditor.audit("The prevailing standard requires intent to deceive.")
    assert report.clean
    assert report.findings == []
    assert "No case or statutory citations" in report.summary()


# --- authority packet -----------------------------------------------------------------


def test_authority_packet_returns_real_corpus_authority(auditor) -> None:
    packet = auditor.authority_packet("scienter requirement under Rule 10b-5", k=3)
    assert "VERIFIED AUTHORITY PACKET" in packet.text
    assert packet.labels
    assert any("Hochfelder" in label for label in packet.labels)
    assert "us-425-185" in packet.record_ids


def test_authority_packet_is_empty_when_nothing_retrieved(auditor) -> None:
    packet = auditor.authority_packet("", k=3)
    assert not packet
    assert packet.text == ""
    assert packet.labels == []
    assert packet.record_ids == frozenset()


def test_authority_packet_drops_offtopic_hits(auditor) -> None:
    # Every corpus record shares some vocabulary with a broad query; only records
    # scoring near the best hit belong in the packet.
    packet = auditor.authority_packet("scienter intent to deceive", k=6)
    assert len(packet.labels) <= 2
    assert any("Hochfelder" in label for label in packet.labels)


def test_authority_packet_ratio_can_be_relaxed(auditor) -> None:
    strict = auditor.authority_packet("securities", k=6, score_ratio=0.9)
    loose = auditor.authority_packet("securities", k=6, score_ratio=0.0, min_score=0.0)
    assert len(loose.labels) >= len(strict.labels)


# --- retrieved-but-unconfirmed ---------------------------------------------------------


def test_retrieved_authority_with_weak_support_is_unconfirmed_not_misattributed(
    auditor,
) -> None:
    """Regression: the panel cited the very authority the packet supplied and the
    entailment check still reported it as misattributed, which read to the user as
    though the model had invented a holding."""

    text = (
        "The governing decision addresses a wholly different question about municipal "
        "zoning variances and deferential review of city council judgments, "
        "425 U.S. 185."
    )
    unconfirmed = auditor.audit(text, retrieved_ids=frozenset({"us-425-185"}))
    finding = next(f for f in unconfirmed.findings if f.kind is CitationKind.REPORTER)
    assert finding.status is GroundingStatus.UNCONFIRMED
    assert not finding.is_problem
    assert unconfirmed.clean

    # The same text without the record in the packet is a genuine red flag.
    flagged = auditor.audit(text)
    finding = next(f for f in flagged.findings if f.kind is CitationKind.REPORTER)
    assert finding.status is GroundingStatus.MISATTRIBUTED
    assert finding.is_problem


def test_unconfirmed_citations_appear_in_summary(auditor) -> None:
    report = auditor.audit(
        "The decision concerns deferential review of municipal zoning variances by "
        "a city council, 425 U.S. 185.",
        retrieved_ids=frozenset({"us-425-185"}),
    )
    assert report.unconfirmed
    assert "unconfirmed" in report.summary()


def test_fabricated_citation_is_still_a_problem_even_if_packet_supplied(auditor) -> None:
    report = auditor.audit(
        "The rule comes from 471 U.S. 477.", retrieved_ids=frozenset({"us-425-185"})
    )
    assert report.fabricated
    assert not report.clean


# --- extraction edge cases -------------------------------------------------------------


def test_extract_statute_without_section_symbol() -> None:
    found = extract_citations("Liability arises under 15 U.S.C. 78j(b) in this circuit.")
    statutes = [c for c in found if c.kind is CitationKind.STATUTE]
    assert statutes
    assert statutes[0].key == "15usc|78jb"


def test_statute_section_must_start_with_a_digit() -> None:
    # Guards against swallowing the next sentence word as a section number.
    found = extract_citations("The claim arises under 15 U.S.C. and related provisions.")
    assert not [c for c in found if c.kind is CitationKind.STATUTE]


@pytest.mark.parametrize(
    "text",
    [
        "In Ernst v. Hochfelder the Court recognized the scienter element.",
        "See Ernst v. Hochfelder and its progeny for the governing rule.",
        "Compare Ernst v. Hochfelder, which controls here.",
        "Moreover Ernst v. Hochfelder held that negligence is insufficient.",
    ],
)
def test_case_name_extraction_trims_narrative_words(text: str) -> None:
    """Regression: signal words were captured as party names, so a real case
    resolved to nothing and was reported as a fabrication."""

    names = [c for c in extract_citations(text) if c.kind is CitationKind.CASE_NAME]
    assert names
    assert names[0].text == "Ernst v. Hochfelder"


def test_trimmed_case_name_resolves_to_the_corpus(auditor) -> None:
    report = auditor.audit("In Ernst v. Hochfelder the Court addressed scienter.")
    case = next(f for f in report.findings if f.kind is CitationKind.CASE_NAME)
    assert case.status is not GroundingStatus.NOT_IN_CORPUS
    assert case.record_id == "us-425-185"


def test_multiword_party_with_of_is_preserved() -> None:
    names = [
        c
        for c in extract_citations("See Central Bank of Denver v. First Interstate Bank.")
        if c.kind is CitationKind.CASE_NAME
    ]
    assert names
    assert names[0].text.startswith("Central Bank of Denver")


def test_sentence_is_not_split_on_legal_abbreviations() -> None:
    """Regression: splitting on the period in "v." truncated the proposition to
    "In Ernst v.", which then looked too short to support-test and silently
    exempted every case-name citation from misattribution checking."""

    text = (
        "In Ernst v. Hochfelder the Court recognized that reckless conduct could "
        "suffice to establish scienter."
    )
    case = next(c for c in extract_citations(text) if c.kind is CitationKind.CASE_NAME)
    assert "reckless conduct" in case.sentence


@pytest.mark.parametrize(
    "prefix",
    [
        "The claim under 15 U.S.C. 78j(b) fails because",
        "The Second Cir. holding means",
        "Acme Co. argued that",
        "Justice J. Harlan wrote that",
    ],
)
def test_abbreviations_do_not_end_sentences(prefix: str) -> None:
    text = f"{prefix} Scienter must be pleaded with particularity here, 425 U.S. 185."
    reporter = next(c for c in extract_citations(text) if c.kind is CitationKind.REPORTER)
    assert "Scienter must be pleaded" in reporter.sentence


def test_real_sentences_are_still_split() -> None:
    text = "Scienter is required. The plaintiff relies on 425 U.S. 185 for that rule."
    reporter = next(c for c in extract_citations(text) if c.kind is CitationKind.REPORTER)
    assert "Scienter is required" not in reporter.sentence


def test_case_name_misattribution_is_detected_after_splitter_fix(auditor) -> None:
    report = auditor.audit(
        "In Ernst v. Hochfelder the Court set the standard of review for municipal "
        "zoning variances and deferred to the city council's legislative judgment."
    )
    case = next(f for f in report.findings if f.kind is CitationKind.CASE_NAME)
    assert case.status is GroundingStatus.MISATTRIBUTED


def test_short_proposition_is_accepted_on_resolution_alone(auditor) -> None:
    # Too little semantic content to entailment-test; resolution is the only
    # meaningful signal, so this must not be reported as misattributed.
    report = auditor.audit("Liability also arises under 15 U.S.C. 78j(b).")
    statute = next(f for f in report.findings if f.kind is CitationKind.STATUTE)
    assert statute.status is GroundingStatus.SUPPORTED
    assert "too short to test" in statute.detail


def test_citation_tokens_do_not_count_toward_support(auditor) -> None:
    # A long but wholly unrelated claim must still be flagged even though the
    # citation string itself overlaps the record's identifying numbers.
    report = auditor.audit(
        "Municipal zoning variances are reviewed under a deferential rational basis "
        "standard that asks only whether the classification bears some conceivable "
        "relationship to a legitimate governmental purpose, 425 U.S. 185."
    )
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.MISATTRIBUTED


def test_chat_threshold_is_relaxed_relative_to_ship_time_threshold(corpus) -> None:
    scorer = LexicalSupportScorer(threshold=0.34)
    relaxed = ChatCitationAuditor(corpus, MockRetriever(corpus), scorer)
    assert relaxed._threshold == pytest.approx(0.17)

    explicit = ChatCitationAuditor(
        corpus, MockRetriever(corpus), scorer, min_support=0.42
    )
    assert explicit._threshold == pytest.approx(0.42)


def test_narrative_framed_true_citation_is_not_flagged(corpus) -> None:
    """Regression: the live panel cited the exact authority the packet supplied and
    the ship-time threshold still reported it as misattributed."""

    auditor = ChatCitationAuditor(
        corpus, MockRetriever(corpus), LexicalSupportScorer(threshold=0.34)
    )
    report = auditor.audit(
        "The Supreme Court held that negligence is insufficient and that a plaintiff "
        "must demonstrate an intent to deceive, manipulate, or defraud to satisfy the "
        "scienter element, 425 U.S. 185."
    )
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.SUPPORTED


def test_relaxed_threshold_still_rejects_unrelated_claims(corpus) -> None:
    auditor = ChatCitationAuditor(
        corpus, MockRetriever(corpus), LexicalSupportScorer(threshold=0.34)
    )
    report = auditor.audit(
        "The court applied a rational basis standard to the municipal zoning variance "
        "and deferred entirely to the legislative judgment of the city council, "
        "425 U.S. 185."
    )
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.MISATTRIBUTED


def test_candidate_propositions_include_clauses() -> None:
    candidates = candidate_propositions(
        "The Court held that scienter requires intent to deceive, and here the "
        "plaintiff would likely fail on the pleadings"
    )
    assert len(candidates) > 1
    assert any("scienter requires intent to deceive" in c for c in candidates)


def test_candidate_propositions_are_bounded() -> None:
    long_sentence = " and ".join(
        f"the {word} element requires additional proof of intent"
        for word in ("first", "second", "third", "fourth", "fifth", "sixth")
    )
    assert len(candidate_propositions(long_sentence)) <= 4


def test_citation_supported_by_one_clause_is_accepted(auditor) -> None:
    """A sentence that bundles a supported holding with unsupported commentary must
    not be reported as misattributed."""

    report = auditor.audit(
        "Section 10(b) and Rule 10b-5 require a showing of scienter, an intent to "
        "deceive, manipulate, or defraud, and the plaintiff here would still need "
        "additional discovery to identify the responsible officers, 425 U.S. 185."
    )
    reporter = next(f for f in report.findings if f.kind is CitationKind.REPORTER)
    assert reporter.status is GroundingStatus.SUPPORTED


# --- notices ---------------------------------------------------------------------------


def test_format_grounding_notice_lists_problem_citations(auditor) -> None:
    report = auditor.audit("The rule comes from 471 U.S. 477.")
    notice = format_grounding_notice(report)
    assert "Citation grounding check" in notice
    assert "471 U.S. 477" in notice


def test_format_grounding_notice_is_empty_when_clean(auditor) -> None:
    report = auditor.audit("The prevailing standard requires intent to deceive.")
    assert format_grounding_notice(report) == ""


def test_unavailable_report_reports_note() -> None:
    report = GroundingReport(available=False, note="Citation grounding unavailable (OSError).")
    assert format_grounding_notice(report) == ""
    assert "unavailable" in report.summary()
