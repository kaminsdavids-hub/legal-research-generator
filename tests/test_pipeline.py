"""End-to-end pipeline (spec §9.3, §9.5)."""

from __future__ import annotations

import re

from legal_research.models import CiteStatus
from legal_research.pipeline import LegalResearchPipeline

RAW_IDEA = (
    "Private plaintiffs may not maintain aiding-and-abetting suits under Section 10(b), "
    "and primary liability requires a showing of scienter."
)


def test_run_all_produces_shippable_paper(pipeline: LegalResearchPipeline) -> None:
    result = pipeline.run_all(RAW_IDEA, title="Secondary Liability Under Section 10(b)")
    bb = result.blackboard

    # Outline exists and starts/ends conventionally.
    assert bb.outline
    assert bb.outline[0].title == "Introduction"

    # At least one citation actually verifies against the corpus.
    verified = [c for c in bb.citations if c.status is CiteStatus.VERIFIED]
    assert verified, "expected at least one verified citation"

    # No citation remains unverified (all are VERIFIED or REMOVED).
    assert all(c.status in (CiteStatus.VERIFIED, CiteStatus.REMOVED) for c in bb.citations)
    assert bb.is_shippable()


def test_final_prose_has_no_raw_cite_tokens(pipeline: LegalResearchPipeline) -> None:
    bb = pipeline.run_all(RAW_IDEA).blackboard
    for section in bb.outline:
        assert "{{cite:" not in section.content
    # The introduction, anchored on the (verified) thesis, carries a footnote marker.
    assert "[^" in bb.outline[0].content


def test_counterargument_section_is_scaffolded(pipeline: LegalResearchPipeline) -> None:
    bb = pipeline.run_all(RAW_IDEA).blackboard
    section = next((s for s in bb.outline if s.title == "Counterargument and Rebuttal"), None)
    assert section is not None
    assert section.content
    assert "counterargument." in section.content.lower()
    assert "rebuttal." in section.content.lower()


def test_novelty_is_grounded_in_retrieved_literature(pipeline: LegalResearchPipeline) -> None:
    bb = pipeline.run_all(RAW_IDEA).blackboard
    assert bb.novelty is not None
    assert bb.novelty.grounded is True
    assert bb.novelty.distinguished_from  # named the authorities it departs from


def test_document_has_table_of_authorities(pipeline: LegalResearchPipeline) -> None:
    bb = pipeline.run_all(RAW_IDEA).blackboard
    doc = pipeline.build_document(bb)
    assert doc.table_of_authorities
    assert any(doc.table_of_authorities.values())
    # Verification report is embedded as an appendix.
    assert "Citation Verification Report" in doc.verification_report_md


def test_long_form_manuscript_hits_configured_word_range(pipeline: LegalResearchPipeline) -> None:
    bb = pipeline.run_all(RAW_IDEA).blackboard
    words = sum(len(re.findall(r"[A-Za-z0-9']+", section.content)) for section in bb.outline)
    assert words >= pipeline.settings.manuscript_target_min_words
    assert words <= pipeline.settings.manuscript_target_max_words
