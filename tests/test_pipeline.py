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


def test_an_overlong_reply_is_broken_into_paragraphs() -> None:
    """Asked for one 180-260 word paragraph, an 8B model returns 500 words in a
    single block. The section then *is* one paragraph, and every stage that
    works paragraph-by-paragraph silently operates on whole sections."""

    from legal_research.agents.writer import _repaginate

    blob = " ".join(
        f"The court weighed argument number {i} and rejected it on this record."
        for i in range(1, 41)
    )
    pieces = _repaginate(blob, "the governing point", 170)

    assert len(pieces) > 1
    assert all(len(text.split()) >= 60 for text, _ in pieces)  # no orphan tail
    # One authority, cited once, at the sentence that asserts the point.
    assert [point for _, point in pieces] == ["the governing point", *([""] * (len(pieces) - 1))]
    # Nothing is lost or invented in the split.
    assert " ".join(t for t, _ in pieces).split() == blob.split()


def test_a_paragraph_within_the_ceiling_is_left_alone() -> None:
    from legal_research.agents.writer import _repaginate

    text = " ".join(f"Sentence number {i} carries the argument forward." for i in range(1, 12))
    assert _repaginate(text, "point", 170) == [(text, "point")]


def test_the_citation_formatter_keeps_paragraph_breaks() -> None:
    """It runs last, so its whitespace collapse decided the whole manuscript's
    structure: three live runs came out as one 400-550 word block per section."""

    from legal_research.agents.citation_formatter import _clean

    text = "First paragraph, with a  double space.\n\nSecond paragraph here."
    out = _clean(text)

    assert out.count("\n\n") == 1
    assert "double space." in out
    assert "  " not in out


def test_a_repeated_proposition_is_grounded_once(pipeline, blackboard) -> None:
    """Run 8 grounded 13 citations over 4 distinct propositions — the thesis
    alone six times — so the verification report asked the same question of the
    same source six times, and "13 citations, 0 verified" was a number inflated
    by repetition."""

    bb = blackboard
    bb.thesis = "Model weights are published information under the Export Administration Regulations."
    for title in ("Introduction", "Analysis", "Conclusion"):
        bb.add_section(title)

    result = pipeline.runtime.run(pipeline.writer, pipeline.context(bb))

    propositions = {c.proposition for c in bb.citations}
    assert len(bb.citations) == len(propositions)  # one citation per claim
    assert result.payload["reused_citations"] > 0
    assert result.payload["distinct_propositions"] == len(bb.citations)


def test_a_reused_citation_still_marks_every_paragraph(pipeline, blackboard) -> None:
    """Reuse must not cost a paragraph its footnote: the Bluebook builder turns
    the second occurrence into "id."/"supra"."""

    bb = blackboard
    bb.thesis = "Model weights are published information under the Export Administration Regulations."
    bb.add_section("Introduction")
    bb.add_section("Conclusion")

    pipeline.runtime.run(pipeline.writer, pipeline.context(bb))

    marked = [s for s in bb.outline if s.content and "{{cite:" in s.content]
    assert len(marked) == len([s for s in bb.outline if s.content])


def test_an_ungroundable_proposition_is_attempted_once(pipeline, blackboard) -> None:
    """Otherwise the retriever is asked the same impossible question once per
    paragraph that asserts it."""

    calls: list[str] = []
    ctx = pipeline.context(blackboard)
    original = ctx.guard.ground

    def counting_ground(proposition: str):
        calls.append(proposition)
        raise __import__("legal_research.citations.verifier", fromlist=["GroundingError"]).GroundingError(
            "no support"
        )

    ctx.guard.ground = counting_ground  # type: ignore[method-assign]
    blackboard.thesis = "A claim that nothing in the corpus can support whatsoever."
    blackboard.add_section("Introduction")
    blackboard.add_section("Conclusion")

    pipeline.writer.act(ctx)

    assert len(calls) == len(set(calls))
    assert original is not None
