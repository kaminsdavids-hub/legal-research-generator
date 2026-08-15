"""Deterministic HTML render golden-file test (spec §6, §9).

Set ``LRG_REGEN=1`` to regenerate the golden file after an intentional change.
"""

from __future__ import annotations

import os
from pathlib import Path

from legal_research.pdf.base import Footnote, PaperDocument, SectionRender
from legal_research.pdf.html_renderer import HtmlPdfRenderer

GOLDEN = Path(__file__).parent / "golden" / "paper.html"


def _doc() -> PaperDocument:
    return PaperDocument(
        title="Rethinking Secondary Liability Under Section 10(b)",
        author="Anonymous Scholar",
        date="2026",
        thesis="Private plaintiffs may not maintain aiding-and-abetting suits under Section 10(b).",
        sections=[
            SectionRender(
                title="Introduction",
                paragraphs=[
                    "The rule is narrow. Central Bank forecloses secondary liability.[^1]",
                ],
                footnotes=[
                    Footnote(
                        number=1,
                        text="Central Bank of Denver, N.A. v. First Interstate Bank of Denver, N.A., 511 U.S. 164 (1994).",
                    )
                ],
            ),
            SectionRender(
                title="I. The Scienter Requirement",
                paragraphs=["Scienter polices the boundary between fraud and negligence.[^2]"],
                footnotes=[
                    Footnote(
                        number=2, text="Ernst & Ernst v. Hochfelder, 425 U.S. 185 (1976)."
                    )
                ],
            ),
        ],
        table_of_authorities={
            "Cases": [
                "Central Bank of Denver, N.A. v. First Interstate Bank of Denver, N.A., 511 U.S. 164 (1994)",
                "Ernst & Ernst v. Hochfelder, 425 U.S. 185 (1976)",
            ]
        },
        verification_report_md="# Citation Verification Report\n\n**Summary:** 2 verified.\n",
    )


def test_html_render_matches_golden() -> None:
    rendered = HtmlPdfRenderer().render_source(_doc())
    if os.environ.get("LRG_REGEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(rendered, encoding="utf-8")
    assert GOLDEN.exists(), "golden file missing; run with LRG_REGEN=1 to create it"
    assert rendered == GOLDEN.read_text(encoding="utf-8")


def test_render_is_deterministic() -> None:
    r = HtmlPdfRenderer()
    assert r.render_source(_doc()) == r.render_source(_doc())
