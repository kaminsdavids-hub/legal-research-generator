"""PDF generation (spec §6).

One :class:`PdfRenderer` interface with two backends: an HTML→PDF renderer
(deterministic, golden-file tested) and a LaTeX (law-review class) renderer. Both
include the citation-verification report as a labeled appendix and a disclaimer.
"""

from __future__ import annotations

from .base import Footnote, PaperDocument, PdfRenderer, SectionRender, build_renderer
from .html_renderer import HtmlPdfRenderer
from .latex_renderer import LatexPdfRenderer

__all__ = [
    "Footnote",
    "PaperDocument",
    "SectionRender",
    "PdfRenderer",
    "build_renderer",
    "HtmlPdfRenderer",
    "LatexPdfRenderer",
]
