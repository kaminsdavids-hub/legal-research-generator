"""PDF document model and renderer interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .. import DISCLAIMER


class Footnote(BaseModel):
    number: int
    text: str


class SectionRender(BaseModel):
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    footnotes: list[Footnote] = Field(default_factory=list)


class PaperDocument(BaseModel):
    """A fully-assembled paper ready to render. Deterministic by construction: no
    field is auto-populated from the clock, so identical inputs render identically."""

    title: str
    author: str = "Anonymous Scholar"
    date: str = ""
    thesis: str = ""
    sections: list[SectionRender] = Field(default_factory=list)
    table_of_authorities: dict[str, list[str]] = Field(default_factory=dict)
    verification_report_md: str = ""
    disclaimer: str = DISCLAIMER


@runtime_checkable
class PdfRenderer(Protocol):
    """Backends implement a source-format renderer plus a file writer."""

    name: str

    def render_source(self, doc: PaperDocument) -> str:
        """Return the deterministic intermediate source (HTML or LaTeX)."""

    def render(self, doc: PaperDocument, out_path: str | Path) -> Path:
        """Write the paper to ``out_path`` (PDF if the backend is available, else the
        intermediate source) and return the path actually written."""


def build_renderer(kind: str = "html") -> PdfRenderer:
    from .html_renderer import HtmlPdfRenderer
    from .latex_renderer import LatexPdfRenderer

    if kind == "latex":
        return LatexPdfRenderer()
    return HtmlPdfRenderer()
