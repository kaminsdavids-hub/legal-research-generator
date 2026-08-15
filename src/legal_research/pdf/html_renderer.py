"""HTML→PDF renderer.

``render_source`` returns deterministic, law-review-styled HTML (golden-file tested).
``render`` produces a real PDF via WeasyPrint when the optional ``pdf`` extra is
installed; otherwise it writes the HTML so the pipeline still yields an artifact.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from .base import PaperDocument, PdfRenderer

_MARKER = re.compile(r"\[\^(\d+)\]")

_CSS = """
:root { --ink: #1a1a1a; --muted: #555; --rule: #ccc; }
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Georgia, serif; color: var(--ink);
       max-width: 46rem; margin: 0 auto; padding: 3rem 2rem; line-height: 1.5; }
.title-page { text-align: center; margin-bottom: 3rem; }
.title-page h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
.title-page .author { font-variant: small-caps; letter-spacing: 0.05em; }
.title-page .date { color: var(--muted); }
.disclaimer { border: 1px solid var(--rule); background: #fafafa; padding: 0.75rem 1rem;
              font-size: 0.8rem; color: var(--muted); margin: 1.5rem 0; }
.abstract { font-style: italic; margin: 1.5rem 0; }
section { margin-bottom: 1.75rem; }
section h2 { font-size: 1.15rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.25rem; }
.footnotes { border-top: 1px solid var(--rule); margin-top: 1rem; padding-top: 0.5rem;
             font-size: 0.8rem; color: var(--muted); }
.footnotes li { margin-bottom: 0.25rem; }
.toa { margin-top: 2.5rem; }
.toa h2 { font-variant: small-caps; }
.toa h3 { font-size: 1rem; margin-bottom: 0.25rem; }
.report { white-space: pre-wrap; font-family: 'SFMono-Regular', Consolas, monospace;
          font-size: 0.75rem; background: #fafafa; border: 1px solid var(--rule);
          padding: 1rem; }
sup { color: #444; }
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _para(text: str) -> str:
    return _MARKER.sub(r"<sup>\1</sup>", _esc(text))


class HtmlPdfRenderer(PdfRenderer):
    name = "html"

    def render_source(self, doc: PaperDocument) -> str:
        parts: list[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append('<html lang="en"><head><meta charset="utf-8">')
        parts.append(f"<title>{_esc(doc.title)}</title>")
        parts.append(f"<style>{_CSS}</style></head><body>")

        # Title page
        parts.append('<div class="title-page">')
        parts.append(f"<h1>{_esc(doc.title)}</h1>")
        parts.append(f'<div class="author">{_esc(doc.author)}</div>')
        if doc.date:
            parts.append(f'<div class="date">{_esc(doc.date)}</div>')
        parts.append("</div>")

        # Disclaimer on page 1
        parts.append(f'<div class="disclaimer">{_esc(doc.disclaimer)}</div>')

        if doc.thesis:
            parts.append(f'<div class="abstract"><strong>Thesis.</strong> {_esc(doc.thesis)}</div>')

        # Body sections with footnotes
        for section in doc.sections:
            parts.append("<section>")
            parts.append(f"<h2>{_esc(section.title)}</h2>")
            for para in section.paragraphs:
                parts.append(f"<p>{_para(para)}</p>")
            if section.footnotes:
                parts.append('<ol class="footnotes">')
                for fn in section.footnotes:
                    parts.append(f'<li value="{fn.number}">{_esc(fn.text)}</li>')
                parts.append("</ol>")
            parts.append("</section>")

        # Table of authorities
        if doc.table_of_authorities:
            parts.append('<div class="toa"><h2>Table of Authorities</h2>')
            for group, entries in doc.table_of_authorities.items():
                parts.append(f"<h3>{_esc(group)}</h3><ul>")
                for entry in entries:
                    parts.append(f"<li>{_esc(entry)}</li>")
                parts.append("</ul>")
            parts.append("</div>")

        # Verification report appendix
        if doc.verification_report_md:
            parts.append('<div class="appendix"><h2>Appendix: Citation Verification Report</h2>')
            parts.append(f'<div class="report">{_esc(doc.verification_report_md)}</div></div>')

        parts.append("</body></html>")
        return "\n".join(parts) + "\n"

    def render(self, doc: PaperDocument, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        source = self.render_source(doc)
        try:  # pragma: no cover - depends on optional native deps
            from weasyprint import HTML

            pdf_path = out.with_suffix(".pdf")
            HTML(string=source).write_pdf(str(pdf_path))
            return pdf_path
        except Exception:  # noqa: BLE001 - fall back to HTML artifact
            html_path = out.with_suffix(".html")
            html_path.write_text(source, encoding="utf-8")
            return html_path
