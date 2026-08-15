"""LaTeX (law-review class) renderer.

``render_source`` returns a deterministic ``.tex`` document with a title page,
footnote citations, a table of authorities and a verification appendix. ``render``
compiles it with ``tectonic`` or ``latexmk`` when available; otherwise it writes the
``.tex`` so the artifact still exists.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import PaperDocument, PdfRenderer

_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(text: str) -> str:
    return "".join(_SPECIAL.get(ch, ch) for ch in text)


class LatexPdfRenderer(PdfRenderer):
    name = "latex"

    def render_source(self, doc: PaperDocument) -> str:
        lines: list[str] = []
        lines.append(r"\documentclass[12pt]{article}")
        lines.append(r"\usepackage[margin=1in]{geometry}")
        lines.append(r"\usepackage{setspace}\doublespacing")
        lines.append(r"\usepackage{times}")
        lines.append(r"\title{" + _tex_escape(doc.title) + "}")
        lines.append(r"\author{" + _tex_escape(doc.author) + "}")
        lines.append(r"\date{" + (_tex_escape(doc.date) if doc.date else "") + "}")
        lines.append(r"\begin{document}")
        lines.append(r"\maketitle")
        lines.append(r"\thispagestyle{plain}")

        # Disclaimer on page 1
        lines.append(r"\begin{quote}\small\itshape " + _tex_escape(doc.disclaimer) + r"\end{quote}")

        if doc.thesis:
            lines.append(r"\begin{abstract}" + _tex_escape(doc.thesis) + r"\end{abstract}")

        for section in doc.sections:
            lines.append(r"\section*{" + _tex_escape(section.title) + "}")
            footnotes = {fn.number: fn.text for fn in section.footnotes}
            for para in section.paragraphs:
                text = _tex_escape(para)
                # Attach footnotes that belong to this section at paragraph end.
                for number, fn_text in footnotes.items():
                    marker = f"[^{number}]"
                    if marker in para:
                        text = text.replace(_tex_escape(marker), r"\footnote{" + _tex_escape(fn_text) + "}")
                lines.append(text)
                lines.append("")

        if doc.table_of_authorities:
            lines.append(r"\section*{Table of Authorities}")
            for group, entries in doc.table_of_authorities.items():
                lines.append(r"\subsection*{" + _tex_escape(group) + "}")
                lines.append(r"\begin{itemize}")
                for entry in entries:
                    lines.append(r"\item " + _tex_escape(entry))
                lines.append(r"\end{itemize}")

        if doc.verification_report_md:
            lines.append(r"\section*{Appendix: Citation Verification Report}")
            lines.append(r"\begin{verbatim}")
            lines.append(doc.verification_report_md)
            lines.append(r"\end{verbatim}")

        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"

    def render(self, doc: PaperDocument, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tex_path = out.with_suffix(".tex")
        tex_path.write_text(self.render_source(doc), encoding="utf-8")

        engine = shutil.which("tectonic") or shutil.which("latexmk")
        if engine is None:  # pragma: no cover - depends on host toolchain
            return tex_path
        try:  # pragma: no cover - depends on host toolchain
            if engine.endswith("tectonic"):
                subprocess.run([engine, str(tex_path)], check=True, cwd=str(out.parent))
            else:
                subprocess.run(
                    [engine, "-pdf", "-interaction=nonstopmode", str(tex_path)],
                    check=True,
                    cwd=str(out.parent),
                )
            pdf_path = out.with_suffix(".pdf")
            return pdf_path if pdf_path.exists() else tex_path
        except (subprocess.CalledProcessError, OSError):
            return tex_path
