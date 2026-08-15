"""End-to-end CLI demo (spec §9.5).

From a raw idea: brainstorm -> ideate -> select -> outline -> research -> draft ->
voice pass -> verify -> format -> PDF, emitting a paper directory with the
manuscript, the verification report and the rendered document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pdf.html_renderer import HtmlPdfRenderer
from .pipeline import LegalResearchPipeline
from .voice.anti_ai_voice import lint_ai_voice

RAW_IDEA = (
    "Private plaintiffs may not maintain aiding-and-abetting suits under Section 10(b), "
    "and primary liability requires a showing of scienter."
)
TITLE = "Rethinking Secondary Liability Under Section 10(b)"


def _manuscript_markdown(pipeline: LegalResearchPipeline, bb) -> str:  # type: ignore[no-untyped-def]
    doc = pipeline.build_document(bb, author="Anonymous Scholar", date="")
    lines = [f"# {doc.title}", "", f"*{doc.author}*", "", f"> {doc.disclaimer}", ""]
    if doc.thesis:
        lines += [f"**Thesis.** {doc.thesis}", ""]
    for section in doc.sections:
        lines += [f"## {section.title}", "", *section.paragraphs, ""]
        for fn in section.footnotes:
            lines.append(f"[^{fn.number}]: {fn.text}")
        if section.footnotes:
            lines.append("")
    if doc.table_of_authorities:
        lines += ["## Table of Authorities", ""]
        for group, entries in doc.table_of_authorities.items():
            lines.append(f"### {group}")
            lines += [f"- {e}" for e in entries]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="legal research generator demo")
    parser.add_argument("--idea", default=RAW_IDEA, help="raw research idea to develop")
    parser.add_argument("--title", default=TITLE, help="paper title")
    parser.add_argument("--out", default="paper", help="output directory")
    args = parser.parse_args()

    pipeline = LegalResearchPipeline()
    result = pipeline.run_all(args.idea, title=args.title)
    bb = result.blackboard

    print("=== Pipeline steps ===")
    for step in result.steps:
        print(f"[{step.runtime:9}] {step.agent:20} {step.summary}")

    print("\n=== Novelty ===")
    if bb.novelty:
        print(f"grounded={bb.novelty.grounded} score={bb.novelty.score}")
        print(bb.novelty.contribution)

    print("\n=== Citations ===")
    for c in bb.citations:
        print(f"{c.id} {c.status.value:12} {c.record_id:16} {c.proposition[:60]}")

    print("\n=== Anti-AI-voice lint (per section) ===")
    for section in bb.outline:
        if not section.content:
            continue
        report = lint_ai_voice(section.content)
        flag = "PASS" if report.passed else "FAIL:" + ",".join(f.kind for f in report.failures())
        print(f"{section.title[:40]:40} score={report.score} {flag}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manuscript.md").write_text(_manuscript_markdown(pipeline, bb), encoding="utf-8")
    (out_dir / "verification_report.md").write_text(
        pipeline.verification_report(bb), encoding="utf-8"
    )
    doc = pipeline.build_document(bb, author="Anonymous Scholar", date="")
    (out_dir / "paper.html").write_text(HtmlPdfRenderer().render_source(doc), encoding="utf-8")
    rendered = pipeline.render_pdf(bb, out_dir / "paper")

    print(f"\nShippable (no unverified cites): {bb.is_shippable()}")
    print(f"Wrote: {out_dir}/manuscript.md, {out_dir}/verification_report.md, {rendered}")


if __name__ == "__main__":
    main()
