"""Citation-verification report (spec §4).

Each cite -> source ID/URL -> verified / removed / needs-review, with the supporting
passage. Rendered as Markdown and embedded in the PDF as a labeled appendix.
"""

from __future__ import annotations

from .. import DISCLAIMER
from ..models import Citation, CiteStatus, MechanismFinding, VerificationResult
from .corpus import Corpus

_STATUS_LABEL = {
    CiteStatus.VERIFIED: "VERIFIED",
    CiteStatus.REMOVED: "REMOVED",
    CiteStatus.NEEDS_REVIEW: "NEEDS REVIEW",
    CiteStatus.PENDING: "PENDING",
}


def build_verification_report(
    citations: list[Citation],
    results: list[VerificationResult],
    corpus: Corpus,
    mechanism_findings: list[MechanismFinding] | None = None,
) -> str:
    by_id = {r.citation_id: r for r in results}
    verified = [c for c in citations if c.status == CiteStatus.VERIFIED]
    removed = [c for c in citations if c.status == CiteStatus.REMOVED]
    review = [c for c in citations if c.status == CiteStatus.NEEDS_REVIEW]

    lines: list[str] = []
    lines.append("# Citation Verification Report")
    lines.append("")
    lines.append(
        f"**Summary:** {len(verified)} verified, {len(removed)} removed, "
        f"{len(review)} needs-review, of {len(citations)} total."
    )
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")

    for citation in citations:
        result = by_id.get(citation.id)
        status = _STATUS_LABEL[citation.status]
        record = corpus.get(citation.record_id)
        url = record.url if record else ""
        lines.append(f"## {citation.id} — {status}")
        lines.append("")
        lines.append(f"- **Proposition:** {citation.proposition}")
        lines.append(f"- **Source record:** `{citation.record_id}`" + (f" — {url}" if url else ""))
        lines.append(f"- **From retrieval:** {'yes' if citation.from_retrieval else 'NO (blocked)'}")
        if citation.quote:
            lines.append(f"- **Quote:** \u201c{citation.quote}\u201d")
        if result:
            lines.append(f"- **Finding:** {result.reason}")
            if result.supporting_passage:
                lines.append(f"- **Supporting passage:** {result.supporting_passage}")
        lines.append("")

    lines.extend(_mechanism_section(mechanism_findings or []))

    return "\n".join(lines).rstrip() + "\n"


def _mechanism_section(findings: list[MechanismFinding]) -> list[str]:
    """Unresolved mechanism findings, appended to the same appendix.

    They belong here rather than in a separate document because they are the
    same kind of fact as an unverified citation: a place where the manuscript
    asserts something the machinery could not stand behind. A reader auditing
    the paper should not have to know that the two gates were written months
    apart to find both sets of results.

    Resolved findings are omitted. What the repair pass fixed is in the edit
    history; what it could not fix is what a reader has to act on.
    """

    unresolved = [f for f in findings if not f.resolved]
    if not unresolved:
        return []

    black_box = [f for f in unresolved if f.severity == "BLACK_BOX"]
    lines = ["## Mechanism findings", ""]
    lines.append(
        f"**Summary:** {len(unresolved)} passage(s) rest on an operation the text "
        f"does not state — {len(black_box)} with no supporting detail at all. "
        "These are evidence for the author, not a verdict on the paper: a regex "
        "cannot tell a missing mechanism from one described in a way it does not "
        "recognise."
    )
    lines.append("")
    for finding in unresolved:
        where = finding.section_title or finding.section_id
        lines.append(
            f"- **[{finding.severity}]** {where}, paragraph "
            f"{finding.paragraph_index + 1} ({finding.source}) — {finding.reason}"
        )
        if finding.excerpt:
            lines.append(f"  - {finding.excerpt}")
    lines.append("")
    return lines
