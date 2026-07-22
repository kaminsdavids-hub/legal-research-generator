"""Citation-verification report (spec §4).

Each cite -> source ID/URL -> verified / removed / needs-review, with the supporting
passage. Rendered as Markdown and embedded in the PDF as a labeled appendix.
"""

from __future__ import annotations

from .. import DISCLAIMER
from ..models import Citation, CiteStatus, VerificationResult
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

    return "\n".join(lines).rstrip() + "\n"
