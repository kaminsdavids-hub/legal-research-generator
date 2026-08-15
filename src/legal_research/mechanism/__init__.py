"""The mechanism gate: an argument may not rest on an operation it never states.

Ported from ``patentgen.blackbox`` / ``patentgen.detector``, where the rule is
that no claim may recite AI as the *mechanism* of an invention, and adapted to
legal scholarship — where mentioning AI is the point of the paper and only
mechanism position is the defect. :mod:`~legal_research.mechanism.blackbox`
carries the reasoning for both the rule and the adaptation.

Three layers, in decreasing authority:

1. :mod:`blackbox` — deterministic, no LLM, no I/O. This is the invariant.
2. :class:`~legal_research.mechanism.gate.MechanismCritic` — an LLM examiner
   that may add findings the regexes cannot see, and may never clear one.
3. :func:`~legal_research.mechanism.gate.repair_paragraph` — one bounded
   rewrite, accepted only if a re-scan shows it worked.

:func:`run_gate` drives all three over a manuscript and returns what survived.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..citations.tokens import (
    collect_tokens,
    join_paragraphs,
    rebuild_with_tokens,
    split_with_tokens,
)
from ..models import MechanismFinding
from .blackbox import Finding, ParagraphReport, Severity, scan, scan_by_paragraph, scan_text
from .gate import (
    MechanismCritic,
    combine,
    lexical_findings,
    paragraph_severity,
    repair_paragraph,
    to_records,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Finding",
    "GateReport",
    "MechanismCritic",
    "MechanismFinding",
    "ParagraphReport",
    "Severity",
    "combine",
    "lexical_findings",
    "paragraph_severity",
    "repair_paragraph",
    "run_gate",
    "scan",
    "scan_by_paragraph",
    "scan_text",
    "to_records",
]


@dataclass
class GateReport:
    """What the gate did, in terms a reader can check.

    ``repaired`` and ``unresolved`` are counted separately and deliberately: a
    stage that reports "12 findings handled" when it fixed four of them and gave
    up on eight is the reporting failure this repository keeps finding in its own
    machinery.
    """

    findings: list[MechanismFinding] = field(default_factory=list)
    repaired: int = 0
    critic_ran: bool = False
    critic_error: str = ""

    @property
    def unresolved(self) -> list[MechanismFinding]:
        return [f for f in self.findings if not f.resolved]

    @property
    def blocking(self) -> list[MechanismFinding]:
        return [
            f
            for f in self.unresolved
            if f.severity == Severity.BLACK_BOX.value
        ]

    def summary(self) -> str:
        critic = "critic on" if self.critic_ran else "critic off"
        if self.critic_error:
            critic = f"critic unavailable ({self.critic_error})"
        return (
            f"mechanism gate: {len(self.findings)} finding(s), "
            f"{self.repaired} repaired, {len(self.unresolved)} unresolved "
            f"({len(self.blocking)} black-box); {critic}"
        )


def run_gate(
    blackboard: Any,
    *,
    critic: MechanismCritic | None = None,
    repair_client: Any | None = None,
    critic_fail_open: bool = True,
    record_edit: bool = True,
) -> GateReport:
    """Scan every drafted section, attempt one repair per flagged paragraph.

    The blackboard is updated in place: repaired prose replaces the section's
    content with its citation placeholders intact, and every finding — repaired
    or not — is written to ``blackboard.mechanism_findings``.

    ``critic_fail_open`` mirrors ``grammar_chain_fail_open``: a critic that
    cannot be reached or cannot be parsed drops out with a warning rather than
    failing the run, and :attr:`GateReport.critic_error` says so, because "the
    critic found nothing" and "the critic never ran" are different facts about
    the manuscript.
    """

    report = GateReport()
    for section in getattr(blackboard, "outline", []):
        content = getattr(section, "content", "") or ""
        if not content.strip():
            continue

        prose_paragraphs, tokens_by_paragraph = split_with_tokens(content)
        found = lexical_findings(section.id, section.title, join_paragraphs(prose_paragraphs))

        if critic is not None:
            try:
                semantic = critic.findings(section.id, section.title, prose_paragraphs)
                report.critic_ran = True
                found = combine(found, semantic)
            except Exception as exc:  # noqa: BLE001 - degrade rather than crash, but say so
                if not critic_fail_open:
                    raise
                report.critic_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                logger.warning(
                    "mechanism critic unavailable for section %s (%s); the "
                    "deterministic layer still ran, but nothing checked the "
                    "assertions its vocabulary cannot see.",
                    section.id,
                    report.critic_error,
                )

        if not found:
            continue

        by_paragraph: dict[int, list[MechanismFinding]] = {}
        for finding in found:
            by_paragraph.setdefault(finding.paragraph_index, []).append(finding)

        revised_paragraphs = list(prose_paragraphs)
        repaired_here = 0
        for index, paragraph_findings in sorted(by_paragraph.items()):
            if repair_client is None or index >= len(revised_paragraphs):
                continue
            revised, improved = repair_paragraph(
                repair_client, revised_paragraphs[index], paragraph_findings
            )
            if not improved:
                continue
            revised_paragraphs[index] = revised
            repaired_here += 1
            # Only the findings the re-scan no longer reproduces are marked
            # resolved. A rewrite that fixed one of a paragraph's two defects
            # must not clear both.
            surviving = {f.term.lower() for f in scan_text(revised)}
            for finding in paragraph_findings:
                if finding.term.lower() not in surviving:
                    finding.resolved = True

        if repaired_here:
            new_content = rebuild_with_tokens(revised_paragraphs, tokens_by_paragraph)
            if collect_tokens(new_content) != collect_tokens(content):
                raise ValueError("citation token mismatch while applying mechanism repair")
            if record_edit and hasattr(blackboard, "add_edit"):
                blackboard.add_edit(
                    section_id=section.id,
                    before=content,
                    after=new_content,
                    note=(
                        "mechanism gate: named the operation in "
                        f"{repaired_here} paragraph(s)"
                    ),
                    author="Mechanism Gate",
                )
            section.content = new_content
            report.repaired += repaired_here

        report.findings.extend(found)

    _replace_findings(blackboard, report.findings)
    return report


def _replace_findings(blackboard: Any, findings: Sequence[MechanismFinding]) -> None:
    """Overwrite the recorded findings, so a re-run reports the current draft.

    Appending would accumulate the findings of every earlier draft and make a
    manuscript look worse the more it was fixed.
    """

    if hasattr(blackboard, "mechanism_findings"):
        blackboard.mechanism_findings = list(findings)
