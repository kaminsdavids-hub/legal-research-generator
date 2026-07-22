"""Anti-template check.

No two papers may share a boilerplate skeleton. We compute structural n-gram overlap
between papers; if two different papers exceed the threshold, that is a violation.
"""

from __future__ import annotations

import re

from .anti_ai_voice import Finding, Severity

_WORD = re.compile(r"[A-Za-z']+")

OVERLAP_THRESHOLD = 0.35


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = [t.lower() for t in _WORD.findall(text)]
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def structural_overlap(a: str, b: str, n: int = 5) -> float:
    """Jaccard overlap of n-grams between two documents (0..1)."""

    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def template_violations(
    papers: dict[str, str], threshold: float = OVERLAP_THRESHOLD, n: int = 5
) -> list[Finding]:
    """Return a FAIL finding for each pair of papers sharing boilerplate scaffolding."""

    findings: list[Finding] = []
    ids = list(papers)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            overlap = structural_overlap(papers[ids[i]], papers[ids[j]], n)
            if overlap >= threshold:
                findings.append(
                    Finding(
                        kind="shared_template",
                        message=(
                            f"papers {ids[i]!r} and {ids[j]!r} share boilerplate "
                            f"scaffolding (overlap={overlap:.2f} >= {threshold:.2f})"
                        ),
                        severity=Severity.FAIL,
                        evidence=f"{ids[i]} vs {ids[j]}",
                    )
                )
    return findings
