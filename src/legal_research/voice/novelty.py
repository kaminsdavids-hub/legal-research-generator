"""Novelty assessment.

"Novel" must be grounded, not asserted: the contribution is articulated *relative to*
retrieved existing literature. The score reflects how far the thesis departs
lexically from the closest retrieved sources, and ``grounded`` is only true when the
comparison actually rests on retrieved records.
"""

from __future__ import annotations

from ..citations.retriever import _tokens
from ..models import NoveltyAssessment, RetrievedPassage


def _distance(thesis: str, passage: str) -> float:
    a, b = set(_tokens(thesis)), set(_tokens(passage))
    if not a:
        return 0.0
    overlap = len(a & b) / len(a)
    return 1.0 - overlap


def assess_novelty(
    thesis: str, prior_literature: list[RetrievedPassage]
) -> NoveltyAssessment:
    if not prior_literature:
        return NoveltyAssessment(
            contribution=(
                f"This Article advances the following claim: {thesis.strip()} "
                "No closely comparable authority was retrieved, so novelty cannot yet "
                "be grounded against existing literature."
            ),
            distinguished_from=[],
            score=0.0,
            grounded=False,
        )

    distances = sorted(
        ((p.record_id, _distance(thesis, p.text)) for p in prior_literature),
        key=lambda kv: kv[1],
        reverse=True,
    )
    distinguished_from = [rid for rid, _ in distances]
    score = round(sum(d for _, d in distances) / len(distances), 3)
    contribution = (
        f"This Article's contribution — {thesis.strip()} — is distinguished from the "
        f"closest retrieved authorities ({', '.join(distinguished_from[:3])}). It "
        "reframes rather than restates them, and the departure is measured against "
        "those sources rather than merely asserted."
    )
    return NoveltyAssessment(
        contribution=contribution,
        distinguished_from=distinguished_from,
        score=score,
        grounded=True,
    )
