"""One shape for what both modules hand back: text a person reads, JSON a run keeps.

The two CLIs were each printing their own findings, which meant two formats,
two places to change a heading, and no shared answer to the question every
artifact has to answer — *what produced this, and how would I get it again?*

Every report carries a reproduction block: evaluator hashes, epoch, seed, and
the command. Without it an archive is a list of fact patterns whose provenance
is a memory, and the first thing anyone asks about a surprising finding is
whether it came from the rules they are looking at now.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import EvaluatorStamp

__all__ = [
    "Report",
    "Section",
    "divergence_report",
    "holdout_notice",
    "portfolio_report",
]


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    #: Machine-readable counterpart. The text is for a person; this is what a
    #: later run diffs against.
    data: Any = None

    def render(self) -> str:
        body = "\n".join(f"  {line}" for line in self.lines)
        return f"{self.title}\n{body}" if body else self.title


@dataclass
class Report:
    title: str
    stamp: EvaluatorStamp
    sections: list[Section] = field(default_factory=list)
    #: Findings the author must act on before trusting the rest. Rendered first
    #: and never collapsed into the body: a report whose caveats are at the
    #: bottom is a report whose caveats are unread.
    caveats: list[str] = field(default_factory=list)

    def add(self, title: str, lines: Sequence[str] = (), data: Any = None) -> Section:
        section = Section(title=title, lines=list(lines), data=data)
        self.sections.append(section)
        return section

    def caveat(self, text: str) -> None:
        if text not in self.caveats:
            self.caveats.append(text)

    def render(self) -> str:
        blocks = [self.title, "=" * len(self.title)]
        if self.caveats:
            blocks.append("\nREAD FIRST")
            blocks.extend(f"  ! {caveat}" for caveat in self.caveats)
        for section in self.sections:
            blocks.append("")
            blocks.append(section.render())
        blocks.append("")
        blocks.append(self._reproduction())
        return "\n".join(blocks)

    def _reproduction(self) -> str:
        return "\n".join(
            [
                "reproduction",
                f"  schema  {self.stamp.schema_digest}",
                f"  rules   {self.stamp.rules_digest}",
                f"  epoch   {self.stamp.epoch}",
                f"  seed    {self.stamp.seed}",
                f"  command {self.stamp.recipe or '(not recorded)'}",
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "stamp": self.stamp.as_dict(),
            "caveats": self.caveats,
            "sections": {s.title: s.data for s in self.sections if s.data is not None},
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Divergence
# --------------------------------------------------------------------------- #
def divergence_report(
    archive: Any,
    collision: Any | None = None,
    *,
    top: int = 8,
    corpus_size: int = 0,
    verified_codings: int = 0,
) -> Report:
    """Precedent collisions first, hypotheticals second, coverage last.

    The ordering is the argument. A hypothetical is a thought experiment an
    author can decline to engage with; a mishandled precedent is a decided case
    their framework gets wrong, and burying it under a list of invented edge
    cases would be presenting the weaker finding as the headline.
    """

    report = Report(title="Divergence: where the readings split", stamp=archive.stamp)

    if corpus_size and not verified_codings:
        report.caveat(
            f"{corpus_size} coded case(s), 0 verified. No precedent collision can be "
            "reported — the force of that finding is that a real decision is "
            "misclassified, which is worth nothing if the coding was guessed. "
            "Hand-code and verify a seed set (30-60 leading cases) to switch this on."
        )

    if collision is not None:
        report.add(
            "precedent collisions",
            [collision.summary()],
            data={"summary": collision.summary()},
        )
        for precedent in collision.precedents[:top]:
            lines = [
                f"held: {precedent.held}",
                f"readings: {', '.join(precedent.verdicts)}",
                f"distance {precedent.distance}, coded by {precedent.coded_by or 'unknown'}",
            ]
            if precedent.contradicting:
                lines.append(f"contradicted by: {', '.join(precedent.contradicting)}")
            report.add(
                f"  MISHANDLED PRECEDENT — {precedent.name} ({precedent.citation})",
                lines,
                data=precedent.__dict__,
            )
        if collision.unverified_candidates:
            report.caveat(
                f"{len(collision.unverified_candidates)} coded case(s) matched a hard "
                f"pattern but are unverified codings, so no collision is claimed: "
                f"{', '.join(collision.unverified_candidates)}"
            )
        if collision.too_undetermined:
            report.caveat(
                f"{len(collision.too_undetermined)} verified coding(s) matched a hard "
                f"pattern only because undetermined axes waive the comparison, so no "
                f"collision is claimed: {', '.join(collision.too_undetermined)}. Each is "
                f"a lead: settle what the record leaves open and it may govern."
            )
        if collision.off_schema:
            report.caveat(
                f"{len(collision.off_schema)} coding(s) matched but sit largely outside "
                f"this schema — too many axes do not arise for them: "
                f"{', '.join(collision.off_schema)}. Not a lead: reading further will "
                f"not fill those axes in, so the question is whether the corpus should "
                f"carry these cases at all."
            )

    elites = list(archive.elites)[:top]
    report.add(
        f"hard cases ({len(archive.elites)} in the archive)",
        [
            f"disagreement={e.objectives['disagreement']:.2f} "
            f"brittleness={e.objectives['brittleness']:.0f}  "
            + ", ".join(f"{k}={v}" for k, v in sorted(e.genome.items()))
            for e in elites
        ],
        data=[{"genome": e.genome, "objectives": e.objectives} for e in elites],
    )

    unanimous = [e for e in archive.elites if e.objectives["disagreement"] == 0.0]
    if len(unanimous) == len(archive.elites) and archive.elites:
        report.caveat(
            "every cell in the archive is unanimous: the readings never split "
            "anywhere the search reached. Either the predicates encode the same "
            "reading, or the feasible region excludes the interesting cases."
        )
    return report


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
def portfolio_report(
    front: Sequence[Any],
    strategies: Sequence[Any],
    *,
    stamp: EvaluatorStamp,
    excluded: Mapping[str, Sequence[str]] | None = None,
    thin: Sequence[Mapping[str, Any]] = (),
) -> Report:
    report = Report(title="Portfolio: the citation front", stamp=stamp)

    if excluded:
        report.add(
            "excluded for negative citator treatment",
            [f"{pid}: {', '.join(ids)}" for pid, ids in sorted(excluded.items())],
            data={k: list(v) for k, v in excluded.items()},
        )

    report.add(f"front: {len(front)} portfolio(s)")

    checks: set[str] = set()
    for strategy in strategies:
        scores = strategy.portfolio.scores
        lines = [
            strategy.why,
            f"{scores.count} citations, binding {scores.binding:.2f}, "
            f"recency {scores.recency:.2f}, concentration {scores.concentration:.2f}, "
            f"spread {scores.spread:.2f}",
            *[f"{pid} <- {aid}" for pid, aid in strategy.portfolio.selection],
        ]
        if strategy.portfolio.manual_citator_checks:
            lines.append(
                "MANUAL_CITATOR_CHECK: "
                + ", ".join(strategy.portfolio.manual_citator_checks)
            )
            checks.update(strategy.portfolio.manual_citator_checks)
        report.add(strategy.name.upper(), lines, data=strategy.portfolio.as_dict())

    if checks:
        report.caveat(
            f"{len(checks)} selected authority(ies) have no citator data and were "
            f"NOT assumed good: {', '.join(sorted(checks))}. Check their standing "
            "before filing."
        )

    # The choice among the front is the author's. Stated in the report rather
    # than only in the CLI, because the report is what survives the session.
    report.add(
        "how to read this",
        [
            "The front is not ranked. Each portfolio is optimal on some trade-off",
            "between footnote economy, binding weight, recency, concentration and",
            "spread; no weighting over those represents an author's real preference.",
            "Your pick is an editorial decision and is recorded as [stated].",
        ],
    )

    if thin:
        report.add(
            f"thin coverage ({len(thin)}) — raised for the NEXT epoch",
            [f"{f['proposition']}: {', '.join(f['reasons'])}" for f in thin],
            data=list(thin),
        )
    return report


# --------------------------------------------------------------------------- #
# Held-out validation
# --------------------------------------------------------------------------- #
def holdout_notice(tuned: Mapping[str, float], holdout: Sequence[Any]) -> str:
    """What to say about any weight a search chose rather than an author.

    Nothing in either module tunes a weight by search today: the objective
    weights, the half-life and the thresholds are all author-set, which is why
    they are in config and not in a fitness function. If that changes, a weight
    fitted on this manuscript and reported on this manuscript is a weight fitted
    to noise, and this is the sentence that has to appear beside it.
    """

    if not tuned:
        return (
            "No weight or threshold here was tuned by search; all are author-set, "
            "so there is nothing to hold out."
        )
    if not holdout:
        return (
            f"WARNING: {len(tuned)} tuned parameter(s) ({', '.join(sorted(tuned))}) "
            "with no held-out set. These are fitted to this manuscript and their "
            "reported performance on it means nothing. Validate on a disjoint set "
            "before quoting any figure that depends on them."
        )
    return (
        f"{len(tuned)} tuned parameter(s) validated on {len(holdout)} held-out "
        f"item(s) disjoint from the tuning set."
    )
