"""What the author reviews. Not the permutation -- nobody can read a permutation.

The report answers the questions an author actually has about an ordering: how
often will a reader be sent forward, how far back do the references reach, how
much did the manuscript move, and which dependencies now cross a section
boundary and so cost an explicit cross-reference.

It also flags what the optimizer could not fix. A dependency the search had to
stretch across half the paper is usually not an ordering problem at all -- it is
the argument wanting an intermediate step that was never written -- so those are
surfaced as candidate structural gaps rather than buried in a distance metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .problem import CostBreakdown, LinearizeProblem, Weights
from .segment import Segmentation

__all__ = [
    "OrderingReport",
    "ablation",
    "build_report",
    "compare_baselines",
    "reading_plan",
]

#: A dependency spanning more than this share of the manuscript is reported as a
#: structural candidate. Not a hard rule: a genuinely global premise can sit at
#: the front and be depended on at the end, and that is fine.
LONG_RANGE_SHARE = 0.5


@dataclass
class ReferenceStats:
    forward: int
    backward: int
    mean_distance: float
    max_distance: int
    total: int

    def as_dict(self) -> dict[str, float]:
        return {
            "forward_references": self.forward,
            "backward_references": self.backward,
            "mean_distance": round(self.mean_distance, 2),
            "max_distance": self.max_distance,
            "total_references": self.total,
        }


@dataclass
class OrderingReport:
    order: list[str]
    cost: CostBreakdown
    references: ReferenceStats
    cohesion: float
    cross_section_dependencies: int
    moved_nodes: int
    edit_distance: int
    new_nodes: list[str]
    long_range: list[dict[str, object]]
    sections: list[dict[str, object]] = field(default_factory=list)
    seed: int | None = None
    solver: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "seed": self.seed,
            "cost": self.cost.as_dict(),
            "references": self.references.as_dict(),
            "cohesion": round(self.cohesion, 4),
            "cross_section_dependencies": self.cross_section_dependencies,
            "stability": {
                "moved_nodes": self.moved_nodes,
                "edit_distance": self.edit_distance,
                "new_nodes": self.new_nodes,
            },
            "structural_candidates": self.long_range,
            "sections": self.sections,
            "order": self.order,
        }

    def render(self) -> str:
        lines = [
            f"ordering: {len(self.order)} nodes, solver={self.solver}, seed={self.seed}",
            f"  cost      total={self.cost.total:.2f} "
            f"(ref {self.cost.reference:.2f} / cohesion {self.cost.cohesion:.2f} / "
            f"stability {self.cost.stability:.0f})",
            f"  refs      {self.references.forward} forward, {self.references.backward} backward, "
            f"mean distance {self.references.mean_distance:.1f}, max {self.references.max_distance}",
            f"  sections  {len(self.sections)}, {self.cross_section_dependencies} "
            f"dependencies cross a boundary (each becomes an explicit cross-reference)",
            f"  stability {self.moved_nodes} node(s) moved, edit distance {self.edit_distance}"
            + (f", {len(self.new_nodes)} new" if self.new_nodes else ""),
        ]
        for section in self.sections:
            lines.append(
                f"    {section['title']}: {section['nodes']} nodes, {section['words']} words"
            )
        if self.long_range:
            lines.append("  structural candidates (long-range dependencies):")
            for item in self.long_range:
                lines.append(
                    f"    {item['dependent']} depends on {item['prerequisite']} "
                    f"across {item['distance']} positions — consider an intermediate step"
                )
        return "\n".join(lines)


def build_report(
    problem: LinearizeProblem,
    order: Sequence[str],
    *,
    segmentation: Segmentation | None = None,
    titles: Sequence[str] | None = None,
    solver: str = "",
    seed: int | None = None,
) -> OrderingReport:
    pos = {node: i for i, node in enumerate(order)}
    n = len(order)

    forward = backward = 0
    distances: list[int] = []
    for referring, referenced, _ in problem.references:
        distance = pos[referenced] - pos[referring]
        if distance == 0:
            continue
        distances.append(abs(distance))
        if distance > 0:
            forward += 1
        else:
            backward += 1

    references = ReferenceStats(
        forward=forward,
        backward=backward,
        mean_distance=(sum(distances) / len(distances)) if distances else 0.0,
        max_distance=max(distances, default=0),
        total=len(distances),
    )

    crossing = 0
    if segmentation is not None:
        for node, prerequisites in problem.prerequisites.items():
            for prerequisite in prerequisites:
                if segmentation.section_of(pos[node]) != segmentation.section_of(pos[prerequisite]):
                    crossing += 1

    before = {node: i for i, node in enumerate(problem.previous)}
    shared = [node for node in order if node in before]
    moved = sum(1 for node in shared if before[node] != pos[node])
    edit_distance = sum(abs(before[node] - pos[node]) for node in shared)
    new_nodes = [node for node in order if node not in before]

    long_range: list[dict[str, object]] = []
    for node, prerequisites in sorted(problem.prerequisites.items()):
        for prerequisite in sorted(prerequisites):
            distance = pos[node] - pos[prerequisite]
            if n > 1 and distance / n > LONG_RANGE_SHARE:
                long_range.append(
                    {
                        "dependent": node,
                        "prerequisite": prerequisite,
                        "distance": distance,
                        "share": round(distance / n, 2),
                    }
                )

    sections: list[dict[str, object]] = []
    if segmentation is not None:
        names = list(titles or [f"Part {i + 1}" for i in range(segmentation.k)])
        for index, (start, end) in enumerate(segmentation.bounds):
            sections.append(
                {
                    "title": names[index] if index < len(names) else f"Part {index + 1}",
                    "nodes": end - start,
                    "words": segmentation.words[index],
                    "first": order[start],
                }
            )

    return OrderingReport(
        order=list(order),
        cost=problem.cost(order),
        references=references,
        cohesion=problem.cohesion_cost(order),
        cross_section_dependencies=crossing,
        moved_nodes=moved,
        edit_distance=edit_distance,
        new_nodes=new_nodes,
        long_range=long_range,
        sections=sections,
        seed=seed,
        solver=solver,
    )


def reading_plan(
    order: Sequence[str],
    segmentation: Segmentation,
    titles: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """``(order, node -> section title)`` — what the renderer needs, as plain data.

    Returned as primitives rather than a renderer type so the dependency runs
    one way: the optimizer knows nothing about prose, and the renderer knows
    nothing about annealing. ``render.ReadingPlan(*reading_plan(...))`` is the
    whole integration.
    """

    names = list(titles or [f"Part {i + 1}" for i in range(segmentation.k)])
    section_of: dict[str, str] = {}
    for index, (start, end) in enumerate(segmentation.bounds):
        title = names[index] if index < len(names) else f"Part {index + 1}"
        for position in range(start, end):
            section_of[order[position]] = title
    return list(order), section_of


def compare_baselines(
    problem: LinearizeProblem,
    solved: Sequence[str],
    *,
    manual: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """The solved order against the baselines it has to beat.

    A search that cannot beat a plain topological sort is not earning its
    runtime, and reporting it on the same metrics is the only way to know. The
    author's own order is the second baseline and the one that matters: it is
    what they will keep if the optimizer offers nothing.
    """

    rows: dict[str, dict[str, float]] = {}
    candidates: dict[str, Sequence[str] | None] = {
        "dfs_topological": problem.topological_order(seed_order=[]),
        "author_manual": manual if manual is not None else (problem.previous or None),
        "optimized": solved,
    }
    for name, order in candidates.items():
        if not order or not problem.is_feasible(order):
            continue
        report = build_report(problem, order)
        rows[name] = {
            **report.cost.as_dict(),
            **report.references.as_dict(),
        }
    return rows


def ablation(problem: LinearizeProblem, order: Sequence[str]) -> dict[str, dict[str, float]]:
    """The objective with each weight zeroed, so the author sees what each buys.

    Reported on one fixed order rather than by re-solving: this answers "what is
    each term contributing to this result", which is the question. Re-solving
    per ablation answers a different one and costs three searches.
    """

    base = problem.weights
    rows = {"full": problem.cost(order).as_dict()}
    for term in ("reference", "cohesion", "stability"):
        problem.weights = _zeroed(base, term)
        rows[f"without_{term}"] = problem.cost(order).as_dict()
    problem.weights = base
    return rows


def _zeroed(weights: Weights, term: str) -> Weights:
    values: Mapping[str, object] = {
        "reference": weights.reference,
        "cohesion": weights.cohesion,
        "stability": weights.stability,
    }
    return Weights(
        reference=0.0 if term == "reference" else float(values["reference"]),
        cohesion=0.0 if term == "cohesion" else float(values["cohesion"]),
        stability=0.0 if term == "stability" else float(values["stability"]),
        forward_multiplier=weights.forward_multiplier,
        distance_exponent=weights.distance_exponent,
        edge_weights=weights.edge_weights,
    )
