"""The cost model and the feasible region for manuscript ordering.

Ordering the argument graph is what makes "backward references only when
necessary" real: the renderer emits a cross-reference exactly when it crosses a
dependency edge into another section, so the *order* decides how many references
exist and how far a reader must reach back. Choosing it is a
precedence-constrained minimum linear arrangement, which is NP-hard, hence the
metaheuristics in :mod:`sa` and :mod:`brkga`. Everything exact lives here or in
:mod:`segment`.

Two places where the specification and this codebase disagreed, resolved in
favour of the codebase and recorded here rather than silently:

**Dependency direction.** The brief says "for every DEPENDS_ON edge (A → B),
pos(A) < pos(B)". This graph's edges run the other way: ``render.order_nodes``
states "A depends on B: B must be read first", so the *target* is the
prerequisite. Following the brief literally would place every claim before the
material it rests on -- the exact inversion of what the constraint is for. So:
for ``Edge(A, B, DEPENDS_ON)``, ``pos(B) < pos(A)``.

**Which direction is "forward".** The brief defines a forward reference as
``pos(v) < pos(u)`` and glosses it as "the reader meets the referring node
first", but those describe opposite arrangements. The intent is unambiguous from
the rest of the paragraph -- backward references are "cheap when near", so
backward must be the case where the referenced node was already read. Forward
here means the referring node comes first and points at material the reader has
not reached: ``pos(u) < pos(v)``, and it is the expensive one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from modules.maieutic.graph import ArgumentGraph, EdgeType, NodeType

__all__ = [
    "CostBreakdown",
    "LinearizeProblem",
    "PrecedenceViolation",
    "Weights",
    "build_problem",
]


class PrecedenceViolation(Exception):
    """Raised when an order or a graph breaks a hard constraint.

    Loud on purpose. The integration gate already rejects DEPENDS_ON cycles, so
    a cycle reaching this module means something upstream failed, and silently
    repairing it would produce a reading order whose guarantees no longer hold
    while still looking like one that does.
    """


#: Per-edge-type reference weights. SUPPORTS and ATTACKS bind tightly: a
#: supporting reason read far from what it supports, or an objection read far
#: from what it attacks, is the reader holding a thread across pages.
#: DISTINGUISHES is looser -- distinguishing a case is a local move that reads
#: fine anywhere after the case appears.
DEFAULT_EDGE_WEIGHTS: Mapping[EdgeType, float] = {
    EdgeType.SUPPORTS: 1.0,
    EdgeType.ATTACKS: 1.0,
    EdgeType.QUALIFIES: 0.8,
    EdgeType.IMPLIES: 0.6,
    EdgeType.DISTINGUISHES: 0.4,
}


@dataclass(frozen=True)
class Weights:
    """The three costs, and the shape of the reference penalty.

    Exposed in config and **not** tuned against a learned critic: a critic that
    scores manuscripts would be scoring the machinery's own preferences, and the
    order is the author's structural choice, not something to be optimised
    toward a model's taste.
    """

    reference: float = 1.0
    cohesion: float = 1.0
    #: High by design. Without it, adding one node reshuffles the manuscript and
    #: the author cannot work iteratively -- the order has to be something they
    #: can accept once and keep. Reordering should have to buy real reference
    #: savings to be worth the churn.
    stability: float = 3.0
    #: How much worse a forward reference is than a backward one of equal span.
    forward_multiplier: float = 6.0
    #: Superlinear, so distance hurts more than proportionally: two references
    #: of distance 5 read far better than one of distance 10.
    distance_exponent: float = 1.5
    edge_weights: Mapping[EdgeType, float] = field(
        default_factory=lambda: dict(DEFAULT_EDGE_WEIGHTS)
    )


@dataclass(frozen=True)
class CostBreakdown:
    reference: float
    cohesion: float
    stability: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reference": round(self.reference, 4),
            "cohesion": round(self.cohesion, 4),
            "stability": round(self.stability, 4),
            "total": round(self.total, 4),
        }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else num / (na * nb)


@dataclass
class LinearizeProblem:
    """Everything an ordering is scored and constrained by."""

    node_ids: list[str]
    #: id -> ids that must appear before it (DEPENDS_ON targets).
    prerequisites: dict[str, set[str]]
    #: id -> ids that must appear after it. The inverse index, kept because the
    #: feasible window needs both ends and recomputing one from the other in the
    #: inner loop is the difference between a usable annealer and a slow one.
    dependents: dict[str, set[str]]
    #: (referring, referenced, weight) for every non-DEPENDS_ON edge.
    references: list[tuple[str, str, float]]
    #: id -> absolute position. Never violated.
    pins: dict[str, int]
    #: (objection, reply) pairs that must stay within ``adjacency_tolerance``.
    pairs: list[tuple[str, str]]
    embeddings: dict[str, Sequence[float]]
    words: dict[str, int]
    #: The author's last accepted order, or empty. Positions in it anchor
    #: ``stability_cost``; nodes absent from it are new and cost nothing to place.
    previous: list[str]
    weights: Weights = field(default_factory=Weights)
    adjacency_tolerance: int = 1

    # ------------------------------------------------------------------ #
    # Feasibility
    # ------------------------------------------------------------------ #
    def assert_acyclic(self) -> None:
        """Fail loudly if the DEPENDS_ON subgraph is not a DAG."""

        colour: dict[str, int] = dict.fromkeys(self.node_ids, 0)

        def walk(node: str, path: list[str]) -> None:
            colour[node] = 1
            for prerequisite in sorted(self.prerequisites.get(node, ())):
                if colour.get(prerequisite) == 1:
                    cycle = path[path.index(prerequisite):] + [prerequisite]
                    raise PrecedenceViolation(
                        "DEPENDS_ON cycle reached the optimizer: "
                        + " -> ".join(cycle)
                        + ". The integration gate should have rejected this; "
                        "ordering it would produce a manuscript whose reading "
                        "guarantees are false."
                    )
                if colour.get(prerequisite) == 0:
                    walk(prerequisite, [*path, prerequisite])
            colour[node] = 2

        for node in self.node_ids:
            if colour[node] == 0:
                walk(node, [node])

    def violations(self, order: Sequence[str]) -> list[str]:
        """Every hard-constraint breach in ``order``, as readable strings."""

        problems: list[str] = []
        if sorted(order) != sorted(self.node_ids):
            problems.append("order is not a permutation of the graph's nodes")
            return problems

        pos = {node: i for i, node in enumerate(order)}
        for node, prerequisites in self.prerequisites.items():
            for prerequisite in prerequisites:
                if pos[prerequisite] >= pos[node]:
                    problems.append(
                        f"precedence: {node} depends on {prerequisite}, which must be read first"
                    )
        for node, position in self.pins.items():
            if pos[node] != position:
                problems.append(f"pin: {node} must sit at position {position}, found {pos[node]}")
        for objection, reply in self.pairs:
            gap = pos[reply] - pos[objection]
            if not 0 < gap <= self.adjacency_tolerance + 1:
                problems.append(
                    f"adjacency: reply {reply} must follow objection {objection} within "
                    f"{self.adjacency_tolerance + 1} position(s), found gap {gap}"
                )
        return problems

    def is_feasible(self, order: Sequence[str]) -> bool:
        return not self.violations(order)

    def window(self, order: Sequence[str], index: int) -> tuple[int, int]:
        """Positions the node at ``index`` may move to without breaking precedence.

        ``[max(prerequisite positions) + 1, min(dependent positions) - 1]``.
        Proposing only inside this window is what lets the annealer skip a repair
        step: every candidate it builds is already precedence-feasible, and
        repair is where a constrained metaheuristic usually spends its time.
        """

        pos = {node: i for i, node in enumerate(order)}
        node = order[index]
        lo = 0
        for prerequisite in self.prerequisites.get(node, ()):  # noqa: SIM118
            lo = max(lo, pos[prerequisite] + 1)
        hi = len(order) - 1
        for dependent in self.dependents.get(node, ()):  # noqa: SIM118
            hi = min(hi, pos[dependent] - 1)
        return lo, hi

    # ------------------------------------------------------------------ #
    # Cost
    # ------------------------------------------------------------------ #
    def reference_cost(self, pos: Mapping[str, int]) -> float:
        total = 0.0
        for referring, referenced, weight in self.references:
            distance = pos[referenced] - pos[referring]
            if distance == 0:
                continue
            span = abs(distance) ** self.weights.distance_exponent
            # distance > 0: the referenced node comes later, so the reader is
            # pointed at material they have not read. See the module docstring.
            direction = self.weights.forward_multiplier if distance > 0 else 1.0
            total += weight * direction * span
        return total

    def cohesion_cost(self, order: Sequence[str]) -> float:
        total = 0.0
        for left, right in zip(order, order[1:], strict=False):
            a, b = self.embeddings.get(left), self.embeddings.get(right)
            if a is not None and b is not None:
                total -= _cosine(a, b)
        return total

    def stability_cost(self, pos: Mapping[str, int]) -> float:
        if not self.previous:
            return 0.0
        before = {node: i for i, node in enumerate(self.previous)}
        return float(
            sum(abs(pos[node] - before[node]) for node in self.node_ids if node in before)
        )

    def cost(self, order: Sequence[str]) -> CostBreakdown:
        pos = {node: i for i, node in enumerate(order)}
        reference = self.reference_cost(pos)
        cohesion = self.cohesion_cost(order)
        stability = self.stability_cost(pos)
        total = (
            self.weights.reference * reference
            + self.weights.cohesion * cohesion
            + self.weights.stability * stability
        )
        return CostBreakdown(reference, cohesion, stability, total)

    # ------------------------------------------------------------------ #
    # Starting points
    # ------------------------------------------------------------------ #
    def topological_order(self, seed_order: Sequence[str] | None = None) -> list[str]:
        """A feasible order, seeded by the previous one where there is one.

        Seeding matters more than it looks: starting from the author's accepted
        order makes ``stability_cost`` near zero at iteration one, so the search
        spends its budget deciding whether a change is worth making rather than
        climbing back to where the manuscript already was.
        """

        # `is not None`, not truthiness: an empty list means "no seed, sort by
        # id", and `[] or self.previous` silently means "seed from the previous
        # order" instead. That difference turned the DFS baseline into a copy of
        # the accepted order, so `compare_baselines` reported the optimizer
        # tying a baseline that was secretly the optimizer's own output --
        # a measurement that could only ever confirm itself.
        seed = self.previous if seed_order is None else seed_order
        preference = {node: i for i, node in enumerate(seed)}
        fallback = len(preference)
        remaining = set(self.node_ids)
        placed: list[str] = []
        done: set[str] = set()

        while remaining:
            ready = [n for n in remaining if self.prerequisites.get(n, set()) <= done]
            if not ready:
                raise PrecedenceViolation(
                    "no node is placeable: the DEPENDS_ON subgraph has a cycle"
                )
            ready.sort(key=lambda n: (preference.get(n, fallback), n))
            chosen = ready[0]
            placed.append(chosen)
            done.add(chosen)
            remaining.discard(chosen)

        return self._apply_pins(self._apply_adjacency(placed))

    def _apply_adjacency(self, order: list[str]) -> list[str]:
        """Pull each reply up against its objection, precedence permitting."""

        for objection, reply in self.pairs:
            if objection not in order or reply not in order:
                continue
            pos = {node: i for i, node in enumerate(order)}
            target = pos[objection] + 1
            if pos[reply] == target:
                continue
            order.pop(pos[reply])
            pos = {node: i for i, node in enumerate(order)}
            insert_at = pos[objection] + 1
            lo = max(
                (pos[p] + 1 for p in self.prerequisites.get(reply, ()) if p in pos), default=0
            )
            order.insert(max(insert_at, lo), reply)
        return order

    def _apply_pins(self, order: list[str]) -> list[str]:
        if not self.pins:
            return order
        free = [n for n in order if n not in self.pins]
        placed: list[str | None] = [None] * len(order)
        for node, position in self.pins.items():
            if not 0 <= position < len(order):
                raise PrecedenceViolation(
                    f"pin: {node} pinned to position {position}, outside 0..{len(order) - 1}"
                )
            if placed[position] is not None:
                raise PrecedenceViolation(f"pin: two nodes pinned to position {position}")
            placed[position] = node
        it = iter(free)
        return [node if node is not None else next(it) for node in placed]


def build_problem(
    graph: ArgumentGraph,
    *,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    previous: Sequence[str] | None = None,
    pins: Mapping[str, int] | None = None,
    weights: Weights | None = None,
) -> LinearizeProblem:
    """Read the ordering problem out of an argument graph.

    Node ids are sorted, so a graph loaded twice produces the same problem and
    therefore the same order: determinism starts here, not at the seed.
    """

    node_ids = sorted(graph.nodes)
    prerequisites: dict[str, set[str]] = {n: set() for n in node_ids}
    dependents: dict[str, set[str]] = {n: set() for n in node_ids}
    references: list[tuple[str, str, float]] = []
    edge_weights = (weights or Weights()).edge_weights

    for edge in graph.edges:
        if edge.source not in prerequisites or edge.target not in prerequisites:
            continue
        if edge.type is EdgeType.DEPENDS_ON:
            # source depends on target -> target is read first.
            prerequisites[edge.source].add(edge.target)
            dependents[edge.target].add(edge.source)
        else:
            weight = edge_weights.get(edge.type, 0.5)
            if weight:
                references.append((edge.source, edge.target, weight))

    # An objection answered by a reply: the REPLY node attacks nothing, it
    # *is* the answer, so the pairing comes from the reply's own edges into the
    # objection. Splitting the two is a reading failure rather than a cost, so
    # the pair is a hard constraint (see LinearizeProblem.violations).
    pairs: list[tuple[str, str]] = []
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        if source.type is NodeType.REPLY and target.type is NodeType.OBJECTION:
            pairs.append((edge.target, edge.source))
    pairs = sorted(set(pairs))

    problem = LinearizeProblem(
        node_ids=node_ids,
        prerequisites=prerequisites,
        dependents=dependents,
        references=references,
        pins=dict(pins or {}),
        pairs=pairs,
        embeddings={k: list(v) for k, v in (embeddings or {}).items()},
        words={n: len(graph.nodes[n].text.split()) for n in node_ids},
        previous=[n for n in (previous or []) if n in prerequisites],
        weights=weights or Weights(),
    )
    problem.assert_acyclic()
    return problem
