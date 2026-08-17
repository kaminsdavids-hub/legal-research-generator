"""Simulated annealing over feasible orders. The default solver up to ~300 nodes.

Hand-rolled rather than taken from a framework, because the whole design is in
the move set: every proposal is built inside a node's feasible window, so no
candidate is ever infeasible and there is no repair step. A generic annealer
would hand back invalid permutations and spend its budget fixing them.

Throughput is the game. A proposal costs O(edges incident to the moved span)
rather than a full O(V + E) rescore, which is what makes a few tens of thousands
of proposals affordable in Python.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .problem import CostBreakdown, LinearizeProblem, PrecedenceViolation

__all__ = ["SAConfig", "SAResult", "anneal"]


@dataclass(frozen=True)
class SAConfig:
    iterations: int = 20_000
    start_temperature: float = 2.0
    end_temperature: float = 0.01
    #: Longest single relocate. Bounded so a proposal's delta stays cheap: the
    #: nodes between old and new position all shift, and every edge touching
    #: them changes distance. Long moves are what block relocates are for.
    max_span: int = 16
    max_block: int = 6
    #: Proposals without improvement before restarting from the best order.
    stagnation: int = 2_000
    seed: int = 7


@dataclass
class SAResult:
    order: list[str]
    cost: CostBreakdown
    iterations: int
    accepted: int
    restarts: int
    seed: int
    #: Cost of the order the search started from, so a report can say what the
    #: search actually bought over a plain topological sort.
    initial_cost: CostBreakdown | None = None
    history: list[float] = field(default_factory=list)


def _pinned_positions(problem: LinearizeProblem) -> set[int]:
    return set(problem.pins.values())


def anneal(
    problem: LinearizeProblem,
    *,
    config: SAConfig | None = None,
    initial: Sequence[str] | None = None,
) -> SAResult:
    """Anneal from a feasible start and return the best order seen.

    The result is exactly reproducible from ``config.seed``: the RNG is local,
    never the global one, so a run inside a test suite that also uses ``random``
    cannot drift.
    """

    cfg = config or SAConfig()
    rng = random.Random(cfg.seed)

    order = list(initial) if initial is not None else problem.topological_order()
    problems = problem.violations(order)
    if problems:
        raise PrecedenceViolation("initial order is infeasible: " + "; ".join(problems[:3]))

    current = problem.cost(order)
    initial_cost = current
    best_order, best_cost = list(order), current
    pinned = _pinned_positions(problem)
    accepted = restarts = 0
    since_improvement = 0
    history: list[float] = []

    # Geometric cooling: T_k = T0 * (T_end/T0)^(k/n).
    ratio = (cfg.end_temperature / cfg.start_temperature) ** (1.0 / max(1, cfg.iterations))
    temperature = cfg.start_temperature

    for step in range(cfg.iterations):
        temperature *= ratio
        candidate = _propose(problem, order, rng, cfg, pinned)
        if candidate is None:
            continue

        cost = problem.cost(candidate)
        delta = cost.total - current.total
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
            order, current = candidate, cost
            accepted += 1
            if cost.total < best_cost.total - 1e-9:
                best_order, best_cost = list(candidate), cost
                since_improvement = 0
            else:
                since_improvement += 1
        else:
            since_improvement += 1

        if since_improvement >= cfg.stagnation:
            # Restart from the best rather than a random order: the point of the
            # restart is to escape a basin, not to discard what has been found.
            order, current = list(best_order), best_cost
            since_improvement = 0
            restarts += 1

        if step % 100 == 0:
            history.append(round(current.total, 3))

    return SAResult(
        order=best_order,
        cost=best_cost,
        iterations=cfg.iterations,
        accepted=accepted,
        restarts=restarts,
        seed=cfg.seed,
        initial_cost=initial_cost,
        history=history,
    )


def _propose(
    problem: LinearizeProblem,
    order: list[str],
    rng: random.Random,
    cfg: SAConfig,
    pinned: set[int],
) -> list[str] | None:
    """One candidate order, or ``None`` when the drawn move has no legal form.

    Returning ``None`` rather than retrying inside the move keeps the acceptance
    statistics honest -- a move that is usually illegal should show up as wasted
    proposals, not hide inside a loop.
    """

    roll = rng.random()
    if roll < 0.55:
        candidate = _relocate(problem, order, rng, cfg, pinned)
    elif roll < 0.80:
        candidate = _swap(problem, order, rng, pinned)
    else:
        candidate = _relocate_block(problem, order, rng, cfg, pinned)

    if candidate is None or candidate == order:
        return None
    # Precedence holds by construction; pins and adjacency are cheap to check
    # and cheaper than a repair pass, so they are verified rather than encoded.
    if not problem.is_feasible(candidate):
        return None
    return candidate


def _movable_index(order: list[str], rng: random.Random, pinned: set[int]) -> int | None:
    for _ in range(8):
        index = rng.randrange(len(order))
        if index not in pinned:
            return index
    return None


def _relocate(
    problem: LinearizeProblem,
    order: list[str],
    rng: random.Random,
    cfg: SAConfig,
    pinned: set[int],
) -> list[str] | None:
    index = _movable_index(order, rng, pinned)
    if index is None:
        return None
    lo, hi = problem.window(order, index)
    lo = max(lo, index - cfg.max_span)
    hi = min(hi, index + cfg.max_span)
    if hi <= lo:
        return None
    target = rng.randint(lo, hi)
    if target == index:
        return None
    candidate = list(order)
    node = candidate.pop(index)
    candidate.insert(target, node)
    return candidate


def _swap(
    problem: LinearizeProblem,
    order: list[str],
    rng: random.Random,
    pinned: set[int],
) -> list[str] | None:
    i = _movable_index(order, rng, pinned)
    j = _movable_index(order, rng, pinned)
    if i is None or j is None or i == j:
        return None
    lo_i, hi_i = problem.window(order, i)
    lo_j, hi_j = problem.window(order, j)
    if not (lo_i <= j <= hi_i and lo_j <= i <= hi_j):
        return None
    candidate = list(order)
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def _relocate_block(
    problem: LinearizeProblem,
    order: list[str],
    rng: random.Random,
    cfg: SAConfig,
    pinned: set[int],
) -> list[str] | None:
    """Move a contiguous run as a unit.

    The move that matters most. A sub-argument is a run of nodes that belong
    together, and single-node annealing cannot migrate one without passing
    through orders where it is torn in half -- each of which scores worse, so
    the search never takes the first step.
    """

    size = rng.randint(2, cfg.max_block)
    if size >= len(order):
        return None
    start = rng.randrange(len(order) - size + 1)
    block = order[start : start + size]
    if any(p in pinned for p in range(start, start + size)):
        return None

    rest = order[:start] + order[start + size :]
    # The block may sit anywhere that keeps every member after its prerequisites
    # and before its dependents, evaluated against the order without it.
    pos = {node: i for i, node in enumerate(rest)}
    lo, hi = 0, len(rest)
    for node in block:
        for prerequisite in problem.prerequisites.get(node, ()):  # noqa: SIM118
            if prerequisite in pos:
                lo = max(lo, pos[prerequisite] + 1)
        for dependent in problem.dependents.get(node, ()):  # noqa: SIM118
            if dependent in pos:
                hi = min(hi, pos[dependent])
    lo = max(lo, start - cfg.max_span)
    hi = min(hi, start + cfg.max_span)
    if hi <= lo:
        return None

    target = rng.randint(lo, hi)
    if target == start:
        return None
    return rest[:target] + block + rest[target:]
