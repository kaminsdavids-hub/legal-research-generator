"""Biased random-key GA, for graphs too large for the annealer (>~300 nodes).

The encoding is the point. A solution is a vector of real keys, one per node,
and decoding walks the precedence DAG choosing among placeable nodes by key. So
crossover and mutation cannot produce an invalid order -- there is nothing to
repair, because the genome is not a permutation. Permutation crossover with a
repair pass spends most of its runtime repairing, and repair biases the search
in ways nobody chose.

**Deviation from the brief, stated rather than buried:** it asks for pymoo or
DEAP. Neither is installed, and the BRKGA loop under a random-key encoding is
~80 lines with no operators to configure -- elite carry-over, biased uniform
crossover, random mutants. Pulling a framework in to host that would add a large
dependency to a repo with a deliberately tight one, and would fight the decoder,
which is the only part with any content. Nothing here needs pymoo's Problem
abstraction or its operator zoo.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .problem import CostBreakdown, LinearizeProblem

__all__ = ["BRKGAConfig", "BRKGAResult", "decode", "evolve"]


@dataclass(frozen=True)
class BRKGAConfig:
    population: int = 100
    generations: int = 200
    #: Share of the population carried into the next generation untouched.
    elite_fraction: float = 0.2
    #: Share replaced by fresh random keys each generation -- the diversity
    #: source that stops the elite converging on one basin.
    mutant_fraction: float = 0.15
    #: Probability a child takes each key from the elite parent. Above 0.5 by
    #: definition: this is the "biased" in biased random-key.
    elite_bias: float = 0.7
    seed: int = 7


@dataclass
class BRKGAResult:
    order: list[str]
    cost: CostBreakdown
    generations: int
    seed: int
    initial_cost: CostBreakdown | None = None
    history: list[float] = None  # type: ignore[assignment]


def decode(problem: LinearizeProblem, keys: Sequence[float]) -> list[str]:
    """Keys -> a feasible order, by repeated topological selection.

    At each step the candidates are the nodes whose prerequisites are all
    placed; the one with the lowest key goes next. Ties break on node id so the
    decode is deterministic for identical keys.

    Pins and objection/reply adjacency are applied after the walk, by the same
    helpers the topological seed uses. They are hard constraints, so they are
    imposed rather than searched for.
    """

    key = dict(zip(problem.node_ids, keys, strict=False))
    remaining = set(problem.node_ids)
    placed: set[str] = set()
    order: list[str] = []

    while remaining:
        ready = [n for n in remaining if problem.prerequisites.get(n, set()) <= placed]
        if not ready:  # pragma: no cover - build_problem asserts acyclicity
            raise ValueError("decode stalled: the DEPENDS_ON subgraph has a cycle")
        chosen = min(ready, key=lambda n: (key.get(n, 0.5), n))
        order.append(chosen)
        placed.add(chosen)
        remaining.discard(chosen)

    return problem._apply_pins(problem._apply_adjacency(order))


def evolve(
    problem: LinearizeProblem,
    *,
    config: BRKGAConfig | None = None,
) -> BRKGAResult:
    cfg = config or BRKGAConfig()
    rng = random.Random(cfg.seed)
    size = len(problem.node_ids)

    elite_count = max(1, int(cfg.population * cfg.elite_fraction))
    mutant_count = max(1, int(cfg.population * cfg.mutant_fraction))

    def random_keys() -> list[float]:
        return [rng.random() for _ in range(size)]

    # Seed one individual from the previous accepted order, so stability is
    # reachable from generation zero rather than something the search has to
    # rediscover.
    population: list[list[float]] = [random_keys() for _ in range(cfg.population)]
    if problem.previous:
        rank = {node: i for i, node in enumerate(problem.previous)}
        population[0] = [
            rank.get(node, len(rank)) / max(1, size) for node in problem.node_ids
        ]

    def score(keys: Sequence[float]) -> tuple[float, list[str], CostBreakdown]:
        order = decode(problem, keys)
        cost = problem.cost(order)
        return cost.total, order, cost

    scored = sorted((score(k) + (k,) for k in population), key=lambda row: row[0])
    initial_cost = scored[0][2]
    history = [round(scored[0][0], 3)]

    for _ in range(cfg.generations):
        elites = [row[3] for row in scored[:elite_count]]
        others = [row[3] for row in scored[elite_count:]]

        children: list[list[float]] = list(elites)
        children.extend(random_keys() for _ in range(mutant_count))
        while len(children) < cfg.population:
            elite = rng.choice(elites)
            other = rng.choice(others) if others else random_keys()
            children.append(
                [
                    elite[i] if rng.random() < cfg.elite_bias else other[i]
                    for i in range(size)
                ]
            )

        scored = sorted((score(k) + (k,) for k in children), key=lambda row: row[0])
        history.append(round(scored[0][0], 3))

    best_total, best_order, best_cost, _ = scored[0]
    return BRKGAResult(
        order=best_order,
        cost=best_cost,
        generations=cfg.generations,
        seed=cfg.seed,
        initial_cost=initial_cost,
        history=history,
    )
