"""Search wrappers: NSGA-II (pymoo) and MAP-Elites (pyribs), plus persistence.

Thin on purpose. The libraries own the search; this module owns the two things
they cannot know about — that an archive is only comparable to another built by
the same evaluator, and that a result has to carry enough to reproduce it.

Archives are keyed by evaluator hash. Loading an archive whose stamp does not
match the current schema and rules is refused rather than merged: the cells
would hold elites scored by a different function, and the resulting "coverage"
figure would be an average over two incompatible experiments.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import EvaluatorStamp

__all__ = [
    "ArchiveMismatch",
    "Elite",
    "EliteArchive",
    "nsga2_front",
    "map_elites",
]


class ArchiveMismatch(Exception):
    """Raised when a stored archive was built by a different evaluator."""


@dataclass(frozen=True)
class Elite:
    #: Genotype as a plain mapping, so an artifact is readable without the code
    #: that produced it.
    genome: dict[str, Any]
    objectives: dict[str, float]
    descriptors: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EliteArchive:
    stamp: EvaluatorStamp
    elites: list[Elite] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "stamp": self.stamp.as_dict(),
                    "elites": [
                        {
                            "genome": e.genome,
                            "objectives": e.objectives,
                            "descriptors": e.descriptors,
                            "meta": e.meta,
                        }
                        for e in self.elites
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, expect: EvaluatorStamp | None = None) -> EliteArchive:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = data["stamp"]
        stamp = EvaluatorStamp(
            schema_digest=raw["schema_digest"],
            rules_digest=raw["rules_digest"],
            epoch=int(raw["epoch"]),
            seed=int(raw["seed"]),
            recipe=raw.get("recipe", ""),
        )
        if expect is not None and not stamp.comparable_to(expect):
            raise ArchiveMismatch(
                f"archive at {path} was built by a different evaluator "
                f"(schema {stamp.schema_digest} rules {stamp.rules_digest} "
                f"epoch {stamp.epoch}; expected {expect.schema_digest} / "
                f"{expect.rules_digest} / {expect.epoch}). Its cells were scored "
                "by different functions; merging them would average two experiments."
            )
        return cls(
            stamp=stamp,
            elites=[
                Elite(
                    genome=e["genome"],
                    objectives=e["objectives"],
                    descriptors=e.get("descriptors", {}),
                    meta=e.get("meta", {}),
                )
                for e in data["elites"]
            ],
        )


# --------------------------------------------------------------------------- #
# NSGA-II
# --------------------------------------------------------------------------- #
def nsga2_front(
    evaluate: Callable[[Sequence[int]], Sequence[float]],
    n_var: int,
    bounds: Sequence[int],
    *,
    n_obj: int,
    population: int = 60,
    generations: int = 40,
    seed: int = 7,
    repair: Callable[[Sequence[int]], list[int]] | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Minimise ``evaluate`` over an integer genotype; return the final front.

    Every objective is a *minimisation*: callers negate anything they want
    maximised, at the point where the sign is obvious, rather than passing a
    direction vector that has to be read alongside the numbers.
    """

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize

    class _Problem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                xl=np.zeros(n_var, dtype=int),
                xu=np.asarray(bounds, dtype=int),
                vtype=int,
            )

        def _evaluate(self, x, out, *args, **kwargs):  # noqa: ANN001, ANN202
            genome = [int(v) for v in x]
            if repair is not None:
                genome = repair(genome)
            out["F"] = np.asarray(evaluate(genome), dtype=float)

    algorithm = NSGA2(
        pop_size=population,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.3, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    result = minimize(
        _Problem(),
        algorithm,
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )

    genomes = np.atleast_2d(result.X)
    values = np.atleast_2d(result.F)
    front: list[tuple[list[int], list[float]]] = []
    for genome, objective in zip(genomes, values, strict=False):
        decoded = [int(v) for v in genome]
        front.append((repair(decoded) if repair else decoded, [float(v) for v in objective]))
    return front


# --------------------------------------------------------------------------- #
# MAP-Elites
# --------------------------------------------------------------------------- #
def map_elites(
    ask: Callable[[Any], list[int]],
    evaluate: Callable[[Sequence[int]], tuple[float, Sequence[float]]],
    *,
    dims: Sequence[int],
    ranges: Sequence[tuple[float, float]],
    solution_dim: int,
    iterations: int = 500,
    batch: int = 20,
    seed: int = 7,
) -> Any:
    """Fill a behavioural grid with the best solution found in each cell.

    The reason this and not a single-objective search: the point is *coverage of
    the space of hard cases*, not one hardest case. An annealer returns the
    knife-edge fact pattern it happened to find; an archive returns one per
    region of the feature space, and structurally distinct hard cases are what a
    referee actually raises.

    ``ask`` proposes a genome given an RNG -- it owns feasibility, so infeasible
    individuals are never created rather than created and repaired.
    """

    import numpy as np
    from ribs.archives import GridArchive

    rng = np.random.default_rng(seed)
    archive = GridArchive(solution_dim=solution_dim, dims=list(dims), ranges=list(ranges))

    for _ in range(max(1, iterations // max(1, batch))):
        solutions = np.asarray([ask(rng) for _ in range(batch)], dtype=float)
        scores = []
        measures = []
        for solution in solutions:
            fitness, descriptor = evaluate([int(v) for v in solution])
            scores.append(fitness)
            measures.append(list(descriptor))
        archive.add(solutions, np.asarray(scores, dtype=float), np.asarray(measures, dtype=float))

    return archive
