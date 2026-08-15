"""MAP-Elites over the feature space, and NSGA-II as the front-based alternative.

The archive is the deliverable, not a winner. A single hardest fact pattern is
one referee question; an archive with one elite per region of the feature space
is the set of structurally distinct hard cases, which is what a referee actually
has. Reporting `max(disagreement)` would hide that the top ten are variants of
each other.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from modules.casework.corpus import CaseCorpus
from modules.casework.schema import EvaluatorStamp, FeatureSchema
from modules.casework.search import Elite, EliteArchive

from .genotype import FactVector, FeasibilityMask, decode, encode
from .objectives import cached_scorer
from .rules import RuleSet

__all__ = ["DivergenceConfig", "run_map_elites", "run_nsga2"]


@dataclass(frozen=True)
class DivergenceConfig:
    #: 2-4 axes used as behavioural descriptors. More than four and the grid is
    #: mostly empty; fewer than two and the archive is a list.
    descriptors: tuple[str, ...] = ()
    iterations: int = 600
    batch: int = 20
    seed: int = 7
    max_radius: int = 2
    #: Author's weight on the realism prior. Zero by default: it is optional in
    #: the brief and it encodes whatever has been coded, so it should be turned
    #: on deliberately.
    realism_weight: float = 0.0
    #: Brittleness is a distance, and a *small* one is interesting, so the
    #: fitness subtracts it. This is the exchange rate between "the readings
    #: disagree" and "one step away, they stop disagreeing".
    brittleness_weight: float = 0.25
    #: The command that produced this run, stamped on the archive verbatim. Set
    #: by the CLI, which is the only layer that knows the whole invocation. The
    #: fallback below is built from what this config holds, and it is not enough
    #: on its own: descriptors, the realism weight and any seeding all change the
    #: archive, and a stamp that omits them is a recipe that does not reproduce.
    recipe: str = ""


def _fallback_recipe(
    schema: FeatureSchema,
    cfg: DivergenceConfig,
    descriptors: Sequence[str],
    epoch: int,
    priors: Sequence[FactVector],
) -> str:
    """A recipe for a caller that did not supply one, honest about its limits.

    Seeds cannot be expressed as flags from in here -- this function sees a list
    of vectors, not the switch that produced them -- so rather than printing a
    command that silently drops them, it says so. A stamp claiming reproduction
    it cannot deliver is worse than no stamp.
    """

    recipe = (
        f"divergence search --schema {schema.source} --rules <dir> "
        f"--epoch {epoch} --seed {cfg.seed} --iterations {cfg.iterations} "
        f"--descriptors {' '.join(descriptors)} "
        f"--realism-weight {cfg.realism_weight}"
    )
    if priors:
        recipe += f"  # INCOMPLETE: plus {len(priors)} seed vector(s) these flags do not name"
    return recipe


def _fitness(scores, config: DivergenceConfig) -> float:
    """One scalar for the archive to rank cells by.

    MAP-Elites needs a single fitness; the multi-objective view is NSGA-II's
    job, and both are offered because they answer different questions. This
    scalarisation is stated here rather than buried: disagreement is the point,
    brittleness sharpens it, realism is optional and off by default.
    """

    return (
        scores.disagreement
        - config.brittleness_weight * scores.brittleness
        + config.realism_weight * scores.realism
    )


def run_map_elites(
    schema: FeatureSchema,
    rules: RuleSet,
    mask: FeasibilityMask,
    *,
    corpus: CaseCorpus | None = None,
    config: DivergenceConfig | None = None,
    epoch: int = 0,
    priors: Sequence[FactVector] = (),
) -> EliteArchive:
    """Fill the archive. Returns elites stamped with the evaluator that scored them.

    ``priors`` are fact vectors the coupling layer wants explored -- the regions
    where Portfolio reported thin coverage. They are injected as *seeds*, never
    as a constraint: a prior that fenced the search would let last epoch's
    citation gaps decide which law gets tested this epoch.
    """

    cfg = config or DivergenceConfig()
    descriptors = cfg.descriptors or tuple(a.name for a in schema.axes[:2])
    for name in descriptors:
        schema.axis(name)  # raises if the author named an axis that is not there

    scorer = cached_scorer(rules, mask, schema, corpus, max_radius=cfg.max_radius)
    rng = random.Random(cfg.seed)
    seeds = list(priors)

    from ribs.archives import GridArchive

    dims = [len(schema.axis(name).values) for name in descriptors]
    ranges = [(0.0, float(d)) for d in dims]
    archive = GridArchive(solution_dim=len(schema.names), dims=dims, ranges=ranges)

    import numpy as np

    for _ in range(max(1, cfg.iterations // max(1, cfg.batch))):
        vectors: list[FactVector] = []
        while len(vectors) < cfg.batch:
            if seeds:
                base = seeds.pop()
                vectors.append(base)
                continue
            vectors.append(mask.sample(rng))

        solutions = np.asarray([encode(v, schema) for v in vectors], dtype=float)
        objectives = []
        measures = []
        for vector in vectors:
            scores = scorer(vector)
            objectives.append(_fitness(scores, cfg))
            measures.append(
                [float(schema.axis(name).index(vector[name])) + 0.5 for name in descriptors]
            )
        archive.add(
            solutions,
            np.asarray(objectives, dtype=float),
            np.asarray(measures, dtype=float),
        )

    elites: list[Elite] = []
    for record in archive:
        vector = decode([int(round(v)) for v in record["solution"]], schema)
        scores = scorer(vector)
        elites.append(
            Elite(
                genome=vector.as_dict(),
                objectives={**scores.as_dict(), "fitness": round(float(record["objective"]), 4)},
                descriptors={
                    name: vector[name] for name in descriptors
                },
                meta={"verdicts": list(scores.labels)},
            )
        )

    elites.sort(key=lambda e: (-e.objectives["disagreement"], e.objectives["brittleness"]))
    return EliteArchive(
        stamp=EvaluatorStamp(
            schema_digest=schema.digest,
            rules_digest=rules.digest,
            epoch=epoch,
            seed=cfg.seed,
            recipe=cfg.recipe or _fallback_recipe(schema, cfg, descriptors, epoch, priors),
        ),
        elites=elites,
    )


def run_nsga2(
    schema: FeatureSchema,
    rules: RuleSet,
    mask: FeasibilityMask,
    *,
    corpus: CaseCorpus | None = None,
    config: DivergenceConfig | None = None,
    epoch: int = 0,
) -> EliteArchive:
    """The front-based alternative: the trade-off between the three objectives.

    Offered because the archive answers "which structurally distinct hard cases
    exist" and the front answers "what does buying more disagreement cost in
    brittleness and plausibility". Neither subsumes the other.
    """

    cfg = config or DivergenceConfig()
    scorer = cached_scorer(rules, mask, schema, corpus, max_radius=cfg.max_radius)

    from modules.casework.search import nsga2_front

    bounds = [len(schema.axis(name).values) - 1 for name in schema.names]

    def evaluate(genome: Sequence[int]) -> list[float]:
        scores = scorer(decode(genome, schema))
        # All minimised: disagreement and realism are negated at the point where
        # the sign is obvious rather than carried as a direction vector.
        return [
            -scores.disagreement,
            scores.brittleness,
            -scores.realism * cfg.realism_weight,
        ]

    front = nsga2_front(
        evaluate,
        n_var=len(schema.names),
        bounds=bounds,
        n_obj=3,
        seed=cfg.seed,
        repair=mask.repair_genome,
    )

    elites: list[Elite] = []
    for genome, _objectives in front:
        vector = decode(genome, schema)
        if not mask.feasible(vector):  # pragma: no cover - repair guarantees this
            continue
        scores = scorer(vector)
        elites.append(
            Elite(
                genome=vector.as_dict(),
                objectives=scores.as_dict(),
                meta={"verdicts": list(scores.labels), "mode": "nsga2"},
            )
        )

    elites.sort(key=lambda e: (-e.objectives["disagreement"], e.objectives["brittleness"]))
    return EliteArchive(
        stamp=EvaluatorStamp(
            schema_digest=schema.digest,
            rules_digest=rules.digest,
            epoch=epoch,
            seed=cfg.seed,
            recipe=f"divergence search --mode nsga2 --epoch {epoch} --seed {cfg.seed}",
        ),
        elites=elites,
    )
