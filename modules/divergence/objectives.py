"""What makes a fact pattern hard: disagreement, brittleness, and plausibility.

All three are computed from frozen predicates and coded data. Nothing here asks
a model anything, which is what makes it safe to point an evolutionary search at
them — a search will find whatever the objective actually rewards, so the
objective has to be something that cannot be satisfied by rewording.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from modules.casework.corpus import CaseCorpus
from modules.casework.schema import FeatureSchema

from .genotype import FactVector, FeasibilityMask
from .rules import RuleSet

__all__ = ["Scores", "brittleness", "disagreement", "realism_prior", "score"]


@dataclass(frozen=True)
class Scores:
    disagreement: float
    brittleness: float
    realism: float
    #: The verdict labels that produced the disagreement figure, kept so a
    #: report can show the split rather than only its entropy.
    labels: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, float]:
        return {
            "disagreement": round(self.disagreement, 4),
            "brittleness": round(self.brittleness, 4),
            "realism": round(self.realism, 4),
        }


def disagreement(labels: list[str]) -> float:
    """Shannon entropy over the verdicts, normalised to ``[0, 1]``.

    Entropy rather than a count of disagreeing pairs, and the difference is the
    point. With five rules, a 4-1 split has four disagreeing pairs and a 2-2-1
    split has eight, but pair-counting also rewards a lopsided 4-1 more than it
    should: the interesting case is the one where the readings genuinely
    fracture, not the one where a single outlier disagrees with a consensus.
    Entropy is maximal exactly when the readings are evenly divided.
    """

    if len(labels) < 2:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
    ceiling = math.log2(min(total, len(counts)) or 1)
    # Normalise by the entropy of an even split across the labels *available*,
    # so a two-rule set that splits 1-1 scores 1.0 rather than being penalised
    # for having fewer rules than another project.
    ceiling = math.log2(total) if total > 1 else 1.0
    return entropy / ceiling if ceiling else 0.0


def brittleness(
    facts: FactVector,
    rules: RuleSet,
    mask: FeasibilityMask,
    schema: FeatureSchema,
    *,
    max_radius: int = 2,
) -> float:
    """Distance to the nearest feasible neighbour where some rule flips.

    Small distance = knife-edge: change one thing a little and the law's answer
    changes. Returned as a *cost to minimise* by convention elsewhere; here it
    is the raw distance, and 0 means a neighbour one step away already flips.

    Bounded local search, not exhaustive: the neighbourhood grows fast and the
    interesting answer is always small. A pattern whose nearest flip is four
    steps away is not knife-edge, and knowing whether it is four or seven buys
    nothing.
    """

    baseline = tuple(rules.labels(facts))
    frontier = [facts]
    seen = {tuple(sorted(facts.as_dict().items()))}

    for radius in range(1, max_radius + 1):
        nxt: list[FactVector] = []
        for current in frontier:
            for neighbour in mask.neighbours(current):
                key = tuple(sorted(neighbour.as_dict().items()))
                if key in seen:
                    continue
                seen.add(key)
                if tuple(rules.labels(neighbour)) != baseline:
                    return float(radius)
                nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break
    return float(max_radius + 1)


def realism_prior(facts: FactVector, corpus: CaseCorpus, schema: FeatureSchema) -> float:
    """How close this pattern sits to the coded real cases, in ``[0, 1]``.

    Optional and author-weighted, because it cuts both ways: it keeps the
    archive from filling with exotic combinations, and it also pulls the search
    toward whatever happens to have been coded. An axis where 28 of 30 coded
    cases share a value will teach this prior that the value is realism itself.
    :meth:`CaseCorpus.coverage` exists so the author can see that before
    trusting it.

    Only *verified* codings count. An unverified coding is a guess, and a prior
    built from guesses is a guess with a number attached.
    """

    cases = corpus.verified()
    if not cases:
        return 0.0
    total_axes = max(1, len(schema.axes))
    best = 0.0
    for case in cases:
        matched = sum(
            1
            for axis in schema.axes
            if case.features.get(axis.name) == facts.values.get(axis.name)
        )
        best = max(best, matched / total_axes)
    return best


def score(
    facts: FactVector,
    rules: RuleSet,
    mask: FeasibilityMask,
    schema: FeatureSchema,
    corpus: CaseCorpus | None = None,
    *,
    max_radius: int = 2,
) -> Scores:
    labels = rules.labels(facts)
    return Scores(
        disagreement=disagreement(labels),
        brittleness=brittleness(facts, rules, mask, schema, max_radius=max_radius),
        realism=realism_prior(facts, corpus, schema) if corpus else 0.0,
        labels=tuple(labels),
    )


def cached_scorer(
    rules: RuleSet,
    mask: FeasibilityMask,
    schema: FeatureSchema,
    corpus: CaseCorpus | None = None,
    *,
    max_radius: int = 2,
):
    """A memoised ``FactVector -> Scores``.

    Brittleness dominates the cost -- it evaluates every rule over a whole
    neighbourhood -- and a search revisits the same cells constantly, so the
    cache is not an optimisation, it is what makes the archive affordable.
    Keyed on the genotype, which is hashable once flattened.
    """

    @lru_cache(maxsize=100_000)
    def _score(key: tuple[tuple[str, str], ...]) -> Scores:
        return score(
            FactVector(dict(key)), rules, mask, schema, corpus, max_radius=max_radius
        )

    def scorer(facts: FactVector | Mapping[str, str]) -> Scores:
        values = facts.values if isinstance(facts, FactVector) else facts
        return _score(tuple(sorted(values.items())))

    scorer.cache_info = _score.cache_info  # type: ignore[attr-defined]
    return scorer
