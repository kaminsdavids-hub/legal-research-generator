"""Starting the search from the decided cases rather than from nowhere.

An unseeded run samples uniformly from the feasible set, and the feasible set is
overwhelmingly territory no court has been near: the last full run put its 48
hard cases a minimum of two axes away from every coded record. Those are worth
having -- a framework's edge cases are not obliged to have been litigated -- but
a search that never passes close to a real decision can never produce the one
finding the module is for, which is that a decided case is misclassified.

Seeds are seeds, never constraints. They are injected as starting vectors and
the search moves off them freely; fencing it to the neighbourhood of the corpus
would let what has been coded decide what gets tested, which is the same failure
as the realism prior taken to its limit.

**A coded case is not itself a seed, because it is not a fact pattern.** Wildcard
axes carry no value the search may hold, so a case is *completed* first: every
combination of real values over its blank axes, filtered through the feasibility
mask. That is the exact meaning of an undetermined axis -- the case is at one of
these points and the record does not say which -- so enumerating them seeds the
whole region the case might occupy.

For an inapplicable axis the completion means something weaker, and the caller
should know it. The case is at *none* of those points; the expansion is only a
way of saying "start the search near this case in the axes it does have". A
completion of an inapplicable axis can never itself be the case, and nothing here
claims otherwise -- the matcher is what decides whether a found pattern is a real
decision, and it applies its own tolerance rules to that question.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from modules.casework.corpus import CaseCorpus
from modules.casework.schema import WILDCARDS, FeatureSchema

from .genotype import FactVector, FeasibilityMask

__all__ = ["SeedPlan", "seeds_from_cases"]


@dataclass
class SeedPlan:
    seeds: list[FactVector] = field(default_factory=list)
    #: case id -> how many completions it contributed.
    per_case: dict[str, int] = field(default_factory=dict)
    #: Cases whose every completion was infeasible under the mask. Worth naming:
    #: a coded real decision that the mask calls impossible is a bug in one of
    #: them, and silence here would hide it.
    infeasible: list[str] = field(default_factory=list)
    #: Cases dropped because their completions exceeded the budget, and how many
    #: were skipped. Reported rather than truncated quietly -- a capped seeding
    #: run that says nothing reads as full coverage of the corpus.
    truncated: list[str] = field(default_factory=list)
    dropped: int = 0

    def summary(self) -> str:
        parts = [f"{len(self.seeds)} seed(s) from {len(self.per_case)} coded case(s)"]
        if self.infeasible:
            parts.append(f"{len(self.infeasible)} case(s) infeasible under the mask")
        if self.truncated:
            parts.append(f"{self.dropped} completion(s) dropped past the budget")
        return ", ".join(parts)


def seeds_from_cases(
    schema: FeatureSchema,
    corpus: CaseCorpus,
    mask: FeasibilityMask,
    *,
    budget: int = 400,
    verified_only: bool = False,
) -> SeedPlan:
    """Feasible fact patterns standing in for each coded case.

    ``verified_only`` is off by default, and that is deliberate even though the
    rest of this package guards verification jealously. Seeding is not a claim:
    a seed says "look here", and being wrong about where a case sits wastes some
    search, whereas a *finding* built on an unverified coding asserts something
    false about a court. The gates that matter sit downstream, in
    ``match_precedents`` and ``emit_precedents``, and they still refuse a draft.

    ``budget`` caps total seeds. Cases are taken in order of how few completions
    they need, so a well-coded record is never crowded out by one blank on four
    axes -- and whatever is dropped is named on the plan.
    """

    plan = SeedPlan()
    cases = corpus.verified() if verified_only else list(corpus.cases)

    expansions: list[tuple[int, str, list[FactVector]]] = []
    for case in cases:
        blanks = [name for name in schema.names if case.features.get(name, "") in WILDCARDS]
        missing = [name for name in schema.names if name not in case.features]
        open_axes = list(dict.fromkeys([*blanks, *missing]))
        fixed = {
            name: value
            for name, value in case.features.items()
            if name in schema.names and name not in open_axes
        }

        candidates: list[FactVector] = []
        for combination in itertools.product(*(schema.axis(a).values for a in open_axes)):
            vector = FactVector({**fixed, **dict(zip(open_axes, combination, strict=True))})
            if mask.feasible(vector):
                candidates.append(vector)
        if not candidates:
            plan.infeasible.append(case.id)
            continue
        expansions.append((len(candidates), case.id, candidates))

    # Cheapest first: a fully coded case costs one seed and must not be dropped
    # so that a case blank on four axes can spend the budget on 60 completions.
    expansions.sort(key=lambda item: (item[0], item[1]))

    for count, case_id, candidates in expansions:
        if len(plan.seeds) + count > budget:
            plan.truncated.append(case_id)
            plan.dropped += count
            continue
        plan.seeds.extend(candidates)
        plan.per_case[case_id] = count

    return plan


def as_regions(seeds: Sequence[FactVector]) -> list[dict[str, str]]:
    """Seeds in the shape the coupling layer already passes priors in."""

    return [seed.as_dict() for seed in seeds]
