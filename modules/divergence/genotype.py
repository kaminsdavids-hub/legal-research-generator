"""The fact vector, and the feasibility mask that decides which ones exist.

**Infeasible individuals are never created.** Not created and repaired, not
created and penalised — the sampler and every mutation draw only from the
feasible set. Repair biases a search in ways nobody chose, and a penalty term
lets the optimizer trade realism against disagreement, which is exactly the
trade that produces an archive full of incoherent fact patterns.

Realism lives in the mask, where it is a predicate an author can read and argue
with. It does not live in a model asked "is this realistic?", because that puts
a judged step inside the generation loop and makes the feasible set depend on
prompt wording.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from modules.casework.schema import FeatureSchema

__all__ = ["FactVector", "FeasibilityMask", "Rule", "decode", "encode"]


@dataclass(frozen=True)
class FactVector:
    """A point in the feature space: one value per axis."""

    values: Mapping[str, str]

    def __getitem__(self, axis: str) -> str:
        return self.values[axis]

    def with_value(self, axis: str, value: str) -> FactVector:
        return FactVector({**self.values, axis: value})

    def as_dict(self) -> dict[str, str]:
        return dict(self.values)

    def distance(self, other: FactVector, schema: FeatureSchema) -> int:
        """Ordinal-aware distance: the unit brittleness is measured in."""

        return sum(
            axis.distance(self.values[axis.name], other.values[axis.name])
            for axis in schema.axes
        )


def encode(vector: FactVector, schema: FeatureSchema) -> list[int]:
    return [schema.axis(name).index(vector[name]) for name in schema.names]


def decode(genome: Sequence[int], schema: FeatureSchema) -> FactVector:
    values = {}
    for index, name in enumerate(schema.names):
        axis = schema.axis(name)
        # Clamp rather than raise: pymoo's integer operators round into range,
        # and a genome one past the end is an operator artifact, not a bug in
        # the schema.
        position = min(max(int(genome[index]), 0), len(axis.values) - 1)
        values[name] = axis.values[position]
    return FactVector(values)


#: A named predicate over a fact vector. Named because a rejected combination
#: should be explainable ("a foreign state entity cannot be a hobbyist") rather
#: than silently absent from the archive.
Rule = Callable[[FactVector], bool]


@dataclass
class FeasibilityMask:
    schema: FeatureSchema
    #: (name, predicate). The predicate returns True when the vector is *allowed*.
    constraints: list[tuple[str, Rule]] = field(default_factory=list)

    def require(self, name: str, predicate: Rule) -> FeasibilityMask:
        self.constraints.append((name, predicate))
        return self

    def violations(self, vector: FactVector) -> list[str]:
        return [name for name, predicate in self.constraints if not predicate(vector)]

    def feasible(self, vector: FactVector) -> bool:
        return not self.violations(vector)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def sample(self, rng: random.Random, attempts: int = 200) -> FactVector:
        """A uniformly-drawn feasible vector.

        Rejection sampling, bounded. If the mask is so tight that this fails,
        that is a fact about the mask the author needs to know — an exhausted
        sampler raises rather than returning the last infeasible draw, which
        would put an incoherent fact pattern into the archive wearing the same
        label as the rest.
        """

        for _ in range(attempts):
            vector = FactVector(
                {axis.name: rng.choice(list(axis.values)) for axis in self.schema.axes}
            )
            if self.feasible(vector):
                return vector
        raise RuntimeError(
            f"no feasible fact vector in {attempts} draws; the mask may be "
            f"over-constrained ({len(self.constraints)} constraints over "
            f"{self.schema.size()} combinations)"
        )

    def neighbours(self, vector: FactVector) -> Iterator[FactVector]:
        """Every feasible vector one axis-step away.

        One *step*, not one axis: on an ordinal axis the neighbours are the
        adjacent levels, so "capability tier 2 -> 3" is a neighbour and
        "tier 2 -> 5" is not. Brittleness measured over whole-axis jumps would
        call almost everything knife-edge.
        """

        for axis in self.schema.axes:
            current = vector[axis.name]
            index = axis.index(current)
            candidates = (
                [index - 1, index + 1]
                if axis.kind.value == "ordinal"
                else [i for i in range(len(axis.values)) if i != index]
            )
            for candidate in candidates:
                if not 0 <= candidate < len(axis.values):
                    continue
                moved = vector.with_value(axis.name, axis.values[candidate])
                if self.feasible(moved):
                    yield moved

    def sample_genome(self, rng: random.Random) -> list[int]:
        return encode(self.sample(rng), self.schema)

    def repair_genome(self, genome: Sequence[int]) -> list[int]:
        """Snap a genome to the nearest feasible vector, for library operators.

        Used only where a third-party operator (pymoo's crossover) can produce
        an out-of-set individual. The MAP-Elites path never calls this: it
        samples feasibly by construction. When repair does run it walks
        neighbours rather than resampling, so the child keeps its parents'
        structure instead of being replaced by a stranger.
        """

        vector = decode(genome, self.schema)
        if self.feasible(vector):
            return encode(vector, self.schema)
        for candidate in self.neighbours(vector):
            if self.feasible(candidate):
                return encode(candidate, self.schema)
        return encode(self.sample(random.Random(sum(genome))), self.schema)
