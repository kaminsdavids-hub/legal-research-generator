"""The feature schema and the verdict vocabulary. Author-authored, machine-read.

Both metaheuristic modules in this package are safe to run a search over for one
reason: **their fitness functions are executed, not judged.** Divergence scores a
fact pattern by whether frozen rule predicates disagree, which is a computation.
Portfolio scores a citation set from metadata. No model sits inside either loop.

That constraint is not stylistic. Evolutionary search is extremely good at
finding adversarial optima, so pointing it at a judged objective means it will
discover whatever phrasing the judge happens to like — the same self-satisfying
gate failure this repository has already paid for three times (REMEDIATION §5,
§11.5, §11.11a). Here the objective cannot be gamed by rewording, because no
wording reaches it.

The schema is the author's. A model may *propose* axes; a human freezes them,
and the frozen file's hash is stamped on every artifact so a result can never be
compared against one produced under a different schema.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "INAPPLICABLE",
    "UNDETERMINED",
    "WILDCARDS",
    "Axis",
    "AxisKind",
    "FeatureSchema",
    "Outcome",
    "Scrutiny",
    "Verdict",
    "load_schema",
]

#: What a coding says when the source does not. Junger v. Daley states no key
#: length for any of the five programs at issue, and the panel held the record
#: did not resolve how proximate the harm was -- before this existed, coding the
#: case meant inventing both, because every axis demanded a value.
#:
#: **It is a coding value, never a search value.** It is deliberately absent
#: from ``data/feature_schema.yaml``, so ``Axis.values`` never contains it and
#: the sampler, the mutation operators and the genome encoding cannot produce
#: it. A generated fact pattern is a hypothetical the author constructed; there
#: is no source for it to be silent about, and an archive cell labelled
#: "undetermined capability" would be describing nothing. Only a record of
#: something that actually happened can be incomplete.
#:
#: Because it is not an axis value, adding it did not change the schema digest
#: and did not invalidate archives built before it existed. That is also why
#: this comment is here and not in ``data/feature_schema.yaml``: the digest is
#: the sha256 of that file's raw bytes, comments included, so even *documenting*
#: the wildcard there would have declared every existing archive incomparable.
UNDETERMINED = "undetermined"

#: What a coding says when the question does not arise. Bartnicki v. Vopper is
#: about an intercepted phone call: there is no artifact whose capability could
#: be graded, so ``capability_tier`` is not a fact the opinion omitted, it is a
#: fact the case does not have.
#:
#: The distinction from ``UNDETERMINED`` is what an author can do about it.
#: Undetermined is a hole in the record -- read more, find the district court
#: opinion, and the case may yet govern; that is why an undetermined match is
#: reported as a research lead. Inapplicable is permanent. No amount of reading
#: will give a phone call a capability tier, and listing it as a lead would send
#: someone after an answer that does not exist.
#:
#: So the two are wildcards alike in distance and differ everywhere it matters:
#: inapplicable axes leave the determinacy denominator rather than lowering the
#: score, and a case with too many of them is not an incompletely-coded case but
#: a case about something else -- which the matcher reports as off-schema.
INAPPLICABLE = "inapplicable"

#: Values a coding may carry on any axis. Neither is in ``Axis.values``, so the
#: search cannot reach either: a generated fact pattern has no source to be
#: silent about and no subject matter to fall outside.
WILDCARDS = (UNDETERMINED, INAPPLICABLE)


class AxisKind(StrEnum):
    #: Unordered categories. Distance between any two distinct values is 1.
    CATEGORICAL = "categorical"
    #: Ordered levels. Distance is the number of steps between them, which is
    #: what makes "one tier more capable" a meaningfully small perturbation and
    #: "academic -> foreign state entity" a large one.
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class Axis:
    name: str
    kind: AxisKind
    values: tuple[str, ...]

    def index(self, value: str) -> int:
        return self.values.index(value)

    def distance(self, left: str, right: str) -> int:
        """Steps between two values on this axis.

        Ordinal axes measure real distance; categorical axes are 0 or 1. A
        brittleness search that treated "hobbyist -> firm" as one step and
        "capability tier 1 -> 5" as one step would report both as equally
        knife-edge, which is false about the law and about the world.
        """

        if left == right:
            return 0
        if left in WILDCARDS or right in WILDCARDS:
            # A wildcard, at zero cost -- for both sentinels alike, since
            # distance answers "how far is this pattern from that case" and an
            # axis the case is silent on, or has no position on at all, is no
            # evidence of distance in either direction. Scoring it as a mismatch
            # would push every incompletely-coded case past the matching
            # tolerance and quietly remove it from the precedent path, which is
            # how a coding that admits what it does not know would end up
            # counting for less than one that made it up.
            #
            # The cost of the wildcard is that a case undetermined on many axes
            # matches a great deal; `CodedCase.determinacy` and the floor in
            # `match_precedents` are what stop that.
            return 0
        if self.kind is AxisKind.ORDINAL:
            return abs(self.index(left) - self.index(right))
        return 1


@dataclass(frozen=True)
class FeatureSchema:
    axes: tuple[Axis, ...]
    #: sha256 of the file this was loaded from. Stamped on every artifact: two
    #: archives built under different schemas describe different spaces and
    #: comparing them is meaningless.
    digest: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis name in schema: {sorted(names)}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    def axis(self, name: str) -> Axis:
        for candidate in self.axes:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no such axis: {name!r}")

    def size(self) -> int:
        """How many combinations exist before feasibility is applied."""

        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        return total

    def validate(self, values: Mapping[str, str]) -> None:
        """Check a *fact pattern* against the schema. Strict: no wildcards.

        This is the search space. A generated individual carrying either
        wildcard would mean the optimizer had invented a fact pattern that is
        silent about itself, or outside its own subject matter, so this rejects
        both and :meth:`validate_coding` is the looser check codings get.
        """

        self._validate(values, allow_wildcards=False)

    def validate_coding(self, values: Mapping[str, str]) -> None:
        """Check a *coded case*, allowing both wildcards.

        A coding describes a source. Sources are silent about things, and they
        are about some things and not others.
        """

        self._validate(values, allow_wildcards=True)

    def _validate(self, values: Mapping[str, str], *, allow_wildcards: bool) -> None:
        missing = set(self.names) - set(values)
        extra = set(values) - set(self.names)
        if missing or extra:
            raise ValueError(
                f"fact vector does not match the schema: missing={sorted(missing)}, "
                f"unknown={sorted(extra)}"
            )
        for axis in self.axes:
            value = values[axis.name]
            if allow_wildcards and value in WILDCARDS:
                continue
            if value not in axis.values:
                expected = list(axis.values) + (list(WILDCARDS) if allow_wildcards else [])
                raise ValueError(
                    f"{value!r} is not a value of axis {axis.name!r}; "
                    f"expected one of {expected}"
                )


def load_schema(path: str | Path) -> FeatureSchema:
    raw = Path(path).read_bytes()
    data: Any = yaml.safe_load(raw)
    axes = tuple(
        Axis(
            name=str(entry["name"]),
            kind=AxisKind(str(entry.get("kind", "categorical"))),
            values=tuple(str(v) for v in entry["values"]),
        )
        for entry in data["axes"]
    )
    return FeatureSchema(
        axes=axes,
        digest=hashlib.sha256(raw).hexdigest()[:16],
        source=str(path),
    )


# --------------------------------------------------------------------------- #
# What a rule returns
# --------------------------------------------------------------------------- #
class Outcome(StrEnum):
    PERMITTED = "permitted"
    RESTRICTED = "restricted"
    #: The reading does not resolve this fact pattern. A first-class answer, not
    #: a failure: a rule that returns `uncertain` where it genuinely has nothing
    #: to say is more honest than one forced to pick, and the disagreement
    #: measure treats it as its own verdict rather than as a missing one.
    UNCERTAIN = "uncertain"


class Scrutiny(StrEnum):
    NONE = "none"
    RATIONAL_BASIS = "rational_basis"
    INTERMEDIATE = "intermediate"
    STRICT = "strict"


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    #: Only where the reading supports one. A rule that reaches an outcome
    #: without a tier says so rather than inventing one.
    scrutiny: Scrutiny = Scrutiny.NONE
    rationale: str = ""

    @property
    def label(self) -> str:
        """What the disagreement measure partitions on.

        Scrutiny is part of the identity: two readings that both permit, one on
        rational-basis and one on strict scrutiny, disagree about something a
        court would spend a whole opinion on.
        """

        return (
            self.outcome.value
            if self.scrutiny is Scrutiny.NONE
            else f"{self.outcome.value}/{self.scrutiny.value}"
        )


@dataclass(frozen=True)
class EvaluatorStamp:
    """What has to match for two results to be comparable.

    Every artifact carries one. A search whose rules changed mid-way produced
    an archive whose cells were scored by different functions, and nothing in
    the file would say so.
    """

    schema_digest: str
    rules_digest: str
    epoch: int
    seed: int
    recipe: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_digest": self.schema_digest,
            "rules_digest": self.rules_digest,
            "epoch": self.epoch,
            "seed": self.seed,
            "recipe": self.recipe,
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }

    def comparable_to(self, other: EvaluatorStamp) -> bool:
        return (
            self.schema_digest == other.schema_digest
            and self.rules_digest == other.rules_digest
            and self.epoch == other.epoch
        )


def digest_of(paths: Sequence[str | Path]) -> str:
    """Stable hash over a set of files, for freezing rule modules.

    Sorted by name and hashing content, so the digest changes when a predicate
    changes and not when the filesystem reorders directory entries.
    """

    hasher = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]
