"""Real cases coded into the feature schema, and how far to trust each coding.

This file is the module's real cost, and the README says so plainly. Coding a
case means reading it and deciding where it sits on every axis, which is
scholarship, not data entry. Budget a hand-coded seed of 30-60 leading cases.

**Verification status is a field on every record, never an assumption.** A
model may draft a coding; a human confirms it. An unverified coding is usable
for exploration and must not be reported as a precedent collision, because the
whole force of that finding is "your framework misclassifies a real decision",
and that claim is worth nothing if the coding of the decision was itself
guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from .schema import INAPPLICABLE, UNDETERMINED, WILDCARDS, FeatureSchema

__all__ = ["CaseCorpus", "CodedCase", "Coding", "load_cases"]


class Coding(StrEnum):
    #: A human read the case and coded it.
    HUMAN = "human"
    #: A model drafted the coding and a human confirmed it, axis by axis.
    MODEL_VERIFIED = "model_verified"
    #: A model drafted it and nobody has checked. Explorable, never citable as
    #: a precedent collision.
    MODEL_DRAFT = "model_draft"

    @property
    def verified(self) -> bool:
        return self in (Coding.HUMAN, Coding.MODEL_VERIFIED)


@dataclass(frozen=True)
class CodedCase:
    id: str
    citation: str
    name: str
    features: Mapping[str, str]
    #: What the court actually held, in the same vocabulary the rules return, so
    #: a collision is a comparison and not an interpretation.
    outcome: str
    coding: Coding = Coding.MODEL_DRAFT
    coded_by: str = ""
    #: Where the *citation* came from, which is a separate question from where
    #: the *coding* came from. A citation confirmed against a primary source is
    #: a fact; the coding of the case onto these axes is a judgement. Conflating
    #: them would let "I checked the reporter" stand in for "I read the case".
    citation_source: str = ""
    note: str = ""

    @property
    def citable(self) -> bool:
        return self.coding.verified

    @property
    def undetermined(self) -> tuple[str, ...]:
        """Axes this coding declines to answer, because the source does not."""

        return tuple(
            sorted(name for name, value in self.features.items() if value == UNDETERMINED)
        )

    @property
    def inapplicable(self) -> tuple[str, ...]:
        """Axes the case has no position on, because the question does not arise."""

        return tuple(
            sorted(name for name, value in self.features.items() if value == INAPPLICABLE)
        )

    @property
    def blanks(self) -> tuple[str, ...]:
        """Every axis this coding leaves open, of either kind.

        The matcher tightens its tolerance by this count. Both sentinels are
        free in distance, so without that they compound: a case blank on two
        axes and allowed one step of tolerance could differ on three, while a
        fully coded case could differ on one.
        """

        return tuple(sorted((*self.undetermined, *self.inapplicable)))

    def applicability(self, schema: FeatureSchema) -> float:
        """Fraction of the schema's axes that mean anything for this case.

        Low applicability is not a coding defect, it is a subject-matter
        mismatch: the case is about something the schema does not describe.
        ``data/cases.yaml`` already drops sixteen confirmed citations on exactly
        this ground, in a comment. This is the same judgement as a number, so
        the matcher can apply it instead of trusting that the comment stayed
        true.
        """

        axes = len(schema.axes)
        if not axes:
            return 0.0
        return (axes - len(self.inapplicable)) / axes

    def determinacy(self, schema: FeatureSchema) -> float:
        """Fraction of the *applicable* axes this coding actually answers.

        The price of the wildcard: a wildcard costs nothing in distance, so a
        case blank on six of eight axes sits within tolerance of most of the
        space and would be reported as the precedent behind findings that have
        almost nothing to do with it.

        Only ``UNDETERMINED`` lowers this. Inapplicable axes leave the
        denominator instead, because determinacy asks how completely the source
        was mined, and an axis the case has no position on was never there to
        find. Charging for it would hold a case permanently below the floor for
        a reason no amount of reading could fix -- and, worse, would file it as
        a research lead pointing at an answer that does not exist.
        :meth:`applicability` is the guard against the obvious abuse.
        """

        applicable = len(schema.axes) - len(self.inapplicable)
        if applicable <= 0:
            return 0.0
        return (applicable - len(self.undetermined)) / applicable


@dataclass
class CaseCorpus:
    cases: list[CodedCase] = field(default_factory=list)
    source: str = ""

    def verified(self) -> list[CodedCase]:
        return [case for case in self.cases if case.citable]

    def validate_against(self, schema: FeatureSchema) -> list[str]:
        """Complaints about codings that do not fit the schema.

        Returned rather than raised: a corpus outliving a schema change is
        normal, and the author needs the list to fix it, not a stack trace on
        the first bad record.
        """

        problems: list[str] = []
        for case in self.cases:
            try:
                schema.validate_coding(case.features)
            except ValueError as exc:
                problems.append(f"{case.id}: {exc}")
        return problems

    def coverage(self, schema: FeatureSchema) -> dict[str, dict[str, int]]:
        """How many coded cases sit at each axis value.

        The realism prior reads this, and an author should see it before
        trusting that prior: an axis where 28 of 30 cases share one value tells
        the search that value is "realistic" and everything else is exotic,
        which may say more about what has been coded than about the world.

        Both wildcards get rows of their own rather than being dropped, and
        they are worth reading apart. An axis where four of six cases are
        *undetermined* is unanswered, and more reading may answer it. An axis
        where four of six are *inapplicable* is an axis most of the corpus has
        no position on -- which is a question about whether the axis belongs in
        the schema, not about the cases.
        """

        counts: dict[str, dict[str, int]] = {
            axis.name: dict.fromkeys((*axis.values, *WILDCARDS), 0) for axis in schema.axes
        }
        for case in self.verified():
            for name, value in case.features.items():
                if name in counts and value in counts[name]:
                    counts[name][value] += 1
        return counts


def load_cases(path: str | Path) -> CaseCorpus:
    data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases = [
        CodedCase(
            id=str(entry["id"]),
            citation=str(entry.get("citation", "")),
            name=str(entry.get("name", "")),
            features={str(k): str(v) for k, v in (entry.get("features") or {}).items()},
            outcome=str(entry.get("outcome", "")),
            coding=Coding(str(entry.get("coding", "model_draft"))),
            coded_by=str(entry.get("coded_by", "")),
            citation_source=str(entry.get("citation_source", "")),
            note=str(entry.get("note", "")),
        )
        for entry in (data.get("cases") or [])
    ]
    return CaseCorpus(cases=cases, source=str(path))
