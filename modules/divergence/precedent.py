"""Precedent collision: when the hard case the search found is a real decision.

The module's most valuable output, and it is a different claim from everything
else it produces. A `Hypothetical` says "here is an edge case your framework
handles badly". A `MishandledPrecedent` says "here is a case a court has already
decided, and your framework gets it wrong". The second is not a thought
experiment an author can wave off.

Which is exactly why the coding has to be verified. The force of the finding
rests entirely on the claim that the real case sits where the corpus says it
sits, so an unverified coding cannot produce one — it produces a candidate for
someone to check, labelled as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from modules.casework.corpus import CaseCorpus, CodedCase
from modules.casework.schema import UNDETERMINED, FeatureSchema, Verdict

from .genotype import FactVector
from .objectives import Scores

__all__ = ["Collision", "Hypothetical", "MishandledPrecedent", "match_precedents"]


@dataclass(frozen=True)
class Hypothetical:
    """A hard case with no real decision at this point in the space."""

    facts: dict[str, str]
    scores: dict[str, float]
    verdicts: tuple[str, ...]


@dataclass(frozen=True)
class MishandledPrecedent:
    """A coded real case your rules classify inconsistently."""

    case_id: str
    citation: str
    name: str
    #: Distance between the coded case and the fact pattern the search found.
    #: Zero means the search landed on the case itself.
    distance: int
    facts: dict[str, str]
    #: What the court held, from the corpus.
    held: str
    #: What the readings say, which is the collision.
    verdicts: tuple[str, ...]
    scores: dict[str, float]
    #: Verified codings only ever reach here; carried so the report can name
    #: who checked it rather than asserting the finding anonymously.
    coded_by: str = ""
    #: Readings whose verdict contradicts the held outcome. The sharpest form of
    #: the finding: not "your readings disagree" but "this reading is wrong
    #: about a decided case".
    contradicting: tuple[str, ...] = ()


@dataclass
class Collision:
    precedents: list[MishandledPrecedent] = field(default_factory=list)
    hypotheticals: list[Hypothetical] = field(default_factory=list)
    #: Cases that matched but whose coding nobody has verified. Reported apart
    #: from the finding, because a collision built on a guessed coding is a
    #: guess with a citation attached.
    unverified_candidates: list[str] = field(default_factory=list)
    #: Verified codings too silent to anchor a finding: they matched only
    #: because their undetermined axes waive the comparison. Named, because
    #: "this case would be the precedent here if the record settled X" is a
    #: research lead, and dropping it silently would hide one.
    too_undetermined: list[str] = field(default_factory=list)
    #: Codings the schema does not describe: too many axes have no bearing on
    #: them. Reported apart from ``too_undetermined`` because the remedy is the
    #: opposite one. A thin record is worth chasing; a case about a different
    #: subject is worth removing from the corpus, and filing it as a lead would
    #: send someone after an answer that does not exist.
    off_schema: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.precedents)} mishandled precedent(s), "
            f"{len(self.hypotheticals)} hypothetical(s)"
            + (
                f", {len(self.unverified_candidates)} unverified candidate(s) "
                "needing a human coding check"
                if self.unverified_candidates
                else ""
            )
            + (
                f", {len(self.too_undetermined)} coding(s) too undetermined to anchor one"
                if self.too_undetermined
                else ""
            )
            + (
                f", {len(self.off_schema)} off-schema"
                if self.off_schema
                else ""
            )
        )


def _tolerance_for(
    case: CodedCase, tolerance: int, schema: FeatureSchema | None = None
) -> int:
    """How much slack this coding gets, after paying for the axes it left open.

    Floored at zero rather than going negative, so a coding blank on more axes
    than the tolerance is not excluded outright -- it must simply agree exactly
    on every axis it did answer. That is the strongest form of the match still
    available to it, and it keeps the guarantee the sentinels were added for: a
    coding is never penalised for admitting what it does not know, only stopped
    from being rewarded for it.

    ``schema`` is accepted for the unverified path, where a coding may predate an
    axis entirely; those missing axes are blanks too and are counted here.
    """

    blanks = set(case.blanks)
    if schema is not None:
        blanks |= set(schema.names) - set(case.features)
    return max(0, tolerance - len(blanks))


def match_precedents(
    schema: FeatureSchema,
    corpus: CaseCorpus,
    findings: Sequence[tuple[FactVector, Scores, Mapping[str, Verdict]]],
    *,
    tolerance: int = 1,
    min_disagreement: float = 0.3,
    min_determinacy: float = 0.75,
    min_applicability: float = 0.625,
) -> Collision:
    """Split search findings into real-case collisions and hypotheticals.

    ``findings`` carries ``(facts, scores, verdicts_by_rule)``; the verdicts are
    passed in rather than recomputed so this function stays free of the rule
    evaluator and can be tested on fixtures. They are :class:`Verdict` objects
    and compared on ``.label``, so a reading that permits under strict scrutiny
    is not silently treated as agreeing with one that permits outright.

    ``tolerance`` is in schema distance, and 1 is deliberate. A coded case one
    ordinal step from the found pattern is the same case for this purpose --
    coding is not precise enough to distinguish adjacent capability tiers, and
    pretending otherwise would report near-misses as exact hits.

    It is spent per case, reduced by that coding's blank axes. Blanks are free in
    distance, so left alone they compound with the allowance: United States v.
    Stevens, blank on two axes, matched a capability-tier-5 fact pattern at a
    reported distance of 1 while actually differing on three axes, and was
    written into a manuscript as an objection about "its facts" -- one of which
    was a capability the case has no position on. A coding that declines to
    answer an axis is not thereby entitled to a wider net; it forgoes the
    comparison on that axis and must match exactly on the rest.

    ``min_determinacy`` guards the other end. Wildcard axes cost nothing in
    distance, so a coding blank on three of eight axes sits within tolerance of
    a large slab of the space and would be named as the precedent behind
    findings it has no bearing on. At 0.75 a case may leave a quarter of its
    applicable axes unanswered and still anchor a finding; beyond that it is
    listed under ``too_undetermined``, which says the true thing -- the case is
    nearby, and the record is too thin to say it governs.

    ``min_applicability`` is the same guard for the other sentinel, and it is
    checked first because it answers a prior question. A case where three of
    eight axes do not arise is not an incompletely-read case, it is a case about
    something this schema does not describe, and its high determinacy over the
    few axes that do apply would otherwise let it match half the space. Those
    are reported as ``off_schema``: the corpus is wrong to contain them, which
    is the author's call and not something to fix by reading harder.
    """

    result = Collision()
    verified, silent, off_schema = [], [], []
    for case in corpus.verified():
        if case.applicability(schema) < min_applicability:
            off_schema.append(case)
        elif case.determinacy(schema) < min_determinacy:
            silent.append(case)
        else:
            verified.append(case)
    unverified = [case for case in corpus.cases if not case.citable]

    for facts, scores, verdicts in findings:
        if scores.disagreement < min_disagreement:
            continue

        best: tuple[int, CodedCase] | None = None
        for case in verified:
            coded = FactVector(dict(case.features))
            try:
                distance = facts.distance(coded, schema)
            except KeyError:
                continue  # a coding that predates a schema change
            if distance <= _tolerance_for(case, tolerance) and (
                best is None or distance < best[0]
            ):
                best = (distance, case)

        if best is None:
            for case in unverified:
                # UNDETERMINED, not "": the case this default exists for is a
                # coding written before an axis was added, which has no value
                # for it -- exactly what the sentinel means. The empty string
                # was a non-value that crashed here on any ordinal axis, since
                # `Axis.distance` fell through to `tuple.index("")` and raised
                # ValueError, which the KeyError guard below does not catch.
                coded = FactVector({**dict.fromkeys(schema.names, UNDETERMINED), **case.features})
                try:
                    if facts.distance(coded, schema) <= _tolerance_for(case, tolerance, schema):
                        result.unverified_candidates.append(case.id)
                except KeyError:
                    continue
            for case, bucket in (
                # Only reported when nothing determinate matched: a blank case
                # sitting near a finding that already has a real precedent is
                # not a lead, it is noise.
                *((c, result.too_undetermined) for c in silent),
                *((c, result.off_schema) for c in off_schema),
            ):
                coded = FactVector(dict(case.features))
                try:
                    if facts.distance(coded, schema) <= _tolerance_for(case, tolerance):
                        bucket.append(case.id)
                except KeyError:
                    continue
            result.hypotheticals.append(
                Hypothetical(
                    facts=facts.as_dict(),
                    scores=scores.as_dict(),
                    verdicts=tuple(v.label for v in verdicts.values()),
                )
            )
            continue

        distance, case = best
        contradicting = tuple(
            name
            for name, verdict in verdicts.items()
            if case.outcome and not verdict.label.startswith(case.outcome)
        )
        result.precedents.append(
            MishandledPrecedent(
                case_id=case.id,
                citation=case.citation,
                name=case.name,
                distance=distance,
                facts=facts.as_dict(),
                held=case.outcome,
                verdicts=tuple(v.label for v in verdicts.values()),
                scores=scores.as_dict(),
                coded_by=case.coded_by,
                contradicting=contradicting,
            )
        )

    # Surfaced first, because it is the finding that matters: sort by how many
    # readings contradict the actual holding, then by how close the match was.
    result.precedents.sort(key=lambda p: (-len(p.contradicting), p.distance))
    result.hypotheticals.sort(key=lambda h: -h.scores["disagreement"])
    result.unverified_candidates = sorted(set(result.unverified_candidates))
    result.too_undetermined = sorted(set(result.too_undetermined))
    result.off_schema = sorted(set(result.off_schema))
    return result
