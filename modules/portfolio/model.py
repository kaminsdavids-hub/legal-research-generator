"""What a citation portfolio is selected from, and the invariant on the pool.

**The candidate pool is the verified-authority set, and nothing else.** Not a
preference the objectives trade against — a hard invariant, enforced where the
pool is built, so no combination of weights can produce a selection containing
an authority that has not passed retrieval and entailment. Portfolio chooses
*among* citations the existing gate already accepted; it can never introduce one.

Everything scored here is metadata: court level, jurisdiction, year, precedential
status, citator flags. No model reads a case and gives an opinion inside the
loop, which is what makes it safe to run an evolutionary search over the
objectives at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Authority",
    "CitatorFlag",
    "PortfolioProblem",
    "Precedential",
    "Proposition",
    "UnverifiedAuthority",
]


class UnverifiedAuthority(Exception):
    """Raised when an unverified authority is offered to the selector."""


class Precedential(StrEnum):
    BINDING = "binding"
    PERSUASIVE = "persuasive"


class CitatorFlag(StrEnum):
    #: Treatment is known and positive/neutral.
    GOOD = "good"
    #: Known negative treatment. Excluded outright.
    OVERRULED = "overruled"
    ABROGATED = "abrogated"
    CRITICISED = "criticised"
    #: No citator data. Selectable, and flagged for a human -- never silently
    #: assumed good, because "we had no data" and "we checked and it is fine"
    #: are different claims and only one of them is safe to publish behind.
    UNKNOWN = "unknown"

    @property
    def negative(self) -> bool:
        return self in (CitatorFlag.OVERRULED, CitatorFlag.ABROGATED, CitatorFlag.CRITICISED)


#: Court weight for the binding objective. Ordinal and coarse on purpose: the
#: difference between a circuit and a district court matters; the difference
#: between two district courts does not, and pretending to rank them would put
#: false precision into an objective the author is asked to trust.
COURT_WEIGHT: dict[str, float] = {
    "supreme": 1.0,
    "appellate": 0.75,
    "trial": 0.45,
    "agency": 0.35,
    "secondary": 0.2,
}


@dataclass(frozen=True)
class Authority:
    id: str
    record_id: str
    #: True only if this citation passed the grounding gate: retrieval resolved
    #: it and the support test accepted it. The pool refuses anything else.
    verified: bool
    court_level: str = "secondary"
    jurisdiction: str = ""
    year: int | None = None
    precedential: Precedential = Precedential.PERSUASIVE
    citator: CitatorFlag = CitatorFlag.UNKNOWN
    #: How the support was established, carried through from the verifier:
    #: "entailed" or "rule_support". A rule-supported citation is weaker
    #: evidence and the report says so rather than averaging it away.
    relation: str = "entailed"
    title: str = ""

    @property
    def needs_citator_check(self) -> bool:
        return self.citator is CitatorFlag.UNKNOWN

    def binding_weight(self, home_jurisdiction: str) -> float:
        """Court hierarchy times jurisdictional match.

        A binding decision from another jurisdiction is not binding here, which
        is why the product is taken rather than the maximum: an appellate case
        from the wrong circuit should not outrank a trial court in the right one
        purely on court level.
        """

        base = COURT_WEIGHT.get(self.court_level, 0.2)
        if self.precedential is Precedential.BINDING and self.jurisdiction == home_jurisdiction:
            return base
        if self.jurisdiction == home_jurisdiction:
            return base * 0.7
        return base * 0.4


@dataclass(frozen=True)
class Proposition:
    id: str
    text: str
    #: Requires at least two authorities from distinct sources.
    contested: bool = False
    #: Jurisdictional spread is an objective for this proposition.
    breadth_matters: bool = False


@dataclass
class PortfolioProblem:
    propositions: list[Proposition]
    #: proposition id -> the authorities available for it.
    candidates: dict[str, list[Authority]]
    home_jurisdiction: str = ""
    #: Years after which an authority counts as half as current. Per-area and
    #: author-set: constitutional doctrine and export-control regulation do not
    #: age at the same rate.
    half_life_years: float = 12.0
    current_year: int = 2026
    #: Propositions added by a Divergence MishandledPrecedent this epoch. Kept
    #: so a report can say which coverage requirements the other module created.
    from_divergence: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        offenders = [
            f"{authority.id} (proposition {pid})"
            for pid, authorities in self.candidates.items()
            for authority in authorities
            if not authority.verified
        ]
        if offenders:
            raise UnverifiedAuthority(
                "the candidate pool may contain only authorities that passed the "
                "grounding gate; refused: " + ", ".join(sorted(offenders)[:5])
            )

    def excluded(self) -> dict[str, list[str]]:
        """Authorities dropped for negative citator treatment, by proposition."""

        return {
            pid: [a.id for a in authorities if a.citator.negative]
            for pid, authorities in self.candidates.items()
            if any(a.citator.negative for a in authorities)
        }

    def selectable(self, proposition_id: str) -> list[Authority]:
        """The pool for one proposition, after citator exclusion.

        Exclusion happens here rather than as an objective penalty. An overruled
        case is not a worse citation, it is not a citation, and letting the
        search trade it against footnote economy would eventually publish one.
        """

        return [a for a in self.candidates.get(proposition_id, []) if not a.citator.negative]

    def infeasible(self) -> list[str]:
        """Coverage requirements no selection can satisfy.

        Reported before the search runs. A proposition whose only authority was
        overruled cannot be covered, and the honest answer is to say so -- not
        to return a portfolio that quietly omits it.
        """

        problems: list[str] = []
        for proposition in self.propositions:
            pool = self.selectable(proposition.id)
            if not pool:
                problems.append(
                    f"{proposition.id}: no verified authority survives citator exclusion"
                )
            elif proposition.contested and len({a.record_id for a in pool}) < 2:
                problems.append(
                    f"{proposition.id}: marked contested but only "
                    f"{len({a.record_id for a in pool})} distinct source(s) available"
                )
        return problems

    def all_authorities(self) -> list[Authority]:
        seen: dict[str, Authority] = {}
        for authorities in self.candidates.values():
            for authority in authorities:
                seen.setdefault(authority.id, authority)
        return list(seen.values())

    def manual_citator_checks(self, selection: Sequence[tuple[str, str]]) -> list[str]:
        """Selected authorities with no citator data, for the output to flag."""

        by_id = {a.id: a for a in self.all_authorities()}
        return sorted(
            {
                authority_id
                for _, authority_id in selection
                if by_id[authority_id].needs_citator_check
            }
        )
