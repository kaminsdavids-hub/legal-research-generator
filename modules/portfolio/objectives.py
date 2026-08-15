"""Scoring a citation set. Five objectives, all computed from metadata.

They genuinely compete: the smallest footnote count is one binding case per
proposition, which is also the most concentrated portfolio possible and the one
that ages worst. No weight vector represents an author's real preference between
those, which is why the module returns a front and asks them to choose.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .model import PortfolioProblem, Precedential

__all__ = ["PortfolioScores", "score_selection"]

Selection = Sequence[tuple[str, str]]  # (proposition id, authority id)


@dataclass(frozen=True)
class PortfolioScores:
    count: int
    binding: float
    recency: float
    concentration: float
    spread: float
    #: Every objective as a *minimisation*, in the order NSGA-II receives them.
    #: Negation happens here, once, where the sign is visible next to the name.
    def as_minimised(self) -> list[float]:
        return [
            float(self.count),
            -self.binding,
            -self.recency,
            self.concentration,
            -self.spread,
        ]

    def as_dict(self) -> dict[str, float]:
        return {
            "citations": self.count,
            "binding_weight": round(self.binding, 4),
            "recency": round(self.recency, 4),
            "concentration": round(self.concentration, 4),
            "jurisdictional_spread": round(self.spread, 4),
        }


def score_selection(problem: PortfolioProblem, selection: Selection) -> PortfolioScores:
    by_id = {a.id: a for a in problem.all_authorities()}
    chosen = [by_id[aid] for _, aid in selection]

    # 1. Footnote economy. The count of *citations*, not of distinct sources: a
    # reader meets every footnote, even when two of them are the same case.
    count = len(selection)

    # 2. Binding weight, averaged so it does not simply reward citing more.
    binding = (
        sum(a.binding_weight(problem.home_jurisdiction) for a in chosen) / len(chosen)
        if chosen
        else 0.0
    )

    # 3. Recency, as exponential decay on a per-area half-life. Decay rather
    # than a cliff: a case does not stop being good law on its twelfth birthday,
    # and a threshold would make the objective jump for a one-year edit.
    recency = 0.0
    if chosen:
        total = 0.0
        for authority in chosen:
            if authority.year is None:
                total += 0.5  # unknown year: neither fresh nor stale
                continue
            age = max(0, problem.current_year - authority.year)
            total += 0.5 ** (age / max(1e-6, problem.half_life_years))
        recency = total / len(chosen)

    # 4. Concentration: Herfindahl over distinct sources. An argument resting
    # 80% on one case is fragile in a way no other objective notices -- if that
    # case is later distinguished, the paper goes with it.
    concentration = 0.0
    if chosen:
        shares = Counter(a.record_id for a in chosen)
        concentration = sum((n / len(chosen)) ** 2 for n in shares.values())

    # 5. Jurisdictional spread, counted only on the propositions the author
    # marked. Breadth is not a virtue everywhere: on a question of one
    # circuit's law, citing five jurisdictions is padding.
    marked = [p for p in problem.propositions if p.breadth_matters]
    spread = 0.0
    if marked:
        per_proposition = []
        for proposition in marked:
            jurisdictions = {
                by_id[aid].jurisdiction
                for pid, aid in selection
                if pid == proposition.id and by_id[aid].jurisdiction
            }
            per_proposition.append(len(jurisdictions))
        spread = sum(per_proposition) / len(per_proposition)

    return PortfolioScores(
        count=count,
        binding=binding,
        recency=recency,
        concentration=concentration,
        spread=spread,
    )


def thin_coverage(
    problem: PortfolioProblem,
    selection: Selection,
    *,
    stale_after: float = 2.0,
) -> list[dict[str, object]]:
    """Propositions supported only thinly, for the coupling layer to act on.

    Thin means one of three things, and the reason is reported rather than
    collapsed into a score: a single authority, nothing binding, or everything
    older than ``stale_after`` half-lives. Thin support is where an adversarial
    hypothetical bites hardest, which is why this feeds Divergence's prior for
    the *next* epoch.
    """

    by_id = {a.id: a for a in problem.all_authorities()}
    findings: list[dict[str, object]] = []

    for proposition in problem.propositions:
        chosen = [by_id[aid] for pid, aid in selection if pid == proposition.id]
        if not chosen:
            continue
        reasons: list[str] = []
        if len({a.record_id for a in chosen}) == 1:
            reasons.append("single source")
        if all(a.precedential is Precedential.PERSUASIVE for a in chosen):
            reasons.append("nothing binding")
        years = [a.year for a in chosen if a.year is not None]
        if years and all(
            (problem.current_year - year) / max(1e-6, problem.half_life_years) > stale_after
            for year in years
        ):
            reasons.append("all stale")
        if any(a.relation == "rule_support" for a in chosen) and len(chosen) == 1:
            reasons.append("rule support only")
        if reasons:
            findings.append(
                {
                    "proposition": proposition.id,
                    "reasons": reasons,
                    "authorities": [a.id for a in chosen],
                }
            )
    return findings
