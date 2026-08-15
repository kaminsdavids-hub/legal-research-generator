"""NSGA-II over the selection matrix, and the named strategies drawn from it.

The genotype is a binary vector over (proposition x candidate). Coverage and the
contested rule are enforced by *repair* rather than by penalty: a portfolio that
leaves a proposition uncited is not a worse portfolio, it is not a portfolio, and
a penalty term lets the search trade one away for footnote economy.

The front is the deliverable. The named strategies are a reading aid pulled from
it — an author choosing "minimal footnotes" over "maximum authority" is making an
editorial decision, and the record says `[stated]` rather than presenting it as
the computed answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .model import Authority, PortfolioProblem, Precedential
from .objectives import PortfolioScores, score_selection, thin_coverage

__all__ = ["Portfolio", "Strategy", "named_strategies", "select"]


@dataclass(frozen=True)
class Portfolio:
    selection: list[tuple[str, str]]
    scores: PortfolioScores
    manual_citator_checks: list[str] = field(default_factory=list)
    thin: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "selection": [{"proposition": p, "authority": a} for p, a in self.selection],
            **self.scores.as_dict(),
            "manual_citator_check": self.manual_citator_checks,
            "thin_coverage": self.thin,
        }


@dataclass(frozen=True)
class Strategy:
    name: str
    why: str
    portfolio: Portfolio


def _slots(problem: PortfolioProblem) -> list[tuple[str, Authority]]:
    """Every (proposition, selectable authority) pair, in a stable order."""

    slots: list[tuple[str, Authority]] = []
    for proposition in problem.propositions:
        for authority in problem.selectable(proposition.id):
            slots.append((proposition.id, authority))
    return slots


def _repair(problem: PortfolioProblem, bits: Sequence[int], slots) -> list[tuple[str, str]]:
    """Turn a bit vector into a portfolio that satisfies every hard constraint.

    Repair, not penalty. Under-covered propositions get their best available
    authority added; a contested proposition short of two distinct sources gets
    the best authority from a source it does not already have. Over-selection is
    left alone — the count objective handles it, and trimming here would be the
    repair function quietly optimising.
    """

    chosen: list[tuple[str, str]] = [
        (pid, authority.id) for bit, (pid, authority) in zip(bits, slots, strict=False) if bit
    ]

    for proposition in problem.propositions:
        pool = problem.selectable(proposition.id)
        if not pool:
            continue  # reported by problem.infeasible(); nothing to repair
        picked = [aid for pid, aid in chosen if pid == proposition.id]
        by_id = {a.id: a for a in pool}
        ranked = sorted(
            pool,
            key=lambda a: (
                -a.binding_weight(problem.home_jurisdiction),
                -(a.year or 0),
                a.id,
            ),
        )
        if not picked:
            chosen.append((proposition.id, ranked[0].id))
            picked = [ranked[0].id]
        if proposition.contested:
            sources = {by_id[aid].record_id for aid in picked}
            for candidate in ranked:
                if len(sources) >= 2:
                    break
                if candidate.record_id not in sources:
                    chosen.append((proposition.id, candidate.id))
                    sources.add(candidate.record_id)
    return sorted(set(chosen))


def select(
    problem: PortfolioProblem,
    *,
    front_size: int = 12,
    population: int = 80,
    generations: int = 60,
    seed: int = 7,
) -> list[Portfolio]:
    """The Pareto front of citation portfolios, deduplicated and trimmed."""

    from modules.casework.search import nsga2_front

    slots = _slots(problem)
    if not slots:
        return []

    def evaluate(genome: Sequence[int]) -> list[float]:
        selection = _repair(problem, genome, slots)
        return score_selection(problem, selection).as_minimised()

    front = nsga2_front(
        evaluate,
        n_var=len(slots),
        bounds=[1] * len(slots),
        n_obj=5,
        population=population,
        generations=generations,
        seed=seed,
    )

    seen: set[tuple[tuple[str, str], ...]] = set()
    portfolios: list[Portfolio] = []
    for genome, _ in front:
        selection = _repair(problem, genome, slots)
        key = tuple(selection)
        if key in seen:
            continue
        seen.add(key)
        portfolios.append(
            Portfolio(
                selection=selection,
                scores=score_selection(problem, selection),
                manual_citator_checks=problem.manual_citator_checks(selection),
                thin=thin_coverage(problem, selection),
            )
        )

    portfolios.sort(key=lambda p: (p.scores.count, -p.scores.binding))
    return portfolios[:front_size]


def named_strategies(portfolios: Sequence[Portfolio]) -> list[Strategy]:
    """A handful of points from the front, named for what they optimise.

    Three, not one. Presenting a single "best" portfolio would be exactly the
    scalarisation this module exists to avoid — and the author's pick among
    these is an editorial decision, recorded as `[stated]`.
    """

    if not portfolios:
        return []

    def best(key, name: str, why: str) -> Strategy:
        return Strategy(name=name, why=why, portfolio=min(portfolios, key=key))

    strategies = [
        best(
            lambda p: (p.scores.count, -p.scores.binding),
            "minimal footnotes",
            "the fewest citations that still cover every proposition",
        ),
        best(
            lambda p: (-p.scores.binding, p.scores.count),
            "maximum authority weight",
            "the heaviest binding authority available, footnote count aside",
        ),
        best(
            lambda p: (-p.scores.spread, p.scores.concentration, p.scores.count),
            "broadest jurisdictional base",
            "widest jurisdictional coverage where the author marked breadth as mattering",
        ),
    ]
    # A front can be small enough that two names land on one portfolio. Saying
    # so is better than presenting the same selection three times under
    # different labels.
    unique: list[Strategy] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for strategy in strategies:
        key = tuple(strategy.portfolio.selection)
        if key in seen:
            continue
        seen.add(key)
        unique.append(strategy)
    return unique


def binding_share(problem: PortfolioProblem, portfolio: Portfolio) -> float:
    by_id = {a.id: a for a in problem.all_authorities()}
    chosen = [by_id[aid] for _, aid in portfolio.selection]
    if not chosen:
        return 0.0
    return sum(1 for a in chosen if a.precedential is Precedential.BINDING) / len(chosen)
