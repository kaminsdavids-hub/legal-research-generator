"""``portfolio select`` — choose the citation set from verified authorities.

    portfolio select --graph paper.graph.json --front-size 12

Cheap and run at render time, unlike Divergence. It reads the event log for
coverage requirements earlier epochs created, and writes back the propositions
it found thinly supported — which the next Divergence epoch may act on, never
this one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modules.casework.report import holdout_notice, portfolio_report
from modules.casework.schema import EvaluatorStamp
from modules.coupling.events import EpochManager, EventLog

from .model import Authority, CitatorFlag, PortfolioProblem, Precedential, Proposition
from .nsga import named_strategies, select
from .objectives import thin_coverage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    choose = sub.add_parser("select", help="return the front of citation portfolios")
    choose.add_argument("--graph", type=Path, required=True, help="propositions + candidates JSON")
    choose.add_argument("--front-size", type=int, default=12)
    choose.add_argument("--epoch", type=int, default=0)
    choose.add_argument("--events", type=Path, default=Path(".casework/events.jsonl"))
    choose.add_argument("--home", default="us_federal", help="home jurisdiction")
    choose.add_argument("--half-life", type=float, default=12.0)
    choose.add_argument("--seed", type=int, default=7)
    choose.add_argument("--out", type=Path)
    return parser


def load_problem(path: Path, *, home: str, half_life: float) -> PortfolioProblem:
    """Read propositions and their already-verified candidates.

    Anything marked unverified is refused by ``PortfolioProblem`` itself rather
    than filtered here: a loader that quietly dropped them would make the
    invariant depend on this function being correct, instead of on the type.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    propositions = [
        Proposition(
            id=str(p["id"]),
            text=str(p.get("text", "")),
            contested=bool(p.get("contested", False)),
            breadth_matters=bool(p.get("breadth_matters", False)),
        )
        for p in data["propositions"]
    ]
    candidates: dict[str, list[Authority]] = {}
    for pid, authorities in data["candidates"].items():
        candidates[str(pid)] = [
            Authority(
                id=str(a["id"]),
                record_id=str(a.get("record_id", a["id"])),
                verified=bool(a.get("verified", False)),
                court_level=str(a.get("court_level", "secondary")),
                jurisdiction=str(a.get("jurisdiction", "")),
                year=(int(a["year"]) if a.get("year") is not None else None),
                precedential=Precedential(str(a.get("precedential", "persuasive"))),
                citator=CitatorFlag(str(a.get("citator", "unknown"))),
                relation=str(a.get("relation", "entailed")),
                title=str(a.get("title", "")),
            )
            for a in authorities
        ]
    return PortfolioProblem(
        propositions=propositions,
        candidates=candidates,
        home_jurisdiction=home,
        half_life_years=half_life,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "select":  # pragma: no cover
        return 2

    problem = load_problem(args.graph, home=args.home, half_life=args.half_life)

    log = EventLog.load(args.events)
    epochs = EpochManager(log=log, epoch=args.epoch)
    inherited = epochs.propositions_for(args.epoch)
    if inherited:
        print(
            f"coupling: {len(inherited)} proposition(s) created by Divergence in "
            f"earlier epochs are in the coverage requirement set."
        )
        problem.from_divergence = {str(p.get("case_id", "")) for p in inherited}

    blocked = problem.infeasible()
    for problem_text in blocked:
        print(f"INFEASIBLE {problem_text}", file=sys.stderr)
    if blocked:
        # Not worked around. A proposition whose only authority was overruled
        # cannot be covered, and a portfolio that silently omitted it would be
        # a portfolio that does not do what the type says it does.
        return 1

    front = select(problem, front_size=args.front_size, seed=args.seed)
    if not front:
        print("no portfolio: there is nothing to select from", file=sys.stderr)
        return 1

    thin = thin_coverage(problem, front[0].selection)
    report = portfolio_report(
        front,
        named_strategies(front),
        stamp=EvaluatorStamp(
            schema_digest="n/a",
            rules_digest="n/a",
            epoch=args.epoch,
            seed=args.seed,
            recipe=(
                f"portfolio select --graph {args.graph} "
                f"--front-size {args.front_size} --seed {args.seed}"
            ),
        ),
        excluded=problem.excluded(),
        thin=thin,
    )
    print(report.render())
    print()
    print(holdout_notice({}, []))

    if thin:
        epochs.record_thin_coverage(thin)

    if args.out:
        report.save(Path(str(args.out) + ".report.json"))
        args.out.write_text(
            json.dumps(
                {
                    "epoch": args.epoch,
                    "seed": args.seed,
                    "recipe": (
                        f"portfolio select --graph {args.graph} "
                        f"--front-size {args.front_size} --seed {args.seed}"
                    ),
                    "front": [p.as_dict() for p in front],
                    "strategies": [
                        {"name": s.name, "why": s.why, **s.portfolio.as_dict()}
                        for s in named_strategies(front)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
