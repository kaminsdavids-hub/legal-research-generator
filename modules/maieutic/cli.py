"""``maieutic`` — drive the loop from a terminal.

Thin by design: argv in, text out. Everything that decides anything lives in
:mod:`modules.maieutic.loop`, so the behaviour can be tested without going
through stdout.

:func:`build_parser` is separate from :func:`main` because a CLI whose parser is
built inline is a CLI no test can reach. That is not hypothetical here — an
earlier tool in this repository was broken for every invocation and pushed, while
98 tests passed, because the suite never invoked it (REMEDIATION §11.11c).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loop import GateReport, Gates, Session, StepResult
from .render import Audience
from .socratic import SocraticEngine

DEFAULT_STATE = Path(".maieutic/session.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maieutic",
        description="Socratic questions in, an argued manuscript out.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help=f"session file (default: {DEFAULT_STATE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    begin = sub.add_parser("begin", help="open a manuscript with your thesis")
    begin.add_argument("thesis")
    begin.add_argument("--section", default="")

    sub.add_parser("ask", help="show the next question")

    answer = sub.add_parser("answer", help="answer the pending question")
    answer.add_argument("text")

    cycle = sub.add_parser("cycle", help="ask and answer until you stop")
    cycle.add_argument(
        "--rounds", type=int, default=0, help="stop after N rounds (0 = until done)"
    )

    sub.add_parser("status", help="gaps, open problems and what is outstanding")

    show = sub.add_parser("render", help="print the manuscript")
    show.add_argument(
        "--review",
        action="store_true",
        help="annotate provenance: what you wrote versus what you accepted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = Session.load(args.state)
    gates = Gates.offline()
    engine = SocraticEngine()

    if args.command == "begin":
        if session.graph.nodes:
            print("This manuscript has already been started.", file=sys.stderr)
            return 1
        node = session.begin(args.thesis, args.section)
        print(f"Opened with your thesis ({node.id}).")
        _ask(session, engine)

    elif args.command == "ask":
        if not _ask(session, engine):
            return 0

    elif args.command == "answer":
        if session.pending is None and _ask(session, engine) is None:
            return 0
        _report(session.answer(args.text, gates))

    elif args.command == "cycle":
        return _cycle(session, engine, gates, args)

    elif args.command == "status":
        _status(session)

    elif args.command == "render":
        print(
            session.manuscript(
                Audience.REVIEW if args.review else Audience.MANUSCRIPT
            ).text,
            end="",
        )

    session.save(args.state)
    return 0


def _cycle(
    session: Session,
    engine: SocraticEngine,
    gates: Gates,
    args: argparse.Namespace,
) -> int:
    """Ask, read an answer, gate, merge. Repeat until the author stops."""
    rounds = 0
    while args.rounds == 0 or rounds < args.rounds:
        question = _ask(session, engine)
        if question is None:
            break

        print("\nYour answer (blank line to stop):")
        try:
            text = input("> ")
        except EOFError:
            break
        if not text.strip():
            break

        _report(session.answer(text, gates))
        # Saved every round. A crash mid-session must not cost the author the
        # answers they already gave.
        session.save(args.state)
        rounds += 1

    session.save(args.state)
    return 0


def _ask(session: Session, engine: SocraticEngine) -> object | None:
    question = session.pending or session.ask(engine)
    if question is None:
        print("No unasked gaps remain.")
        outstanding = session.outstanding()
        if outstanding:
            print(
                f"{len(outstanding)} gap(s) were asked about and are still open; "
                "see `maieutic status`."
            )
        return None
    print(f"\n{question.text}")
    return question


def _report(result: StepResult) -> None:
    if result.merged:
        print(f"\nMerged: {len(result.added)} node(s) added.")
    else:
        print("\nNot merged. The question stays open.")
    for reason in result.refusals:
        print(f"  refused — {reason}")
    if result.report:
        _advise(result.report)
    if result.unresolved:
        print(
            "  note — this gap needs an edit rather than an addition, so your "
            "answer did not close it."
        )
    if result.adaptation and result.adaptation.refused:
        for refusal in result.adaptation.refused:
            print(f"  dropped at the boundary — {refusal}")


def _advise(report: GateReport) -> None:
    for advisory in report.advisories:
        print(f"  advisory — {advisory}")


def _status(session: Session) -> None:
    gaps = session.gaps()
    outstanding = {g.key for g in session.outstanding()}
    print(f"{len(session.graph.nodes)} node(s), {len(gaps)} open gap(s).")
    for gap in gaps:
        mark = " (already asked)" if gap.key in outstanding else ""
        print(f"  {gap.kind.value}: {gap.detail}{mark}")

    manuscript = session.manuscript()
    for problem in manuscript.open_problems:
        print(f"  unanswered objection: {problem}")
    for warning in manuscript.warnings:
        print(f"  warning: {warning}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
