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
from typing import Any

from .corpus import from_env
from .learn import Journal
from .loop import (
    GateReport,
    Gates,
    SelfGroundingEdge,
    Session,
    StepResult,
    TurnProvider,
)
from .render import Audience
from .socratic import SocraticEngine

DEFAULT_STATE = Path(".maieutic/session.json")
#: Separate from the session on purpose: evidence about which questions are worth
#: asking accumulates across manuscripts, and one manuscript never supplies enough.
DEFAULT_JOURNAL = Path(".maieutic/journal.json")


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
    parser.add_argument(
        "--journal",
        type=Path,
        default=DEFAULT_JOURNAL,
        help=f"question-history file, shared across manuscripts (default: {DEFAULT_JOURNAL})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    begin = sub.add_parser("begin", help="open a manuscript with your thesis")
    begin.add_argument("thesis")
    begin.add_argument("--section", default="")

    sub.add_parser("ask", help="show the next question")

    live = argparse.ArgumentParser(add_help=False)
    live.add_argument(
        "--live",
        action="store_true",
        help=(
            "run the dialectic engine over your answer, so the machine argues "
            "both sides of it. Needs the configured local models; without this "
            "a merge carries only your own words."
        ),
    )

    answer = sub.add_parser("answer", help="answer the pending question", parents=[live])
    answer.add_argument("text")

    cycle = sub.add_parser("cycle", help="ask and answer until you stop", parents=[live])
    cycle.add_argument(
        "--rounds", type=int, default=0, help="stop after N rounds (0 = until done)"
    )

    sub.add_parser("skip", help="pass on the pending question")

    depends = sub.add_parser(
        "depends",
        help="state that one claim rests on another (DEPENDS_ON)",
        description=(
            "Nothing in this package infers a dependency. It asserts that a "
            "reader must accept one claim before another, which is a claim "
            "about your argument's structure — so you state it, and the "
            "ordering optimizer honours it."
        ),
    )
    depends.add_argument("dependent", help="the claim that rests on something (id or prefix)")
    depends.add_argument("prerequisite", help="what must be read first (id or prefix)")
    depends.add_argument(
        "--retract",
        action="store_true",
        help="take the dependency back. A wrong one is not inert: it reorders "
        "the manuscript and makes the renderer claim a reasoning path the "
        "argument does not contain.",
    )

    sub.add_parser("status", help="gaps, open problems and what is outstanding")

    sub.add_parser(
        "policy", help="what the loop has learned about which questions you answer"
    )

    show = sub.add_parser("render", help="print the manuscript")
    show.add_argument(
        "--review",
        action="store_true",
        help="annotate provenance: what you wrote versus what you accepted",
    )

    training = sub.add_parser(
        "training",
        help="what has been captured for a future fine-tune, and how to export it",
    )
    training.add_argument(
        "--export",
        type=Path,
        default=None,
        help="write chat-format SFT samples here (your answers are the target)",
    )
    training.add_argument(
        "--merged-only",
        action="store_true",
        help="export only exchanges whose patch survived every gate",
    )
    training.add_argument(
        "--include-synthesis",
        action="store_true",
        help="add the machine's synthesis as a further turn (for training the "
        "dialectic role itself, not your voice)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = Session.load(args.state)
    session.journal = Journal.load(args.journal)
    # None unless the author opted in; see modules/maieutic/corpus.py.
    session.training = from_env()
    if session.training is not None and not session.manuscript_id:
        session.manuscript_id = str(args.state)
    engine = SocraticEngine()
    # The gates must match the exchange: a live run produces authorities, and
    # offline gates have no verifier to confirm them. See REMEDIATION 12.2.
    gates = Gates.live(_settings()) if getattr(args, "live", False) else Gates.offline()

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
        _report(session.answer(args.text, gates, _turns(args)))

    elif args.command == "cycle":
        return _cycle(session, engine, gates, args)

    elif args.command == "skip":
        question = session.decline()
        if question is None:
            print("No question is pending.")
        else:
            print("Skipped. It stays an open gap; you will not be asked again.")
            _ask(session, engine)

    elif args.command == "policy":
        _policy(session)

    elif args.command == "status":
        _status(session)

    elif args.command == "depends":
        code = _depends(session, args)
        if code:
            return code

    elif args.command == "training":
        return _training(session, args)

    elif args.command == "render":
        print(
            session.manuscript(
                Audience.REVIEW if args.review else Audience.MANUSCRIPT
            ).text,
            end="",
        )

    session.save(args.state)
    session.journal.save(args.journal)
    return 0


def _settings() -> Any:
    from legal_research.config import get_settings

    return get_settings()


def _resolve(session: Session, needle: str) -> str:
    """A node id from an id or a unique prefix.

    Ids are twelve hex characters. Typing one correctly is not a thing anyone
    does, and an author who mistypes should be told which nodes matched rather
    than watching the command silently address the wrong claim.
    """

    if needle in session.graph.nodes:
        return needle
    matches = [nid for nid in sorted(session.graph.nodes) if nid.startswith(needle)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no node id starts with {needle!r}")
    raise KeyError(
        f"{needle!r} matches {len(matches)} nodes: " + ", ".join(m[:8] for m in matches[:5])
    )


def _depends(session: Session, args: argparse.Namespace) -> int:
    try:
        dependent = _resolve(session, args.dependent)
        prerequisite = _resolve(session, args.prerequisite)
        if args.retract:
            return _retract(session, dependent, prerequisite)
        session.depends_on(dependent, prerequisite)
    except (KeyError, ValueError) as exc:
        # KeyError's str() wraps its message in quotes; the author should read
        # the sentence, not the repr of the sentence.
        print(exc.args[0] if exc.args else str(exc), file=sys.stderr)
        return 1
    except SelfGroundingEdge as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Stated: {dependent[:8]} depends on {prerequisite[:8]}.")
    print(f"  {_excerpt(session.graph.nodes[prerequisite].text)}")
    print("  must be read before")
    print(f"  {_excerpt(session.graph.nodes[dependent].text)}")
    print("\nThe ordering optimizer will keep them in that order and reference "
          "across sections when it cannot keep them together.")
    return 0


def _retract(session: Session, dependent: str, prerequisite: str) -> int:
    """Report what was taken back, or that there was nothing to take back.

    "Nothing to retract" is a success, not an error: the author asked for a
    state in which that dependency does not hold, and it does not. But it is
    said out loud, because the alternative is an author believing they removed
    a constraint that is still shaping their manuscript — most often because
    they gave the two ids the wrong way round.
    """

    if not session.retract_dependency(dependent, prerequisite):
        print(
            f"No stated dependency from {dependent[:8]} to {prerequisite[:8]}; "
            f"nothing retracted. (Check the order of the two ids: this removes "
            f"'{dependent[:8]} depends on {prerequisite[:8]}'.)"
        )
        return 0

    print(f"Retracted: {dependent[:8]} no longer depends on {prerequisite[:8]}.")
    print("The reading order is free to change; re-run the optimizer to see it.")
    return 0


def _excerpt(text: str, limit: int = 96) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _training(session: Session, args: argparse.Namespace) -> int:
    """Report what has been captured, and export it on request.

    Prints the two variables rather than a generic "disabled" when capture is
    off: a feature the author cannot find the switch for is one that will be
    reported as broken.
    """

    corpus = session.training
    if corpus is None:
        print("Training capture is off. Nothing has been written.")
        print("To turn it on — this keeps your answers verbatim, in a file that")
        print("outlives the manuscript, so that a model can later be trained on them:")
        print("  export LRG_MAIEUTIC_TRAINING_CAPTURE=1")
        print("  export LRG_MAIEUTIC_TRAINING_PATH=.maieutic/training.jsonl")
        return 1

    summary = corpus.summary()
    print(f"corpus: {corpus.path}")
    print(
        f"  {summary['samples']} exchange(s), {summary['merged']} merged, "
        f"{summary['with_synthesis']} with a synthesis"
    )
    print(
        f"  {summary['authored_words']} words you wrote, across "
        f"{summary['manuscripts']} manuscript(s)"
    )
    if summary["gap_kinds"]:
        print(f"  question kinds: {', '.join(summary['gap_kinds'])}")

    if args.export is None:
        return 0

    written = corpus.export_sft(
        args.export,
        merged_only=args.merged_only,
        include_synthesis=args.include_synthesis,
    )
    print(f"\nwrote {written} sample(s) to {args.export}")
    if not written:
        # An empty export file is indistinguishable from a training set that
        # simply has not accumulated yet, and only one of those is a mistake.
        print(
            "That is an empty file. Nothing matched the filters you passed — "
            "check --merged-only against the merged count above.",
            file=sys.stderr,
        )
        return 1
    return 0


def _turns(args: argparse.Namespace) -> TurnProvider | None:
    """Build the live turn provider, once, when --live is given.

    A failure to *construct* it is fatal: the author asked for the exchange and
    is entitled to know it will not happen before they start answering, rather
    than discovering it one refusal at a time.
    """
    if not getattr(args, "live", False):
        return None
    from .service import build_turn_provider

    provider: TurnProvider = build_turn_provider(_settings())
    return provider


def _cycle(
    session: Session,
    engine: SocraticEngine,
    gates: Gates,
    args: argparse.Namespace,
) -> int:
    """Ask, read an answer, gate, merge. Repeat until the author stops."""
    turns = _turns(args)
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
            # Leaving is not the same as passing, so the blank line ends the
            # session without recording an opinion about the question. `skip`
            # is how the author says this one was not worth answering.
            break

        _report(session.answer(text, gates, turns))
        session.journal.save(args.journal)
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
    if result.turn_error:
        print(
            f"  the dialectic exchange failed ({result.turn_error}); your answer "
            "was kept, the machine's objections were not produced"
        )
    for node_id in result.dropped:
        print(
            f"  dropped — the machine's node {node_id} could not be grounded; "
            "your answer merged without it"
        )
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


def _policy(session: Session) -> None:
    from .learn import LearnedPolicy

    rows = session.journal.summary()
    if not rows:
        print("Nothing asked yet, so nothing learned.")
        return
    print("What you did with the questions so far:\n")
    for line in LearnedPolicy(session.journal).explain():
        print(f"  {line}")
    print(
        "\nEngagement is measured from whether you answered, never from whether "
        "the gates accepted it: an answer they refused is still a question worth "
        "having asked."
    )


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
