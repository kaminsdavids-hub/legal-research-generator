"""Run the maieutic loop over scripted sessions and report what it did.

``build_parser`` is separate from ``main`` so the CLI is reachable from a test.
The lesson is §11.11c: a runner that no test invokes can be broken for every
invocation while the suite stays green, and this repository has already shipped
that once.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.loop_harness import LoopReport, load_scripts, run_all  # noqa: E402
from modules.maieutic.loop import Gates  # noqa: E402

DEFAULT_SESSIONS = Path(__file__).resolve().parent / "loop_sessions.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_loop_eval",
        description="Measure what the maieutic loop does for an author who answers.",
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "run the real dialectic engine. Slow: roughly three minutes per "
            "answer across four local models. Without it the run measures the "
            "loop's structure only, and no authority is ever proposed."
        ),
    )
    parser.add_argument(
        "--cite-for",
        choices=["claim", "premise"],
        default="claim",
        help=(
            "what the exchange is asked to cite for. 'premise' asks the models "
            "for the settled propositions the claim rests on rather than for the "
            "claim itself; see REMEDIATION 14 for why that is worth measuring."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument(
        "--only", default="", help="run one session by id, for a quick check"
    )
    return parser


def _describe(value: float | None, fmt: str = "{:.2f}") -> str:
    """Missing data reads as such, never as a number."""
    return "n/a (nothing measured)" if value is None else fmt.format(value)


def report_lines(report: LoopReport) -> list[str]:
    summary = report.summary()
    lines = [
        f"mode: {summary['mode']}",
        f"sessions: {summary['sessions']} ({summary['sessions_errored']} errored)",
        f"exchanges: {summary['exchanges']}",
        f"answer survival: {_describe(summary['answer_survival'], '{:.0%}')}",
        f"objections per merged exchange: {_describe(summary['objections_per_exchange'])}",
        f"authority survival: {_describe(summary['authority_survival'], '{:.0%}')}",
        f"sessions that ran dry: {summary['ran_dry']}",
        f"wall time: {summary['seconds']}s",
        "",
    ]
    for session in report.sessions:
        lines.append(f"  {session.script_id} [{session.source}]")
        if session.error:
            lines.append(f"    ERROR {session.error}")
            continue
        lines.append(
            f"    {len(session.exchanges)} exchange(s), {session.nodes} node(s), "
            f"{session.open_problems} open problem(s)"
            + (
                f", ran dry with {session.answers_unused} answer(s) unused"
                if session.ran_dry
                else ""
            )
        )
        for exchange in session.exchanges:
            state = "merged" if exchange.merged else "REFUSED"
            lines.append(
                f"    #{exchange.index} {exchange.gap_kind}: {state}"
                f" (+{exchange.author_nodes} author, +{exchange.machine_nodes} machine,"
                f" {exchange.dropped} dropped, {exchange.seconds:.1f}s)"
            )
            for refusal in exchange.refusals:
                lines.append(f"        refused: {refusal}")
            if exchange.turn_error:
                lines.append(f"        exchange failed: {exchange.turn_error}")
            if exchange.error:
                lines.append(f"        ERROR: {exchange.error}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scripts = load_scripts(args.sessions)
    if args.only:
        scripts = [s for s in scripts if s.id == args.only]
        if not scripts:
            print(f"no session with id {args.only!r}", file=sys.stderr)
            return 1

    gates: Gates
    turns = None
    if args.live:
        from legal_research.config import get_settings
        from modules.maieutic.service import build_turn_provider

        settings = get_settings()
        gates = Gates.live(settings)
        turns = build_turn_provider(settings, args.cite_for)
    else:
        gates = Gates.offline()

    mode = "live" if args.live else "offline"
    if args.live and args.cite_for != "claim":
        mode = f"{mode}/cite-for-{args.cite_for}"
    report = run_all(scripts, gates, turns, mode=mode)
    print("\n".join(report_lines(report)))

    if args.out:
        payload: dict[str, Any] = {
            "summary": report.summary(),
            "sessions": [asdict(s) for s in report.sessions],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
