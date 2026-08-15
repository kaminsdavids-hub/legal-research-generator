#!/usr/bin/env python3
"""Check that the judge discriminates before trusting it to score a full run.

A judge that returns 8 for everything produces a number that cannot rank
anything, and the plateau rule would then converge on noise. This feeds it
responses of deliberately graded quality and reports whether the scores order
correctly and spread far enough to be informative.

The fixtures are synthetic `DialecticTurn`s, so this costs one judge call each
and no debate generation.

    python evals/calibrate_judge.py --judge saul:7b-instruct-v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.harness import EvalSet, judge_response  # noqa: E402
from modules.dialectic.models import (  # noqa: E402
    CitationSlot,
    Crux,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)


def _turn(
    thesis: list[CitationSlot],
    antithesis: list[CitationSlot],
    synthesis: str,
    cruxes: list[Crux] | None = None,
) -> DialecticTurn:
    return DialecticTurn(
        question="Are trained model weights speech under the First Amendment?",
        thesis=Position(side="thesis", model="hermes3:8b", family="hermes", propositions=thesis),
        antithesis=Position(
            side="antithesis", model="llama3.1:8b", family="llama", propositions=antithesis
        ),
        synthesis=synthesis,
        cruxes=cruxes or [],
    )


def _slot(text: str, weight: Weight = Weight.SUPPORTING, cite: str = "") -> CitationSlot:
    return CitationSlot(
        proposition=text,
        weight=weight,
        normalized_cite=cite,
        status=SlotStatus.VERIFIED if cite else SlotStatus.PENDING,
    )


# Ordered worst to best. The judge should rank them in this order.
FIXTURES: list[tuple[str, str, DialecticTurn]] = [
    (
        "failed",
        "Both sides failed generation entirely. Should score at the floor.",
        _turn(
            [_slot("(generation failed or contained a citation string)")],
            [_slot("(generation failed or contained a citation string)")],
            "(synthesis contained a citation string and was rejected)",
        ),
    ),
    (
        "vacuous",
        "Restates the question on both sides. The rubric says this scores 1-2.",
        _turn(
            [_slot("Model weights might be speech under the First Amendment.")],
            [_slot("Model weights might not be speech under the First Amendment.")],
            "It depends on how the court views the question. Both sides have a point.",
        ),
    ),
    (
        "one_sided",
        "A real thesis, but the antithesis concedes instead of arguing.",
        _turn(
            [
                _slot(
                    "Trained parameters are the expressive product of human choices about "
                    "data selection and objectives, and so fall within protected expression.",
                    Weight.CONTROLLING,
                    "176 F.3d 1132",
                )
            ],
            [_slot("That argument seems largely correct, though some may disagree.")],
            "The thesis is persuasive and the antithesis does not seriously contest it.",
        ),
    ),
    (
        "strong",
        "Both sides argued from distinct doctrine, with an identified crux.",
        _turn(
            [
                _slot(
                    "Trained parameters are the expressive product of human choices about data "
                    "selection, architecture and objectives, and publishing them communicates "
                    "those choices to other researchers.",
                    Weight.CONTROLLING,
                    "176 F.3d 1132",
                ),
                _slot(
                    "The crypto-era holdings protected code because it was the medium in which "
                    "scientists communicate, and that rationale does not depend on human "
                    "readability.",
                    Weight.PERSUASIVE,
                    "209 F.3d 481",
                ),
            ],
            [
                _slot(
                    "Protection in those cases turned on a human reader being able to understand "
                    "the artifact; a parameter tensor conveys nothing to any reader and functions "
                    "only as a machine component.",
                    Weight.CONTROLLING,
                    "391 U.S. 367",
                ),
                _slot(
                    "Treating capability thresholds as content-neutral conduct regulation avoids "
                    "the expressive question entirely and is the narrower ground.",
                    Weight.PERSUASIVE,
                    "576 U.S. 155",
                ),
            ],
            "The decisive crux is whether the expressive status of code survives the loss of a "
            "human reader. If protection rested on scientific communication rather than on "
            "legibility, weights are covered; if legibility was doing the work, they are not. No "
            "controlling authority resolves it, and the circuits are not aligned.",
            cruxes=[
                Crux(
                    thesis_prop=_slot("Weights are expressive.", Weight.CONTROLLING),
                    antithesis_prop=_slot("Weights convey nothing to a reader.", Weight.CONTROLLING),
                    negates=True,
                    partition="open",
                    outcome_bearing=True,
                    nli_source="model",
                )
            ],
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="saul:7b-instruct-v1")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-spread", type=float, default=3.0,
                        help="required gap between the worst and best fixture")
    args = parser.parse_args()

    from legal_research.llm.openai_compat import OpenAICompatLLM
    from modules.dialectic.service import _ClientAdapter

    judge = _ClientAdapter(
        args.judge,
        OpenAICompatLLM(
            name=args.judge, base_url=args.base_url, model=args.judge,
            api_key="ollama", timeout=args.timeout,
        ),
    )
    question = next(q for q in EvalSet.load(ROOT / "evals/openweights_first_amendment.json").questions
                    if q.id == "A1")

    print(f"judge: {args.judge}\n" + "=" * 70)
    means: list[tuple[str, float]] = []
    for name, why, turn in FIXTURES:
        verdict = judge_response(judge, question, turn)
        means.append((name, verdict.mean))
        print(f"{name:<10} mean={verdict.mean:5.2f}  parsed={verdict.parsed}  ({why})")
        if verdict.parsed:
            print(f"           {verdict.scores}")
        else:
            print(f"           !! {verdict.rationale[:120]}")

    print("=" * 70)
    ordered = [m for _, m in means]
    monotonic = all(a <= b for a, b in zip(ordered, ordered[1:], strict=False))
    spread = max(ordered) - min(ordered)
    print(f"ranks worst-to-best correctly : {monotonic}  {[f'{m:.1f}' for m in ordered]}")
    print(f"spread (best - worst)         : {spread:.2f}  (need >= {args.min_spread})")

    ok = monotonic and spread >= args.min_spread
    if not ok:
        print(
            "\nJUDGE NOT USABLE AS-IS. It cannot separate weak work from strong, so a full "
            "run would produce numbers that rank nothing and a plateau rule that converges "
            "on noise. Try a different judge model or a harsher rubric before spending the "
            "GPU time."
        )
    else:
        print("\nJudge discriminates. Safe to use for a scored run.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
