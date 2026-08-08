#!/usr/bin/env python3
"""Run the correlation guard against live local models (REMEDIATION 5, W9).

Compares crux yield for two arms over one question set, through the production
path:

* **distinct-family** — two different base families (the production pairing).
* **same-family** — two genuinely different checkpoints of the *same* family.

The same-family arm is the point of the experiment: the previous version built
it by relabelling one client with the other's name, so both roles ran the same
client over byte-identical text and the "collapse" it reported measured only
that identical text does not contradict itself.

This script needs a running Ollama (or any OpenAI-compatible endpoint) with the
named checkpoints pulled. It is NOT part of the test suite: the suite runs
offline. Usage::

    python scripts/correlation_guard_experiment.py \
        --same-family llama3.1:8b llama3.2:3b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.dialectic.engine import DialecticChat  # noqa: E402
from modules.dialectic.service import _ClientAdapter  # noqa: E402

# A real question set: contested legal questions with genuine two-sided answers.
QUESTIONS = [
    "Does a warrantless search of a cell phone incident to arrest violate the Fourth Amendment?",
    "Is scienter required for liability under the securities antifraud provisions?",
    "Does the exclusionary rule apply to evidence obtained in good-faith reliance on a defective warrant?",
    "Can a state compel an out-of-state seller to collect sales tax without physical presence?",
    "Is a mandatory arbitration clause in an employment contract enforceable against a class claim?",
    "Does qualified immunity shield an officer who used force during a fleeing-suspect stop?",
    "Is legislative history a legitimate aid to interpreting an unambiguous statute?",
    "Does the dormant Commerce Clause bar a state law that burdens interstate commerce incidentally?",
]


def _client(model: str, base_url: str, timeout: float) -> _ClientAdapter:
    from legal_research.llm.openai_compat import OpenAICompatLLM

    return _ClientAdapter(
        model,
        OpenAICompatLLM(
            name=model, base_url=base_url, model=model, api_key="ollama", timeout=timeout
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--distinct",
        nargs=2,
        metavar=("THESIS", "ANTITHESIS"),
        default=["hermes3:8b", "llama3.1:8b"],
        help="two checkpoints from DIFFERENT families",
    )
    parser.add_argument(
        "--same-family",
        nargs=2,
        metavar=("THESIS", "ANTITHESIS"),
        required=True,
        help="two genuinely different checkpoints of the SAME family",
    )
    parser.add_argument("--synthesis", default="gemma3:4b")
    parser.add_argument("--nli", default="nemotron-3-nano:4b")
    parser.add_argument("--questions", type=int, default=len(QUESTIONS))
    args = parser.parse_args()

    questions = QUESTIONS[: args.questions]

    def make(model: str) -> _ClientAdapter:
        return _client(model, args.base_url, args.timeout)

    chat = DialecticChat(
        thesis_client=make(args.distinct[0]),
        antithesis_client=make(args.distinct[1]),
        synthesis_client=make(args.synthesis),
        nli_client=make(args.nli),
    )

    print(f"distinct-family arm : {args.distinct[0]} vs {args.distinct[1]}")
    print(f"same-family arm     : {args.same_family[0]} vs {args.same_family[1]}")
    print(f"synthesis={args.synthesis}  nli={args.nli}  questions={len(questions)}")
    print("-" * 78)

    started = time.time()
    report = chat.correlation_guard_report(
        questions,
        distinct_pair=(make(args.distinct[0]), make(args.distinct[1])),
        same_family_pair=(make(args.same_family[0]), make(args.same_family[1])),
    )
    elapsed = time.time() - started

    print(report.summary())
    print(f"elapsed: {elapsed / 60:.1f} min")
    print("-" * 78)
    if not report.separates:
        print(
            "The crux counts did NOT separate. Per the work order this result "
            "stands as recorded; the family guard's justification needs "
            "reconsidering rather than the experiment adjusting."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
