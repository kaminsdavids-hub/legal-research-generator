#!/usr/bin/env python3
"""Diagnose where crux yield is lost (REMEDIATION 9.8: base rate 3/8, 2/8).

For each question this dumps:

* every thesis and antithesis proposition, with its weight,
* the NLI relation for every cross-product pair, with the label's source.

That distinguishes the three candidate causes:

1. **too few propositions** — a side returns 0 or 1 claim,
2. **talking past each other** — both sides produce claims, but they address
   different predicates, so nothing contradicts (expected: the debaters are
   generated independently and never see each other's output),
3. **NLI too strict** — pairs that plainly contradict come back neutral.

Needs a running Ollama. Not part of the test suite.
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

QUESTIONS = [
    "Does a warrantless search of a cell phone incident to arrest violate the Fourth Amendment?",
    "Is scienter required for liability under the securities antifraud provisions?",
    "Does the exclusionary rule apply to evidence obtained in good-faith reliance on a defective warrant?",
    "Can a state compel an out-of-state seller to collect sales tax without physical presence?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--thesis", default="hermes3:8b")
    parser.add_argument("--antithesis", default="llama3.1:8b")
    parser.add_argument("--synthesis", default="gemma3:4b")
    parser.add_argument("--nli", default="nemotron-3-nano:4b")
    parser.add_argument("--questions", type=int, default=len(QUESTIONS))
    args = parser.parse_args()

    from legal_research.llm.openai_compat import OpenAICompatLLM

    def make(model: str) -> _ClientAdapter:
        return _ClientAdapter(
            model,
            OpenAICompatLLM(
                name=model,
                base_url=args.base_url,
                model=model,
                api_key="ollama",
                timeout=args.timeout,
            ),
        )

    chat = DialecticChat(
        thesis_client=make(args.thesis),
        antithesis_client=make(args.antithesis),
        synthesis_client=make(args.synthesis),
        nli_client=make(args.nli),
    )

    totals = {
        "pairs": 0, "contradiction": 0, "entailment": 0, "neutral": 0,
        "antithesis_props": 0, "mirrors": 0,
    }
    prop_counts: list[tuple[int, int]] = []
    started = time.time()

    for i, question in enumerate(QUESTIONS[: args.questions], start=1):
        print("=" * 78, flush=True)
        print(f"Q{i}. {question}", flush=True)
        turn = chat.chat(question)

        print(f"\n  THESIS ({turn.thesis.model}) — {len(turn.thesis.propositions)} propositions", flush=True)
        for s in turn.thesis.propositions:
            print(f"    [{s.weight}] {s.proposition}", flush=True)
        print(f"\n  ANTITHESIS ({turn.antithesis.model}) — {len(turn.antithesis.propositions)} propositions", flush=True)
        for s in turn.antithesis.propositions:
            print(f"    [{s.weight}] {s.proposition}", flush=True)

        prop_counts.append((len(turn.thesis.propositions), len(turn.antithesis.propositions)))

        print("\n  CROSS-PRODUCT NLI:", flush=True)
        nli = chat.crux_extractor.nli
        for t in turn.thesis.propositions:
            for a in turn.antithesis.propositions:
                label, source = nli.relate(a.proposition, t.proposition)
                totals["pairs"] += 1
                totals[label] += 1
                mark = "  <-- CRUX" if label == "contradiction" else ""
                print(f"    {label:<13} ({source:<9}) T:{t.proposition[:44]!r} A:{a.proposition[:44]!r}{mark}", flush=True)

        # How many antithesis propositions still merely negate the thesis after
        # the independence guard has spent its retry budget.
        mirrors = chat.independence.find_mirrors(
            [s.proposition for s in turn.antithesis.propositions],
            [s.proposition for s in turn.thesis.propositions],
        )
        totals["antithesis_props"] += len(turn.antithesis.propositions)
        totals["mirrors"] += len(mirrors)
        if mirrors:
            print("\n  RESIDUAL MIRRORS:", flush=True)
            for m in mirrors:
                print(f"    j={m.jaccard:.2f} c={m.containment:.2f} {m.candidate[:70]}", flush=True)

        print(
            f"\n  => cruxes: {len(turn.cruxes)}  | mirrors: {len(mirrors)}"
            f"/{len(turn.antithesis.propositions)}  | regenerated: {turn.regenerated}"
            f"  | crux_note: {turn.crux_note[:60]}",
            flush=True,
        )

    print("=" * 78, flush=True)
    print(f"proposition counts (thesis, antithesis): {prop_counts}", flush=True)
    print(f"pairs compared: {totals['pairs']}", flush=True)
    for label in ("contradiction", "entailment", "neutral"):
        pct = 100.0 * totals[label] / totals["pairs"] if totals["pairs"] else 0.0
        print(f"  {label:<13} {totals[label]:>3}  ({pct:.0f}%)", flush=True)
    mirrored = totals["mirrors"]
    props = totals["antithesis_props"]
    pct = 100.0 * mirrored / props if props else 0.0
    print(f"residual mirrors: {mirrored}/{props} antithesis propositions ({pct:.0f}%)", flush=True)
    print(f"elapsed: {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
