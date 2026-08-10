#!/usr/bin/env python3
"""Report run-to-run variance across repeated eval runs.

The plateau rule declares convergence when the overall mean moves less than a
threshold across consecutive rounds. That threshold is only meaningful next to
the noise floor: if repeated runs of an unchanged system already move by more
than the threshold, the rule fires on noise.

Reads every result file for an eval set and reports per-question and overall
spread, plus how many questions were never scored.

    python evals/variance_report.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "evals/results"))
    parser.add_argument("--eval-set", default="openweights_first_amendment")
    parser.add_argument("--plateau-delta", type=float, default=0.2)
    args = parser.parse_args()

    paths = sorted(Path(args.results).glob(f"{args.eval_set}-*.json"))
    runs = [json.loads(p.read_text()) for p in paths]
    # Only full runs are comparable; partial smoke runs would distort everything.
    full = [(p, r) for p, r in zip(paths, runs, strict=True) if r.get("total_count", len(r["questions"])) >= 30]
    if len(full) < 2:
        print(f"need at least 2 full runs to estimate variance; found {len(full)}")
        for p, r in zip(paths, runs, strict=True):
            print(f"  {p.name}  questions={len(r['questions'])}  mean={r['overall_mean']}")
        return 1

    print(f"{len(full)} full run(s):")
    means = []
    for p, r in full:
        scored = r.get("scored_count", len(r["questions"]))
        total = r.get("total_count", len(r["questions"]))
        means.append(r["overall_mean"])
        print(f"  {p.name}  mean={r['overall_mean']:.3f}  scored={scored}/{total}  "
              f"gate_failures={len(r['gate_failures'])}  unscored={len(r.get('unscored_judge_failed', []))}")

    print("\n--- overall ---")
    print(f"means      : {[f'{m:.3f}' for m in means]}")
    print(f"range      : {max(means) - min(means):.3f}")
    if len(means) > 1:
        print(f"stdev      : {st.stdev(means):.3f}")
    deltas = [abs(b - a) for a, b in zip(means, means[1:], strict=False)]
    print(f"consecutive deltas: {[f'{d:.3f}' for d in deltas]}")
    worst = max(deltas) if deltas else 0.0
    print(f"largest consecutive delta: {worst:.3f}   plateau threshold: {args.plateau_delta}")
    margin = args.plateau_delta - worst
    if worst >= args.plateau_delta:
        print("  -> NOISE EXCEEDS THE THRESHOLD. An unchanged system already moves more "
              "than the plateau rule tolerates, so the rule would fire on noise. Raise "
              "the threshold above the noise floor or average several runs per round.")
    elif margin < 0.25 * args.plateau_delta:
        print(f"  -> MARGIN IS THIN ({margin:.3f}). The observed maximum is close enough to "
              "the threshold that one more unlucky run would cross it. Treat this as "
              "'not yet established' rather than safe.")
    else:
        print("  -> noise floor is below the threshold; a change larger than it is signal.")

    per_q: dict[str, list[float | None]] = defaultdict(list)
    never_scored: dict[str, int] = defaultdict(int)
    for _, r in full:
        for q in r["questions"]:
            per_q[q["id"]].append(q["mean"])
            if q["mean"] is None:
                never_scored[q["id"]] += 1

    print("\n--- per question (scored runs only) ---")
    unstable = []
    for qid, vals in sorted(per_q.items()):
        got = [v for v in vals if v is not None]
        if len(got) < 2:
            print(f"  {qid:<4} insufficient data  raw={vals}")
            continue
        spread = max(got) - min(got)
        if spread > 0:
            unstable.append((spread, qid, got))
        flag = "" if spread == 0 else f"   <-- varies by {spread:.2f}"
        print(f"  {qid:<4} mean={st.fmean(got):.2f}  spread={spread:.2f}  {[f'{v:.1f}' for v in got]}{flag}")

    print("\n--- summary ---")
    stable = sum(1 for _, vals in per_q.items() if len({v for v in vals if v is not None}) <= 1)
    print(f"questions identical across runs: {stable}/{len(per_q)}")
    if unstable:
        print("most variable:")
        for spread, qid, got in sorted(unstable, reverse=True)[:5]:
            print(f"  {qid}  spread={spread:.2f}  {[f'{v:.1f}' for v in got]}")
    if never_scored:
        print(f"judge failed at least once on: {dict(never_scored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
