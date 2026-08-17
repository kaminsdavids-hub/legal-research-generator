#!/usr/bin/env python3
"""Apply the plateau rule to rounds rather than to single runs.

A citation gate that flips between otherwise identical runs moves a
32-question mean by about 0.24 — more than the originally specified 0.2
threshold — and four different questions have flipped across four sets. Fixing
them one at a time looked like convergence and was not: each fix was real and
none exhausted the category (REMEDIATION §11.11b).

Averaging removes the assumption instead of tolerating it. A flip moves a round
of N runs by 0.24/N, and ordinary drift falls as 1/sqrt(N), so at N=3 the round
figure is stable to roughly 0.13 — inside 0.2 with margin.

    python evals/round_report.py --per-round 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from evals.harness import (  # noqa: E402
    PLATEAU_ROUNDS,
    ROUND_PLATEAU_DELTA,
    has_plateaued,
    round_means,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "evals/results"))
    parser.add_argument("--eval-set", default="openweights_first_amendment")
    parser.add_argument("--per-round", type=int, default=3)
    parser.add_argument("--delta", type=float, default=ROUND_PLATEAU_DELTA)
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="use only the N most recent full runs (0 = all). Runs from "
        "different code versions are not comparable.",
    )
    args = parser.parse_args()

    paths = sorted(Path(args.results).glob(f"{args.eval_set}-*.json"))
    full = []
    for p in paths:
        d = json.loads(p.read_text())
        if d.get("total_count", len(d["questions"])) >= 30:
            full.append((p, d))
    if args.last > 0:
        full = full[-args.last :]
    if not full:
        print("no full runs found")
        return 1

    print(f"{len(full)} full run(s), grouped into rounds of {args.per_round}:")
    runs: list[tuple[str, float]] = []
    for p, d in full:
        rid = d.get("round_id", "")
        runs.append((rid, d["overall_mean"]))
        flips = len(d["gate_failures"])
        print(f"  {p.name}  mean={d['overall_mean']:.3f}  round={rid or '(untagged)'}  "
              f"gate_failures={flips}")

    means = round_means(runs, per_round=args.per_round)
    print(f"\nround means: {[f'{m:.3f}' for m in means]}")
    if len(means) < 2:
        print(
            f"\nOnly {len(means)} round(s). The round-level noise floor needs at least two "
            f"rounds on IDENTICAL code — {2 * args.per_round} runs — and comparing rounds "
            "that ran different code measures the change, not the noise."
        )
        return 1

    deltas = [abs(b - a) for a, b in zip(means, means[1:], strict=False)]
    print(f"round deltas: {[f'{d:.3f}' for d in deltas]}")
    worst = max(deltas)
    print(f"largest round delta: {worst:.3f}   threshold: {args.delta}")
    if worst >= args.delta:
        print("  -> exceeds the threshold; the rule would fire on noise at this round size")
    else:
        print("  -> inside the threshold; a round-to-round change larger than it is signal")

    if has_plateaued(means, delta=args.delta, rounds=PLATEAU_ROUNDS):
        print(f"\nPLATEAU: round mean moved < {args.delta} across {PLATEAU_ROUNDS} rounds")
    else:
        print(f"\nno plateau across the last {PLATEAU_ROUNDS} rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
