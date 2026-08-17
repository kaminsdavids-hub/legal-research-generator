"""``linearize order`` — turn a saved argument graph into a reading order.

    linearize order --graph paper.graph.json --prev-order last.json --sections 6

Thin, like ``maieutic``: argv in, report out. Everything that decides anything
lives in the modules, so it can be tested without going through stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from modules.maieutic.graph import ArgumentGraph
from modules.maieutic.novelty import LexicalEmbedder

from .brkga import BRKGAConfig, evolve
from .problem import PrecedenceViolation, Weights, build_problem
from .report import ablation, build_report, compare_baselines, reading_plan
from .sa import SAConfig, anneal
from .segment import SegmentConfig, section_titles, segment, sweep_k

#: Above this many nodes the annealer's per-proposal cost stops paying and the
#: random-key GA takes over. The brief's threshold, kept.
BRKGA_THRESHOLD = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linearize", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    order = sub.add_parser("order", help="compute a reading order and section breaks")
    order.add_argument("--graph", type=Path, required=True, help="saved ArgumentGraph JSON")
    order.add_argument("--prev-order", type=Path, help="the author's last accepted order")
    order.add_argument("--pins", type=Path, help='JSON {"node_id": position}')
    order.add_argument("--sections", type=int, default=0, help="fix k (0 sweeps for the elbow)")
    order.add_argument("--out", type=Path, help="write the order + report as JSON")
    order.add_argument("--seed", type=int, default=7)
    order.add_argument("--iterations", type=int, default=20_000)
    order.add_argument("--solver", choices=("auto", "sa", "brkga"), default="auto")
    order.add_argument("--reference-weight", type=float, default=1.0)
    order.add_argument("--cohesion-weight", type=float, default=1.0)
    order.add_argument("--stability-weight", type=float, default=3.0)
    order.add_argument("--min-words", type=int, default=150)
    order.add_argument("--max-words", type=int, default=4_000)
    order.add_argument("--ablation", action="store_true", help="report each weight's contribution")
    return parser


def _load_graph(path: Path) -> ArgumentGraph:
    return ArgumentGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_order(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("order", [])
    return [str(node) for node in data]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "order":  # pragma: no cover - argparse enforces this
        return 2

    graph = _load_graph(args.graph)
    if not graph.nodes:
        print("the graph has no nodes; nothing to order", file=sys.stderr)
        return 1

    node_ids = sorted(graph.nodes)
    embedder = LexicalEmbedder()
    vectors = embedder.embed([graph.nodes[n].text for n in node_ids])
    embeddings = dict(zip(node_ids, vectors, strict=False))

    pins = {}
    if args.pins and args.pins.exists():
        pins = {str(k): int(v) for k, v in json.loads(args.pins.read_text()).items()}

    weights = Weights(
        reference=args.reference_weight,
        cohesion=args.cohesion_weight,
        stability=args.stability_weight,
    )

    try:
        problem = build_problem(
            graph,
            embeddings=embeddings,
            previous=_load_order(args.prev_order),
            pins=pins,
            weights=weights,
        )
    except PrecedenceViolation as exc:
        # Loud, and not repaired: see problem.PrecedenceViolation.
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    solver = args.solver
    if solver == "auto":
        solver = "brkga" if len(problem.node_ids) > BRKGA_THRESHOLD else "sa"

    if solver == "sa":
        result: Any = anneal(problem, config=SAConfig(iterations=args.iterations, seed=args.seed))
    else:
        result = evolve(problem, config=BRKGAConfig(seed=args.seed))

    order = result.order
    config = SegmentConfig(min_words=args.min_words, max_words=args.max_words)
    segmentation = None
    try:
        if args.sections:
            segmentation = segment(problem, order, args.sections, config)
        else:
            upper = max(2, min(8, len(order) // 3 or 2))
            segmentation, _curve = sweep_k(problem, order, range(2, upper + 1), config)
    except ValueError as exc:
        print(f"segmentation skipped: {exc}", file=sys.stderr)

    report = build_report(
        problem,
        order,
        segmentation=segmentation,
        titles=section_titles(order, segmentation) if segmentation else None,
        solver=solver,
        seed=args.seed,
    )
    print(report.render())

    baselines = compare_baselines(problem, order)
    if len(baselines) > 1:
        print("\nbaselines (same metrics):")
        for name, row in baselines.items():
            print(
                f"  {name:18s} total={row['total']:9.2f}  forward={row['forward_references']:3.0f}  "
                f"mean_distance={row['mean_distance']:.1f}"
            )

    if args.ablation:
        print("\nablation (this order, each weight zeroed):")
        for name, row in ablation(problem, order).items():
            print(f"  {name:20s} total={row['total']:9.2f}")

    if args.out:
        payload = report.as_dict()
        payload["baselines"] = baselines
        if segmentation is not None:
            # The renderer's half of the artifact. Without a node -> section map
            # the file records what the optimizer decided but not enough to act
            # on it: `render.ReadingPlan(order, section_of)` is the handoff, and
            # a downstream step would otherwise have to re-run the search to
            # recover a plan that was already computed here.
            _, section_of = reading_plan(
                order, segmentation, section_titles(order, segmentation)
            )
            payload["section_of"] = section_of
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
