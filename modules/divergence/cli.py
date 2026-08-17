"""``divergence search`` — find where the frozen readings disagree.

    divergence search --schema data/feature_schema.yaml --rules data/rules --epoch 4

Episodic by design: this is the expensive module, and the trigger for running it
is that a rule predicate was added or changed. It never calls Portfolio; the two
exchange events through the log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from modules.casework.corpus import load_cases
from modules.casework.report import divergence_report
from modules.casework.schema import load_schema
from modules.coupling.events import EpochManager, EventLog, regions_from_priors

from .genotype import FactVector, FeasibilityMask
from .mapelites import DivergenceConfig, run_map_elites, run_nsga2
from .objectives import cached_scorer
from .precedent import match_precedents
from .rules import load_rules
from .seeding import seeds_from_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="divergence", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="fill an archive of hard cases")
    search.add_argument("--schema", type=Path, required=True)
    search.add_argument("--rules", type=Path, required=True)
    search.add_argument("--cases", type=Path, default=Path("data/cases.yaml"))
    search.add_argument("--mask", type=Path, help="python file defining build(mask)")
    search.add_argument("--epoch", type=int, required=True)
    search.add_argument("--events", type=Path, default=Path(".casework/events.jsonl"))
    search.add_argument("--out", type=Path, help="write the archive here")
    search.add_argument("--descriptors", nargs="*", default=[])
    search.add_argument("--iterations", type=int, default=600)
    search.add_argument("--seed", type=int, default=7)
    search.add_argument("--mode", choices=("mapelites", "nsga2"), default="mapelites")
    search.add_argument("--realism-weight", type=float, default=0.0)
    search.add_argument(
        "--seed-from-cases",
        action="store_true",
        help="start the search at the coded cases, expanded over their blank axes",
    )
    search.add_argument("--seed-budget", type=int, default=400)
    search.add_argument("--top", type=int, default=8)
    return parser


def _load_mask(schema, path: Path | None) -> FeasibilityMask:
    """The mask is author-authored Python, like the rules.

    A YAML predicate language would be a worse version of Python with none of
    its tooling, and the constraints are the place where realism is supposed to
    be inspectable rather than implicit.
    """

    mask = FeasibilityMask(schema)
    if path is None:
        return mask
    spec = importlib.util.spec_from_file_location("_divergence_mask", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load mask: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_divergence_mask"] = module
    spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if build is None:
        raise AttributeError(f"{path.name} defines no build(mask) function")
    build(mask)
    return mask


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "search":  # pragma: no cover
        return 2

    schema = load_schema(args.schema)
    rules = load_rules(args.rules)
    corpus = load_cases(args.cases) if args.cases.exists() else None
    mask = _load_mask(schema, args.mask)

    complaints = corpus.validate_against(schema) if corpus else []
    for complaint in complaints:
        print(f"corpus: {complaint}", file=sys.stderr)

    log = EventLog.load(args.events)
    epochs = EpochManager(log=log, epoch=args.epoch)
    snapshot = epochs.snapshot(args.epoch)
    priors = [
        FactVector({**{a.name: schema.axis(a.name).values[0] for a in schema.axes}, **region})
        for region in regions_from_priors(snapshot["priors"])
    ]
    priors = [p for p in priors if mask.feasible(p)]

    if args.seed_from_cases:
        if corpus is None:
            print("seeding: --seed-from-cases needs a corpus; none loaded", file=sys.stderr)
        else:
            plan = seeds_from_cases(schema, corpus, mask, budget=args.seed_budget)
            print(f"seeding: {plan.summary()}")
            for case_id in plan.infeasible:
                # A coded real decision the mask calls impossible is a bug in
                # one of the two, and it must not pass in silence.
                print(f"seeding: {case_id} has no feasible completion", file=sys.stderr)
            if plan.truncated:
                print(f"seeding: budget reached, not seeded: {', '.join(plan.truncated)}")
            # Appended after the coupling priors, and `run_map_elites` pops from
            # the end, so the corpus is explored first and last epoch's thin
            # coverage still gets its turn.
            priors = [*priors, *plan.seeds]

    if snapshot["priors"]:
        print(
            f"coupling: {len(snapshot['priors'])} ThinCoverage signal(s) from earlier "
            f"epochs, {len(priors)} usable as seeds. Signals raised during epoch "
            f"{args.epoch} apply to the next one."
        )

    # Every flag that changes the archive, so the stamp is a command and not a
    # summary. Descriptors, the realism weight and seeding all move the result;
    # a recipe omitting them reproduces a different run.
    recipe = (
        f"divergence search --schema {args.schema} --rules {args.rules} "
        f"--cases {args.cases} --epoch {args.epoch} --seed {args.seed} "
        f"--iterations {args.iterations} --mode {args.mode} "
        f"--realism-weight {args.realism_weight}"
    )
    if args.descriptors:
        recipe += f" --descriptors {' '.join(args.descriptors)}"
    if args.mask:
        recipe += f" --mask {args.mask}"
    if args.seed_from_cases:
        recipe += f" --seed-from-cases --seed-budget {args.seed_budget}"

    config = DivergenceConfig(
        descriptors=tuple(args.descriptors) or (),
        iterations=args.iterations,
        seed=args.seed,
        realism_weight=args.realism_weight,
        recipe=recipe,
    )
    runner = run_map_elites if args.mode == "mapelites" else run_nsga2
    kwargs = {"priors": priors} if args.mode == "mapelites" else {}
    archive = runner(schema, rules, mask, corpus=corpus, config=config, epoch=args.epoch, **kwargs)

    collision = None
    if corpus:
        scorer = cached_scorer(rules, mask, schema, corpus)
        findings = [
            (FactVector(dict(e.genome)), scorer(FactVector(dict(e.genome))),
             rules.verdicts(FactVector(dict(e.genome))))
            for e in archive.elites
        ]
        collision = match_precedents(schema, corpus, findings)

    report = divergence_report(
        archive,
        collision,
        top=args.top,
        corpus_size=len(corpus.cases) if corpus else 0,
        verified_codings=len(corpus.verified()) if corpus else 0,
    )
    print()
    print(report.render())

    if args.out:
        archive.save(args.out)
        report.save(Path(str(args.out) + ".report.json"))
        if collision is not None:
            Path(str(args.out) + ".collisions.json").write_text(
                json.dumps(
                    {
                        "epoch": args.epoch,
                        "precedents": [p.__dict__ for p in collision.precedents],
                        "unverified_candidates": collision.unverified_candidates,
                        "too_undetermined": collision.too_undetermined,
                        "off_schema": collision.off_schema,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        print(f"\nwrote {args.out}")

    if collision is not None:
        epochs.record_precedents(
            [
                {"case_id": p.case_id, "citation": p.citation, "features": p.facts}
                for p in collision.precedents
            ]
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
