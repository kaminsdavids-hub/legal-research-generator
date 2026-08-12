"""Could any retriever have grounded these claims?

§17 left the retrieval stage as the last unvaried suspect: it chooses *which*
corpus record is attached to a proposition, so a wrong choice looks exactly like
a model citing badly. Comparing retrieval modes would answer a narrower question
than the one that matters, and would need a live run per mode.

This measures the **ceiling** instead. For every authority claim, it scores that
claim against *every* record in the corpus and asks whether any record at all
would clear the support threshold. That is what a perfect retriever could
achieve, and it separates the two explanations cleanly:

* ceiling ≈ what retrieval actually got → retrieval is not the problem, and
  varying it cannot help. The claims are not supportable by this corpus.
* ceiling ≫ what retrieval actually got → retrieval picked the wrong record from
  a corpus that contained a right one, and is worth working on.

No models are called: the propositions come from a completed run, and only the
corpus and the scorer are exercised. The upper bound is the honest thing to
establish before optimising a component — an optimisation that cannot exceed the
ceiling is not worth measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.compare_support_scorers import Pair, build, load_pairs  # noqa: E402
from legal_research.citations.corpus import load_corpus  # noqa: E402
from legal_research.config import get_settings  # noqa: E402
from modules.dialectic.service import _CorpusCiteRetriever  # noqa: E402


@dataclass
class ClaimCeiling:
    claim: str
    chosen_cite: str
    #: What the live run decided, using whatever scorer it was configured with.
    chosen_grounded: bool
    #: What the record retrieval chose scores under *this* scorer. Comparing the
    #: live verdict against a ceiling computed with a different scorer would mix
    #: two variables and blame retrieval for a scorer disagreement.
    chosen_score: float = 0.0
    chosen_supported: bool = False
    best_cite: str = ""
    best_title: str = ""
    best_score: float = 0.0
    reachable: bool = False
    #: True when a record other than the one retrieval chose would have grounded.
    retrieval_missed: bool = False


@dataclass
class CeilingReport:
    mode: str
    scorer: str
    threshold: float
    claims: list[ClaimCeiling] = field(default_factory=list)

    @property
    def actual(self) -> int:
        """What retrieval's choice achieves under this scorer, not the live run's
        verdict: the live run used whatever scorer it was configured with."""
        return sum(1 for c in self.claims if c.chosen_supported)

    @property
    def live_grounded(self) -> int:
        return sum(1 for c in self.claims if c.chosen_grounded)

    @property
    def ceiling(self) -> int:
        return sum(1 for c in self.claims if c.reachable)

    @property
    def missed(self) -> list[ClaimCeiling]:
        return [c for c in self.claims if c.retrieval_missed]


def index_corpus(path: str) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for record in getattr(load_corpus(path), "records", []):
        cite = _CorpusCiteRetriever._format(record)
        if cite:
            index[cite] = record
    return index


def best_record(claim: str, index: dict[str, Any], scorer: Any) -> tuple[str, Any, float]:
    """The corpus record that best supports *claim*, whatever retrieval chose."""
    best_cite, best_rec, best = "", None, 0.0
    for cite, record in index.items():
        passages = list(getattr(record, "passages", []))
        if not passages:
            continue
        score = max(float(scorer.score(claim, p)) for p in passages)
        if score > best:
            best_cite, best_rec, best = cite, record, score
    return best_cite, best_rec, best


def measure(pairs: list[Pair], index: dict[str, Any], mode: str) -> CeilingReport:
    scorer = build(mode)
    report = CeilingReport(
        mode=mode, scorer=type(scorer).__name__, threshold=float(scorer.threshold)
    )
    for pair in pairs:
        cite, record, score = best_record(pair.claim, index, scorer)
        chosen = index.get(pair.citation.strip())
        chosen_passages = list(getattr(chosen, "passages", [])) if chosen else []
        chosen_score = (
            max(float(scorer.score(pair.claim, p)) for p in chosen_passages)
            if chosen_passages
            else 0.0
        )
        entry = ClaimCeiling(
            claim=pair.claim,
            chosen_cite=pair.citation,
            chosen_grounded=pair.grounded,
            chosen_score=round(chosen_score, 3),
            chosen_supported=chosen_score >= scorer.threshold,
            best_cite=cite,
            best_title=str(getattr(record, "title", "")) if record else "",
            best_score=round(score, 3),
            reachable=score >= scorer.threshold,
        )
        # Retrieval is only at fault when a better record existed *by the same
        # measure* that judged the one it chose.
        entry.retrieval_missed = entry.reachable and not entry.chosen_supported
        report.claims.append(entry)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieval_ceiling",
        description="Measure what a perfect retriever could have grounded.",
    )
    parser.add_argument("report", type=Path, help="a --live run's --out JSON")
    parser.add_argument("--modes", default="lexical,embedding")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--show-missed",
        action="store_true",
        help="name the record retrieval should have chosen, where one existed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.report)
    if not pairs:
        print("no authorities in that report; nothing to measure", file=sys.stderr)
        return 1

    settings = get_settings()
    index = index_corpus(settings.corpus_path)
    print(f"{len(pairs)} claim(s) from {args.report.name}")
    print(f"corpus: {settings.corpus_path} ({len(index)} citable records)\n")

    reports = []
    for mode in args.modes.split(","):
        report = measure(pairs, index, mode.strip())
        reports.append(report)
        print(
            f"{report.mode:10s} {report.scorer:24s} thr={report.threshold:<5} "
            f"retrieval's pick {report.actual}/{len(pairs)}, "
            f"ceiling {report.ceiling}/{len(pairs)} "
            f"({len(report.missed)} recoverable by better retrieval)"
        )
        if args.show_missed:
            for claim in report.missed:
                print(f"    claim:  {claim.claim[:110]}")
                print(f"    chose:  {claim.chosen_cite}")
                print(f"    better: {claim.best_cite} ({claim.best_title[:44]}) "
                      f"score {claim.best_score}")

    if args.out and reports:
        args.out.write_text(
            json.dumps(
                [
                    {
                        "mode": r.mode,
                        "scorer": r.scorer,
                        "threshold": r.threshold,
                        "actual": r.actual,
                        "ceiling": r.ceiling,
                        "claims": [vars(c) for c in r.claims],
                    }
                    for r in reports
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
