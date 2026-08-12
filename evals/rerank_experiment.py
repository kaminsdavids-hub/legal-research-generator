"""Rerank retrieval on signals the grounding gate does not read.

§18 found headroom in retrieval and then found that the obvious way to close it —
ranking by the gate's own support scorer — reaches the ceiling by construction and
makes the gate decorative. The gap has to be closed on an *independent* signal, or
the check downstream stops being a check.

The gate reads exactly one thing: ``record.passages``, scored against the claim.
So the reranker here reads everything else and never that:

* ``headnotes`` — the curated one-line summaries of what each case was admitted
  to the corpus **for**, preserved when the corpus was rebuilt from opinion text.
  The gate has never read them.
* ``title``, ``court``, ``year``, ``type``, ``code``/``section`` — metadata the
  support check ignores entirely.

Strict disjointness is the point, and it is enforced rather than intended: for the
seven non-case records that have no headnotes, ``passages`` *is* what the gate
reads, so this scores their title and code/section only. Reaching into passages
for those would reintroduce the alignment the whole exercise exists to avoid.

The measurement is offline. Three numbers over the same claims:

* what retrieval actually chose,
* what ranking by this independent signal would choose,
* the ceiling — the best any retriever could do, by the gate's own measure.

A rerank score between the first and third is a real gain. A rerank score *at*
the ceiling would be the warning sign, not the success: it would mean the
independent signal is not independent after all.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.compare_support_scorers import build, load_pairs  # noqa: E402
from evals.retrieval_ceiling import index_corpus  # noqa: E402
from legal_research.config import get_settings  # noqa: E402
from modules.dialectic import textnorm  # noqa: E402

#: Fields the reranker may read. `passages` is deliberately absent: it is the
#: gate's input, and reading it here would align retrieval with the check.
RERANK_FIELDS = ("headnotes", "title", "court", "type", "code", "section")


def rerank_text(record: Any) -> str:
    """Everything about a record that the grounding gate never sees."""
    parts: list[str] = []
    for field_name in RERANK_FIELDS:
        value = getattr(record, field_name, None)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def rerank_score(claim: str, record: Any) -> float:
    """Content-word containment of the claim in a record's non-passage text.

    Deliberately crude and deliberately not the gate's measure. It answers a
    different question — is this record *about* the right thing — which is what a
    headnote encodes and what an opinion's full text is too diffuse to say.
    """
    claim_words = textnorm.content_words(claim, stemmed=True)
    if not claim_words:
        return 0.0
    record_words = textnorm.content_words(rerank_text(record), stemmed=True)
    if not record_words:
        return 0.0
    return len(claim_words & record_words) / len(claim_words)


@dataclass
class Outcome:
    claim: str
    chosen_cite: str
    chosen_supported: bool
    rerank_cite: str = ""
    rerank_supported: bool = False
    ceiling_cite: str = ""
    reachable: bool = False


@dataclass
class RerankReport:
    mode: str
    scorer: str
    threshold: float
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def actual(self) -> int:
        return sum(1 for o in self.outcomes if o.chosen_supported)

    @property
    def reranked(self) -> int:
        return sum(1 for o in self.outcomes if o.rerank_supported)

    @property
    def ceiling(self) -> int:
        return sum(1 for o in self.outcomes if o.reachable)

    @property
    def agrees_with_ceiling(self) -> int:
        """How often the independent signal picks exactly the gate's favourite.

        High agreement is a warning, not a result: it would mean the reranking
        signal has collapsed into the gate's own measure.
        """
        return sum(1 for o in self.outcomes if o.rerank_cite == o.ceiling_cite)


def best_by(claim: str, index: dict[str, Any], score: Any) -> tuple[str, float]:
    best_cite, best = "", 0.0
    for cite, record in index.items():
        value = score(claim, record)
        if value > best:
            best_cite, best = cite, value
    return best_cite, best


def measure(pairs: list[Any], index: dict[str, Any], mode: str) -> RerankReport:
    scorer = build(mode)
    report = RerankReport(
        mode=mode, scorer=type(scorer).__name__, threshold=float(scorer.threshold)
    )

    def supports(cite: str, claim: str) -> bool:
        record = index.get(cite.strip())
        passages = list(getattr(record, "passages", [])) if record else []
        if not passages:
            return False
        best = max(float(scorer.score(claim, p)) for p in passages)
        return bool(best >= float(scorer.threshold))

    def gate_score(claim: str, record: Any) -> float:
        passages = list(getattr(record, "passages", []))
        return max((float(scorer.score(claim, p)) for p in passages), default=0.0)

    for pair in pairs:
        rerank_cite, _ = best_by(pair.claim, index, rerank_score)
        ceiling_cite, ceiling_value = best_by(pair.claim, index, gate_score)
        report.outcomes.append(
            Outcome(
                claim=pair.claim,
                chosen_cite=pair.citation,
                chosen_supported=supports(pair.citation, pair.claim),
                rerank_cite=rerank_cite,
                rerank_supported=supports(rerank_cite, pair.claim) if rerank_cite else False,
                ceiling_cite=ceiling_cite,
                reachable=ceiling_value >= scorer.threshold,
            )
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rerank_experiment",
        description="Rerank retrieval on signals the grounding gate does not read.",
    )
    parser.add_argument("report", type=Path, help="a --live run's --out JSON")
    parser.add_argument("--modes", default="lexical,embedding")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.report)
    if not pairs:
        print("no authorities in that report; nothing to measure", file=sys.stderr)
        return 1

    settings = get_settings()
    index = index_corpus(settings.corpus_path)
    with_headnotes = sum(1 for r in index.values() if getattr(r, "headnotes", None))
    print(f"{len(pairs)} claim(s) from {args.report.name}")
    print(f"corpus: {settings.corpus_path} ({len(index)} records, {with_headnotes} with headnotes)\n")

    reports = []
    for mode in args.modes.split(","):
        report = measure(pairs, index, mode.strip())
        reports.append(report)
        n = len(report.outcomes)
        print(
            f"{report.mode:10s} retrieval {report.actual}/{n}  "
            f"reranked {report.reranked}/{n}  ceiling {report.ceiling}/{n}   "
            f"(rerank picked the gate's favourite {report.agrees_with_ceiling}/{n})"
        )

    if args.out and reports:
        args.out.write_text(
            json.dumps(
                [
                    {
                        "mode": r.mode,
                        "actual": r.actual,
                        "reranked": r.reranked,
                        "ceiling": r.ceiling,
                        "agrees_with_ceiling": r.agrees_with_ceiling,
                        "outcomes": [vars(o) for o in r.outcomes],
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
