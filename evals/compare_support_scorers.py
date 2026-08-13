"""D7: does the support scorer explain the loop's authority survival rate?

The first live loop run grounded 1 authority in 19 (OBSERVABLES D25). Three
explanations were live and that run could not separate them: the models cite
badly, the corpus is small, or `LexicalSupportScorer` is too crude a proxy for
"this passage supports this claim". This re-scores the *same* (claim, citation)
pairs under every available scorer, so the scorer's contribution is isolated
without re-running a single model.

**The silent fallback is the hazard here.** `build_support_scorer` returns
`LexicalSupportScorer` when the extra is missing, so an experiment that just sets
`LRG_SUPPORT_SCORER=embedding` and reports "no difference" may have measured
lexical three times. That failure has a history in this repository: a log wrapper
that dropped a method (§11.6) and a CLI no test invoked (§11.11c) both reported
success while doing nothing. So every scorer is constructed, its concrete class
is checked against what was asked for, and a fallback is a hard error rather than
a footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_research.citations.corpus import load_corpus  # noqa: E402
from legal_research.config import get_settings  # noqa: E402
from modules.dialectic.service import _CorpusCiteRetriever  # noqa: E402

#: What each mode must actually construct. A mismatch means the extra is absent
#: and the run would silently measure lexical overlap under another name.
EXPECTED: dict[str, str] = {
    "lexical": "LexicalSupportScorer",
    "embedding": "EmbeddingSupportScorer",
    "nli": "NliSupportScorer",
}


class FellBackToLexical(RuntimeError):
    """Raised when a scorer could not be built and lexical was substituted."""


@dataclass(frozen=True)
class Pair:
    citation: str
    claim: str
    grounded: bool


def load_pairs(path: Path) -> list[Pair]:
    """Every authority the loop proposed, from a `--live` report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for session in data["sessions"]:
        for exchange in session["exchanges"]:
            for authority in exchange.get("authorities", []):
                pairs.append(
                    Pair(
                        citation=authority["citation"],
                        claim=authority["claim"],
                        grounded=bool(authority["grounded"]),
                    )
                )
    return pairs


def build(mode: str) -> Any:
    """Construct a scorer and refuse to proceed if it is not the one asked for."""
    import os

    from legal_research.citations.support import build_support_scorer

    previous = os.environ.get("LRG_SUPPORT_SCORER")
    os.environ["LRG_SUPPORT_SCORER"] = mode
    try:
        settings = type(get_settings())()  # re-read env rather than reuse a cache
        scorer = build_support_scorer(settings)
    finally:
        if previous is None:
            os.environ.pop("LRG_SUPPORT_SCORER", None)
        else:
            os.environ["LRG_SUPPORT_SCORER"] = previous

    actual = type(scorer).__name__
    if actual != EXPECTED[mode]:
        raise FellBackToLexical(
            f"asked for {mode} ({EXPECTED[mode]}) and got {actual}; the extra is "
            "missing, so this run would report lexical results under another name. "
            "Install the gpu extra rather than reading the numbers."
        )
    return scorer


def passages_for(citation: str, index: dict[str, Any]) -> list[str]:
    record = index.get(citation.strip())
    return list(getattr(record, "passages", [])) if record else []


def score_all(pairs: list[Pair], mode: str, index: dict[str, Any]) -> dict[str, Any]:
    scorer = build(mode)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        passages = passages_for(pair.citation, index)
        if not passages:
            # No text to check against is missing data, not a zero: the scorer
            # never got the chance to disagree with anything.
            rows.append({"citation": pair.citation, "best": None, "supports": None})
            continue
        best = max(float(scorer.score(pair.claim, p)) for p in passages)
        rows.append(
            {
                "citation": pair.citation,
                "best": round(best, 3),
                "supports": best >= scorer.threshold,
            }
        )
    judged = [r for r in rows if r["supports"] is not None]
    return {
        "mode": mode,
        "scorer": type(scorer).__name__,
        "threshold": scorer.threshold,
        "judged": len(judged),
        "unjudgeable": len(rows) - len(judged),
        "supported": sum(1 for r in judged if r["supports"]),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_support_scorers",
        description="Re-score a live loop run's authorities under every scorer.",
    )
    parser.add_argument("report", type=Path, help="a --live run's --out JSON")
    parser.add_argument("--modes", default="lexical,embedding,nli")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.report)
    if not pairs:
        print("no authorities in that report; nothing to compare", file=sys.stderr)
        return 1

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    index = {}
    for record in getattr(corpus, "records", []):
        cite = _CorpusCiteRetriever._format(record)
        if cite:
            index[cite] = record

    print(f"{len(pairs)} authority claim(s) from {args.report.name}")
    print(f"corpus: {settings.corpus_path} ({len(index)} citable records)")
    print(f"grounded in the original run: {sum(1 for p in pairs if p.grounded)}\n")

    results = []
    for mode in args.modes.split(","):
        try:
            result = score_all(pairs, mode.strip(), index)
        except FellBackToLexical as exc:
            print(f"{mode}: SKIPPED — {exc}\n")
            continue
        results.append(result)
        print(
            f"{result['mode']:10s} {result['scorer']:24s} thr={result['threshold']:<5} "
            f"supported {result['supported']}/{result['judged']}"
            + (f" ({result['unjudgeable']} unjudgeable)" if result["unjudgeable"] else "")
        )

    if args.out and results:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
