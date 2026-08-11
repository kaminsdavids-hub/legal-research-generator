#!/usr/bin/env python3
"""Pre-resolve every citation the corpus can propose, in as few calls as possible.

An eval run makes one lookup per position, so a full run costs ~64 calls against
a 125/day quota — and a run that exhausts it mid-way marks real authorities
NOT_FOUND, which the gate reports exactly like a fabricated cite. Three runs on
the Spark were lost that way.

The endpoint accepts many citations in one text block, so the entire demand
surface fits in two or three calls. Warming the cache first means every
subsequent run is served from disk and spends nothing, and verification becomes
constant across runs — so a variance estimate measures generation and judging
rather than API weather.

    python evals/warm_cite_cache.py --corpus data/corpus/openweights.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.dialectic.service import _CorpusCiteRetriever  # noqa: E402
from modules.dialectic.verification import (  # noqa: E402
    CourtListenerClient,
    PersistentCiteCache,
    RateBudget,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus/openweights.jsonl")
    parser.add_argument("--cite-cache", default=str(ROOT / "evals/.cache/courtlistener.json"))
    parser.add_argument("--batch", type=int, default=9, help="citations per request")
    parser.add_argument("--sleep", type=float, default=13.0, help="seconds between calls (5/min)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("LRG_COURTLISTENER_TOKEN", "").strip()

    from legal_research.citations.corpus import load_corpus

    corpus = load_corpus(args.corpus)

    # Only case citations need the API; statutes and regulations are verified by
    # the corpus itself and would never resolve here.
    wanted: set[str] = set()
    for record in corpus.records:
        cite = _CorpusCiteRetriever._format(record)
        if cite and record.volume is not None and record.page is not None and record.reporter:
            wanted.add(cite)

    cache = PersistentCiteCache(args.cite_cache)
    missing = sorted(c for c in wanted if cache.get(c) is None)
    print(f"corpus case citations : {len(wanted)}")
    print(f"already cached        : {len(wanted) - len(missing)}")
    print(f"to resolve            : {len(missing)}")
    if not missing:
        print("cache is complete; runs will spend no quota")
        return 0
    batches = [missing[i : i + args.batch] for i in range(0, len(missing), args.batch)]
    print(f"calls required        : {len(batches)}")
    for m in missing:
        print(f"  {m}")
    if args.dry_run:
        return 0
    if not token:
        print("FATAL: LRG_COURTLISTENER_TOKEN is not set", file=sys.stderr)
        return 2

    client = CourtListenerClient(token=token, budget=RateBudget(), cite_cache=cache)
    for n, batch in enumerate(batches, start=1):
        block = "\n".join(batch)
        data = client.lookup(block, cites=None)  # cites=None: force a live call
        if isinstance(data, dict) and data.get("error"):
            print(f"  call {n}/{len(batches)}: FAILED ({data['error']})", file=sys.stderr)
            print("  stopping; rerun after the quota resets", file=sys.stderr)
            return 1
        results = data if isinstance(data, list) else data.get("results", [])
        print(f"  call {n}/{len(batches)}: {len(results)} result(s)")
        if n < len(batches):
            time.sleep(args.sleep)

    still = sorted(c for c in wanted if cache.get(c) is None)
    print(f"\ncache now holds {len(cache)} key(s)")
    if still:
        print(f"still unresolved ({len(still)}): {still}")
        print("those will need live calls during a run")
        return 1
    print("every corpus citation is cached; runs will spend no quota")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
