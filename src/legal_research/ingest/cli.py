"""``lrg-ingest`` — build the retrieval corpus from real legal sources.

Every subcommand writes/append :class:`CorpusRecord` JSONL to ``--out`` (default
``data/corpus/corpus.jsonl``). Point ``LRG_CORPUS_PATH`` at that file to have the
retriever and verifier trust it.

Examples::

    lrg-ingest uploads ./briefs --type secondary
    lrg-ingest courtlistener --query "scienter 10b-5" --count 25
    lrg-ingest cap --file cap_bulk.jsonl
    lrg-ingest usc --file usc-title15.uslm.xml
    lrg-ingest cfr --file cfr-title17.xml --title 17
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from ..citations.corpus import CorpusRecord
from ..models import SourceType
from .base import HttpxFetcher, write_jsonl
from .cap import CapIngestor
from .courtlistener import CourtListenerIngestor
from .statutes import CfrIngestor, UsCodeIngestor
from .uploads import UploadIngestor

DEFAULT_OUT = "data/corpus/corpus.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lrg-ingest", description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output JSONL (default {DEFAULT_OUT})")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite --out instead of appending (append is the default, and idempotent)",
    )
    sub = parser.add_subparsers(dest="source", required=True)

    up = sub.add_parser("uploads", help="ingest local PDF/TXT/MD/HTML files or directories")
    up.add_argument("paths", nargs="+", help="files or directories to ingest")
    up.add_argument(
        "--type",
        default="secondary",
        choices=[t.value for t in SourceType],
        help="source type to tag uploads with (default: secondary)",
    )

    cl = sub.add_parser("courtlistener", help="ingest opinions from CourtListener v4")
    cl.add_argument("--query", required=True, help="full-text search query")
    cl.add_argument("--count", type=int, default=20, help="max opinions to ingest")
    cl.add_argument("--court", default=None, help="optional court id filter, e.g. scotus")
    cl.add_argument("--token", default=None, help="API token (or set LRG_COURTLISTENER_TOKEN)")

    cap = sub.add_parser("cap", help="ingest a Caselaw Access Project bulk file")
    cap.add_argument("--file", required=True, help="CAP bulk JSON array or JSONL file")

    usc = sub.add_parser("usc", help="ingest U.S. Code USLM XML")
    usc.add_argument("--file", required=True, help="USLM XML file (uscode.house.gov)")

    cfr = sub.add_parser("cfr", help="ingest CFR eCFR/GPO XML")
    cfr.add_argument("--file", required=True, help="eCFR/GPO title XML file")
    cfr.add_argument("--title", default=None, help="CFR title number (else inferred)")

    return parser


def _collect(args: argparse.Namespace) -> list[CorpusRecord]:
    if args.source == "uploads":
        ingestor = UploadIngestor(source_type=SourceType(args.type))
        return ingestor.records_from_paths(args.paths)
    if args.source == "courtlistener":
        token = args.token or os.environ.get("LRG_COURTLISTENER_TOKEN")
        cl = CourtListenerIngestor(HttpxFetcher(), api_token=token)
        return cl.ingest(args.query, count=args.count, court=args.court)
    if args.source == "cap":
        return CapIngestor().load_bulk(args.file)
    if args.source == "usc":
        return UsCodeIngestor().ingest_file(args.file)
    if args.source == "cfr":
        return CfrIngestor(HttpxFetcher()).ingest_file(args.file, title=args.title)
    return []


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _collect(args)
    if not records:
        print("No records produced; nothing written.", file=sys.stderr)
        return 1
    written = write_jsonl(records, args.out, append=not args.overwrite)
    action = "wrote" if args.overwrite else "appended"
    print(f"{action} {written} record(s) to {args.out} (from {len(records)} parsed)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
