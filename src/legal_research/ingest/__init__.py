"""Corpus ingestion: turn real legal sources into trusted :class:`CorpusRecord`s.

The retriever and verifier may only cite what is in the corpus, so this package is
how real authority enters the system: CourtListener and the Caselaw Access Project
for case law, U.S. Code (USLM) and CFR (eCFR/GPO) for statutes and regulations,
and the scholar's own uploads. All network access goes through the injectable
:class:`Fetcher`, keeping ingestion fully testable offline.
"""

from __future__ import annotations

from .base import (
    Fetcher,
    HttpxFetcher,
    chunk_passages,
    dedupe_records,
    records_to_corpus,
    write_jsonl,
)
from .cap import CapIngestor
from .courtlistener import CourtListenerIngestor
from .statutes import CfrIngestor, UsCodeIngestor
from .uploads import UploadIngestor

__all__ = [
    "Fetcher",
    "HttpxFetcher",
    "chunk_passages",
    "dedupe_records",
    "records_to_corpus",
    "write_jsonl",
    "CapIngestor",
    "CourtListenerIngestor",
    "CfrIngestor",
    "UsCodeIngestor",
    "UploadIngestor",
]
