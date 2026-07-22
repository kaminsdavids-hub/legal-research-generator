"""The legal corpus: real case-law/statute/regulation records the system may cite.

In production this is populated from CourtListener / Caselaw Access Project, the
U.S. Code and CFR, plus the scholar's uploaded PDFs. Here it loads a small pinned
JSONL sample so the pipeline is fully runnable offline. The loader is the single
source of truth: nothing may be cited that is not a record in this corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from ..models import SourceType


class CorpusRecord(BaseModel):
    id: str
    type: SourceType
    title: str
    # Case fields
    reporter: str = ""
    volume: int | None = None
    page: int | None = None
    court: str = ""
    # Statute / regulation fields
    code: str = ""
    section: str = ""
    year: int | None = None
    url: str = ""
    passages: list[str] = Field(default_factory=list)

    def full_text(self) -> str:
        return " ".join([self.title, *self.passages])


class Corpus(BaseModel):
    records: list[CorpusRecord] = Field(default_factory=list)

    def get(self, record_id: str) -> CorpusRecord | None:
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CorpusRecord]:  # type: ignore[override]
        return iter(self.records)


def load_corpus(path: str | Path) -> Corpus:
    """Load a corpus from a JSONL file (one :class:`CorpusRecord` per line)."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"corpus file not found: {p}")
    records: list[CorpusRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(CorpusRecord.model_validate_json(line))
    return Corpus(records=records)


def corpus_from_records(records: list[CorpusRecord]) -> Corpus:
    """Build a corpus directly from records (used by tests and uploads)."""

    return Corpus(records=list(records))
