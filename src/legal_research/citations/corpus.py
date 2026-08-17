"""The legal corpus: real case-law/statute/regulation records the system may cite.

In production this is populated from CourtListener / Caselaw Access Project, the
U.S. Code and CFR, plus the scholar's uploaded PDFs. Here it loads a small pinned
JSONL sample so the pipeline is fully runnable offline. The loader is the single
source of truth: nothing may be cited that is not a record in this corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ..models import SourceType


class AuthorityStatus(str, Enum):
    """Operative status of an authority, independent of whether it is good law.

    A rescinded rule is not currently operative but may still be precedential —
    it shows what an agency believed it could do, and its reasoning survives its
    repeal. Collapsing "rescinded" into "removed" would lose that, and citing it
    as operative law is a different error from citing it at all.
    """

    IN_FORCE = "in_force"
    RESCINDED = "rescinded"
    SUPERSEDED = "superseded"
    PROPOSED = "proposed"
    #: A court withdrew its own opinion — distinct from superseded, where a later
    #: decision replaced it, and from rescinded, which is what agencies do to
    #: rules. A withdrawn opinion was un-issued: it is not merely non-operative,
    #: it is not precedent at all, and its reasoning carries no weight beyond
    #: what a reader chooses to find persuasive.
    #:
    #: Added for *Bernstein v. U.S. Dep't of Justice*, 176 F.3d 1132 (9th Cir.
    #: 1999), which this corpus recorded as ``in_force`` while the Ninth
    #: Circuit's own order says: "The three-judge panel opinion ... is
    #: withdrawn." 192 F.3d 1308 (9th Cir. 1999). Collapsing that into
    #: ``superseded`` would tell a reader a later opinion replaced it; none did.
    WITHDRAWN = "withdrawn"


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
    #: Curated one-line summaries of what this record was admitted to the corpus
    #: for. Distinct from `passages`, which carry quotable source text: the
    #: support check reads passages and never these, so they remain available as
    #: a retrieval signal independent of the grounding gate (REMEDIATION §19).
    headnotes: list[str] = Field(default_factory=list)
    #: Whether this authority is currently operative. Defaults to in_force so
    #: existing corpora keep their meaning.
    status: AuthorityStatus = AuthorityStatus.IN_FORCE
    #: Prose explaining a non-in_force status, e.g. what rescinded it and when.
    status_note: str = ""
    #: True when a hand-authored record still needs human verification. The
    #: corpus is the ground truth a citation-integrity gate is measured against,
    #: so a record nobody has checked must not silently become that truth.
    unverified: bool = False

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
