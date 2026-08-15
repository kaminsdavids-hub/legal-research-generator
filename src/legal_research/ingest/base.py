"""Shared ingestion utilities: HTTP fetching, passage chunking, JSONL I/O.

Every network call goes through the :class:`Fetcher` protocol so ingestion is
fully testable offline: tests inject a fake fetcher that returns fixtures, and
nothing in CI touches the network. Ingestors emit :class:`CorpusRecord` objects —
the single source of truth the retriever and verifier are allowed to trust.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..citations.corpus import Corpus, CorpusRecord, load_corpus

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_PASSAGE_CHARS = 1200
_MIN_PASSAGE_CHARS = 40

_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.;:!?])\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@runtime_checkable
class Fetcher(Protocol):
    """Minimal HTTP surface used by network ingestors; trivially fakeable in tests."""

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any: ...

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str: ...


class HttpxFetcher:
    """Default :class:`Fetcher` backed by httpx. Used outside tests (real network)."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def get_json(  # pragma: no cover - real network
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        import httpx

        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def get_text(  # pragma: no cover - real network
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        import httpx

        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return str(resp.text)


def normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip ends."""

    return _WS_RE.sub(" ", text or "").strip()


def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace (cheap, dependency-free)."""

    return normalize_ws(_TAG_RE.sub(" ", text or ""))


def slugify(text: str) -> str:
    """Lowercase, hyphenated slug safe for use in a record id."""

    return _SLUG_RE.sub("-", (text or "").lower()).strip("-")


def make_id(prefix: str, *parts: str) -> str:
    """Build a stable, readable record id from a prefix and slugified parts."""

    tail = "-".join(slugify(p) for p in parts if p)
    return f"{prefix}-{tail}" if tail else prefix


def chunk_passages(
    text: str,
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    *,
    min_chars: int = _MIN_PASSAGE_CHARS,
) -> list[str]:
    """Split text into passages no longer than ``max_chars``, on sentence bounds.

    Whitespace is normalized first. Paragraphs are honored (split on blank lines),
    then greedily packed by sentence so a passage never straddles a paragraph and
    rarely straddles a sentence. Fragments shorter than ``min_chars`` are merged
    forward so we never emit trivial one-word "passages".
    """

    paragraphs = [normalize_ws(p) for p in re.split(r"\n\s*\n", text or "")]
    passages: list[str] = []
    for para in paragraphs:
        if not para:
            continue
        if len(para) <= max_chars:
            passages.append(para)
            continue
        current = ""
        for sentence in _SENTENCE_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            for piece in _hard_wrap(sentence, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    passages.append(current)
                    current = piece
                else:
                    current = f"{current} {piece}".strip()
        if current:
            passages.append(current)

    # Merge short trailing fragments forward so nothing tiny survives on its own.
    merged: list[str] = []
    for p in passages:
        if merged and len(p) < min_chars:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)
    return [m for m in merged if m]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Word-pack ``text`` into pieces no longer than ``max_chars`` (last resort)."""

    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def dedupe_records(records: Iterable[CorpusRecord]) -> list[CorpusRecord]:
    """Keep the first record per id, preserving order."""

    seen: set[str] = set()
    out: list[CorpusRecord] = []
    for record in records:
        if record.id in seen:
            continue
        seen.add(record.id)
        out.append(record)
    return out


def write_jsonl(
    records: Iterable[CorpusRecord],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Write records as JSONL (one :class:`CorpusRecord` per line). Returns count.

    When ``append`` is true and the file exists, records whose id already appears
    in the file are skipped so re-running an ingest is idempotent.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if append and p.exists():
        existing_ids = {r.id for r in load_corpus(p).records}

    written = 0
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as fh:
        for record in dedupe_records(records):
            if record.id in existing_ids:
                continue
            fh.write(record.model_dump_json() + "\n")
            existing_ids.add(record.id)
            written += 1
    return written


def records_to_corpus(records: Iterable[CorpusRecord]) -> Corpus:
    """Build an in-memory :class:`Corpus` from records (deduped by id)."""

    return Corpus(records=dedupe_records(records))
