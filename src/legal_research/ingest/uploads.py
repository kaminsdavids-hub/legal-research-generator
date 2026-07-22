"""Ingest the scholar's own uploads into the corpus.

Supported inputs: ``.txt`` / ``.md`` (read directly), ``.html`` / ``.htm``
(tags stripped), and ``.pdf`` (text extracted via the optional ``pypdf`` dep from
the ``ingest`` extra). Each file becomes one :class:`CorpusRecord` — by default a
``secondary`` source — with its text split into passages. Ids are stable across
runs (derived from the filename plus a short content hash), so re-ingesting an
unchanged file is idempotent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..citations.corpus import CorpusRecord
from ..models import SourceType
from .base import chunk_passages, make_id, normalize_ws, strip_html

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES


class UnsupportedUploadError(ValueError):
    """Raised when a file's type is not one we know how to read."""


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        return strip_html(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in PDF_SUFFIXES:
        return _extract_pdf_text(path)
    raise UnsupportedUploadError(
        f"unsupported upload type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise UnsupportedUploadError(
            "reading PDFs requires the 'ingest' extra: pip install -e '.[ingest]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


class UploadIngestor:
    def __init__(self, source_type: SourceType = SourceType.SECONDARY) -> None:
        self._source_type = source_type

    def record_from_file(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        source_type: SourceType | None = None,
    ) -> CorpusRecord | None:
        p = Path(path)
        text = normalize_ws(extract_text(p))
        if not text:
            return None
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return CorpusRecord(
            id=make_id("upload", p.stem, digest),
            type=source_type or self._source_type,
            title=title or _title_from_stem(p.stem),
            url=p.resolve().as_uri(),
            passages=chunk_passages(text),
        )

    def records_from_paths(
        self,
        paths: Iterable[str | Path],
        *,
        source_type: SourceType | None = None,
    ) -> list[CorpusRecord]:
        out: list[CorpusRecord] = []
        for path in paths:
            for resolved in _expand(Path(path)):
                if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                record = self.record_from_file(resolved, source_type=source_type)
                if record is not None and record.passages:
                    out.append(record)
        return out


def _expand(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file())
    return [path]


def _title_from_stem(stem: str) -> str:
    return normalize_ws(stem.replace("_", " ").replace("-", " ")).title()
