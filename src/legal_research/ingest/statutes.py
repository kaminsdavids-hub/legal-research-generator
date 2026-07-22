"""Ingest statutes and regulations.

* :class:`UsCodeIngestor` parses U.S. Code USLM XML (from uscode.house.gov) into
  ``statute`` records — one per ``<section>``.
* :class:`CfrIngestor` parses eCFR / GPO CFR XML into ``regulation`` records — one
  per ``<DIV8 TYPE="SECTION">`` — and can fetch a title's XML from the eCFR API.

Both operate on XML *text*, so they are fully testable offline with a fixture and
work identically whether the XML came from a local file or the network. XML
namespaces are handled by matching on the element's local name.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from ..citations.corpus import CorpusRecord
from ..models import SourceType
from .base import Fetcher, HttpxFetcher, chunk_passages, make_id, normalize_ws

ECFR_ROOT = "https://www.ecfr.gov/api/versioner/v1"


def _local(tag: str) -> str:
    """Return an element tag's local name, dropping any ``{namespace}`` prefix."""

    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter(elem: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in elem.iter() if _local(e.tag) == name]


def _text_of(elem: ET.Element) -> str:
    return normalize_ws("".join(elem.itertext()))


class UsCodeIngestor:
    def records_from_uslm(self, xml_text: str) -> list[CorpusRecord]:
        root = ET.fromstring(xml_text)
        records: list[CorpusRecord] = []
        for section in _iter(root, "section"):
            record = self._record_from_section(section)
            if record is not None and record.passages:
                records.append(record)
        return records

    def _record_from_section(self, section: ET.Element) -> CorpusRecord | None:
        identifier = section.attrib.get("identifier", "")
        title_num, section_num = _parse_uslm_identifier(identifier)
        if section_num is None:
            for child in section:
                if _local(child.tag) == "num":
                    section_num = child.attrib.get("value") or normalize_ws(child.text or "")
                    break
        if not section_num:
            return None

        heading = ""
        passages_src: list[str] = []
        for child in section:
            local = _local(child.tag)
            if local == "heading" and not heading:
                heading = _text_of(child)
            elif local in {"content", "chapeau", "subsection", "paragraph", "subparagraph"}:
                text = _text_of(child)
                if text:
                    passages_src.append(text)
        if not passages_src:
            passages_src = [_text_of(section)]

        code = f"{title_num} U.S.C." if title_num else "U.S.C."
        record_id = (
            make_id("usc", str(title_num), section_num) if title_num else make_id("usc", section_num)
        )
        passages: list[str] = []
        for block in passages_src:
            passages.extend(chunk_passages(block))

        return CorpusRecord(
            id=record_id,
            type=SourceType.STATUTE,
            title=heading or f"{code} § {section_num}",
            code=code,
            section=section_num,
            url=f"https://www.govinfo.gov/link/uscode/{title_num}/{section_num}" if title_num else "",
            passages=passages,
        )

    def ingest_file(self, path: str | Path) -> list[CorpusRecord]:
        return self.records_from_uslm(Path(path).read_text(encoding="utf-8"))


class CfrIngestor:
    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher or HttpxFetcher()

    def records_from_gpo_xml(self, xml_text: str, *, title: str | None = None) -> list[CorpusRecord]:
        root = ET.fromstring(xml_text)
        resolved_title = title or _find_title_number(root)
        records: list[CorpusRecord] = []
        for div in root.iter():
            if _local(div.tag) != "DIV8":
                continue
            if div.attrib.get("TYPE", "").upper() != "SECTION":
                continue
            record = self._record_from_div(div, resolved_title)
            if record is not None and record.passages:
                records.append(record)
        return records

    def _record_from_div(self, div: ET.Element, title: str | None) -> CorpusRecord | None:
        section_num = div.attrib.get("N", "").strip()
        if not section_num:
            return None
        heading = ""
        passages: list[str] = []
        for child in div:
            local = _local(child.tag)
            if local in {"HEAD", "HEADING"} and not heading:
                heading = _text_of(child).lstrip("§ ").strip()
            elif local in {"P", "FP"}:
                text = _text_of(child)
                if text:
                    passages.extend(chunk_passages(text))

        code = f"{title} C.F.R." if title else "C.F.R."
        record_id = make_id("cfr", str(title) if title else "", section_num)
        url = (
            f"https://www.ecfr.gov/current/title-{title}/section-{section_num}" if title else ""
        )
        return CorpusRecord(
            id=record_id,
            type=SourceType.REGULATION,
            title=heading or f"{code} § {section_num}",
            code=code,
            section=section_num,
            url=url,
            passages=passages,
        )

    def fetch_title(self, title: int | str, date: str) -> list[CorpusRecord]:  # pragma: no cover - network
        """Fetch a full CFR title's XML for ``date`` (YYYY-MM-DD) from the eCFR API."""

        url = f"{ECFR_ROOT}/full/{date}/title-{title}.xml"
        xml_text = self._fetcher.get_text(url)
        return self.records_from_gpo_xml(xml_text, title=str(title))

    def ingest_file(self, path: str | Path, *, title: str | None = None) -> list[CorpusRecord]:
        return self.records_from_gpo_xml(Path(path).read_text(encoding="utf-8"), title=title)


def _parse_uslm_identifier(identifier: str) -> tuple[str | None, str | None]:
    """``/us/usc/t15/s78j`` -> (``15``, ``78j``); tolerant of trailing subsections."""

    if not identifier:
        return None, None
    title_num: str | None = None
    section_num: str | None = None
    for part in identifier.split("/"):
        if part.startswith("t") and part[1:].isalnum() and title_num is None:
            title_num = part[1:]
        elif part.startswith("s") and len(part) > 1 and section_num is None:
            section_num = part[1:]
    return title_num, section_num


def _find_title_number(root: ET.Element) -> str | None:
    for div in root.iter():
        if _local(div.tag) == "DIV1" and div.attrib.get("TYPE", "").upper() == "TITLE":
            n = div.attrib.get("N", "").strip()
            if n:
                return n
    return None
