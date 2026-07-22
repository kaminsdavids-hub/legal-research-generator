"""Ingest case law from the Caselaw Access Project (CAP).

CAP's live API was retired in 2024; the data now ships as static bulk JSON on
``static.case.law``. This module maps a CAP case object (from a bulk file or the
legacy API shape) into a :class:`CorpusRecord`. Reading is defensive so both the
``casebody.data.opinions[].text`` (bulk) and flattened shapes are handled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..citations.corpus import CorpusRecord
from ..models import SourceType
from .base import chunk_passages, make_id, normalize_ws
from .courtlistener import parse_citation, slug_reporter


class CapIngestor:
    def official_citation(self, case: dict[str, Any]) -> str:
        cites = case.get("citations") or []
        official = [
            c.get("cite", "")
            for c in cites
            if isinstance(c, dict) and c.get("type") == "official" and c.get("cite")
        ]
        if official:
            return str(official[0])
        for c in cites:
            if isinstance(c, dict) and c.get("cite"):
                return str(c["cite"])
        return ""

    def _opinion_text(self, case: dict[str, Any]) -> str:
        casebody = case.get("casebody") or {}
        data = casebody.get("data", casebody)
        if isinstance(data, str):
            return normalize_ws(data)
        if isinstance(data, dict):
            opinions = data.get("opinions") or []
            texts = [
                normalize_ws(o.get("text", ""))
                for o in opinions
                if isinstance(o, dict) and o.get("text")
            ]
            if texts:
                return "\n\n".join(texts)
            head = data.get("head_matter")
            if isinstance(head, str):
                return normalize_ws(head)
        return ""

    def record_from_case(self, case: dict[str, Any]) -> CorpusRecord | None:
        cite = self.official_citation(case)
        parsed = parse_citation(cite)
        if parsed:
            record_id = make_id(slug_reporter(parsed[1]), str(parsed[0]), str(parsed[2]))
        elif case.get("id") is not None:
            record_id = f"cap-{case['id']}"
        else:
            return None

        text = self._opinion_text(case)
        passages = chunk_passages(text) if text else []
        court = case.get("court") or {}
        court_name = court.get("name", "") if isinstance(court, dict) else str(court)
        title = normalize_ws(case.get("name_abbreviation", "") or case.get("name", ""))

        return CorpusRecord(
            id=record_id,
            type=SourceType.CASE,
            title=title,
            reporter=(parsed[1] if parsed else ""),
            volume=(parsed[0] if parsed else None),
            page=(parsed[2] if parsed else None),
            court=normalize_ws(court_name),
            year=_year(case.get("decision_date")),
            url=str(case.get("frontend_url", "") or case.get("url", "")),
            passages=passages,
        )

    def from_cases(self, cases: list[dict[str, Any]]) -> list[CorpusRecord]:
        out: list[CorpusRecord] = []
        for case in cases:
            record = self.record_from_case(case)
            if record is not None and record.passages:
                out.append(record)
        return out

    def load_bulk(self, path: str | Path) -> list[CorpusRecord]:
        """Load a CAP bulk file: either a JSON array or JSONL (one case per line)."""

        raw = Path(path).read_text(encoding="utf-8").strip()
        if not raw:
            return []
        cases: list[dict[str, Any]]
        if raw[0] == "[":
            cases = json.loads(raw)
        else:
            cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return self.from_cases(cases)


def _year(date_str: Any) -> int | None:
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None
