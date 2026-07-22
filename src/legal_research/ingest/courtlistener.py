"""Ingest case law from CourtListener (Free Law Project) REST API v4.

Search opinions, then materialize each into a :class:`CorpusRecord` of type
``case``. Full opinion text is fetched from the ``/opinions/{id}/`` resource when
available; otherwise the search snippet is used. An API token is optional but
lifts rate limits — pass it in or set ``LRG_COURTLISTENER_TOKEN``.

The exact JSON shape of the v4 API is read defensively (many keys are optional),
so a schema tweak degrades to fewer fields rather than a crash.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from ..citations.corpus import CorpusRecord
from ..models import SourceType
from .base import Fetcher, HttpxFetcher, chunk_passages, make_id, normalize_ws, strip_html

API_ROOT = "https://www.courtlistener.com/api/rest/v4"
_CITE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s+(\d+)\s*$")


def parse_citation(cite: str) -> tuple[int, str, int] | None:
    """Parse a reporter citation like ``425 U.S. 185`` into (volume, reporter, page)."""

    m = _CITE_RE.match(cite or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip(), int(m.group(3))


def citation_to_id(cite: str) -> str | None:
    """Turn ``425 U.S. 185`` into a stable id like ``us-425-185``."""

    parsed = parse_citation(cite)
    if not parsed:
        return None
    volume, reporter, page = parsed
    return make_id(slug_reporter(reporter), str(volume), str(page))


def slug_reporter(reporter: str) -> str:
    """Compact a reporter abbreviation into an id prefix (``U.S.`` -> ``us``)."""

    letters = re.sub(r"[^a-z0-9]", "", reporter.lower())
    return letters or "cite"


class CourtListenerIngestor:
    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        api_token: str | None = None,
    ) -> None:
        self._fetcher = fetcher or HttpxFetcher()
        self._api_token = api_token

    def _headers(self) -> dict[str, str]:
        if self._api_token:
            return {"Authorization": f"Token {self._api_token}"}
        return {}

    def search(
        self,
        query: str,
        *,
        count: int = 20,
        court: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield up to ``count`` opinion search results for ``query``."""

        params: dict[str, str] = {"q": query, "type": "o", "order_by": "score desc"}
        if court:
            params["court"] = court

        url: str | None = f"{API_ROOT}/search/"
        yielded = 0
        current_params: dict[str, str] | None = params
        while url and yielded < count:
            data = self._fetcher.get_json(url, params=current_params, headers=self._headers())
            for result in data.get("results", []):
                yield result
                yielded += 1
                if yielded >= count:
                    return
            url = data.get("next")
            current_params = None  # ``next`` is a fully-formed URL

    def _opinion_text(self, opinion_id: int | str) -> str:
        url = f"{API_ROOT}/opinions/{opinion_id}/"
        try:
            data = self._fetcher.get_json(url, headers=self._headers())
        except Exception:  # noqa: BLE001 - text is best-effort; snippet is the fallback
            return ""
        for key in ("plain_text", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_ws(value)
        html = data.get("html") or data.get("html_lawbox") or data.get("html_with_citations")
        if isinstance(html, str) and html.strip():
            return strip_html(html)
        return ""

    def record_from_result(
        self,
        result: dict[str, Any],
        *,
        fetch_text: bool = True,
    ) -> CorpusRecord | None:
        """Map one v4 search result (an opinion cluster) to a :class:`CorpusRecord`."""

        citations = [c for c in result.get("citation", []) if isinstance(c, str)]
        best_cite = next((c for c in citations if parse_citation(c)), citations[0] if citations else "")
        parsed = parse_citation(best_cite)

        record_id = citation_to_id(best_cite)
        if record_id is None:
            cluster_id = result.get("cluster_id") or result.get("id")
            if cluster_id is None:
                return None
            record_id = f"cl-{cluster_id}"

        text = ""
        opinions = result.get("opinions") or []
        if fetch_text and opinions:
            op_id = opinions[0].get("id") if isinstance(opinions[0], dict) else None
            if op_id is not None:
                text = self._opinion_text(op_id)
        if not text:
            snippets = [
                normalize_ws(o.get("snippet", ""))
                for o in opinions
                if isinstance(o, dict) and o.get("snippet")
            ]
            text = " ".join(snippets)

        passages = chunk_passages(text) if text else []
        year = _year(result.get("dateFiled"))
        url = result.get("absolute_url", "")
        if url.startswith("/"):
            url = f"https://www.courtlistener.com{url}"

        return CorpusRecord(
            id=record_id,
            type=SourceType.CASE,
            title=normalize_ws(result.get("caseName", "") or result.get("caseNameFull", "")),
            reporter=(parsed[1] if parsed else ""),
            volume=(parsed[0] if parsed else None),
            page=(parsed[2] if parsed else None),
            court=normalize_ws(result.get("court", "")),
            year=year,
            url=url,
            passages=passages,
        )

    def ingest(
        self,
        query: str,
        *,
        count: int = 20,
        court: str | None = None,
        fetch_text: bool = True,
    ) -> list[CorpusRecord]:
        """Search and materialize records, dropping any that cannot be mapped."""

        out: list[CorpusRecord] = []
        for result in self.search(query, count=count, court=court):
            record = self.record_from_result(result, fetch_text=fetch_text)
            if record is not None and record.passages:
                out.append(record)
        return out


def _year(date_str: Any) -> int | None:
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None
