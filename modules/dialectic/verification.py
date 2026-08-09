"""CourtListener v4 citation-lookup client with token-bucket rate budget."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx

from .models import CitationSlot, Position, SlotStatus

DEFAULT_PER_MINUTE = 5
DEFAULT_PER_HOUR = 50
DEFAULT_PER_DAY = 125

COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

_LOG = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when the CourtListener rate budget is exhausted."""


class ContentCache:
    """Content-addressed LRU cache keyed on the query blob hash."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Any | None:
        key = self.key(text)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def set(self, text: str, value: Any) -> None:
        key = self.key(text)
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __contains__(self, text: str) -> bool:
        # Take the lock: this was the only unsynchronised read of `_data`, and an
        # OrderedDict membership test can race a concurrent `popitem` eviction.
        key = self.key(text)
        with self._lock:
            return key in self._data


class PersistentCiteCache:
    """Disk-backed cache of citation-lookup results, keyed per citation.

    ``ContentCache`` dies with the process, so every run re-verified the same
    authorities from scratch. At roughly 64 lookups per full eval run against a
    125/day quota, that caps the project at one run per day and made a repeat
    run for variance impossible.

    Keyed on the individual citation rather than the request's text block, so a
    later run whose propositions produce the same authorities in a different
    order, or a subset of them, still hits. Block-level keying would miss all of
    those.

    Only successful lookups are stored. Caching a 429 or a timeout would turn a
    transient outage into a permanent "this citation does not exist", which is
    indistinguishable from a fabricated cite.
    """

    def __init__(self, path: str | Path, *, max_age_days: float | None = None) -> None:
        self.path = Path(path)
        self.max_age_days = max_age_days
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("cite cache at %s unreadable (%s); starting empty", self.path, exc)
            return
        if isinstance(raw, dict):
            self._data = {k: v for k, v in raw.items() if isinstance(v, dict)}

    def _fresh(self, entry: dict[str, Any]) -> bool:
        if self.max_age_days is None:
            return True
        stamped = entry.get("cached_at")
        if not isinstance(stamped, int | float):
            return False
        return (time.time() - stamped) <= self.max_age_days * 86400

    def get(self, cite: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(cite)
            if entry is None or not self._fresh(entry):
                return None
            item = entry.get("item")
            return dict(item) if isinstance(item, dict) else None

    def get_all(self, cites: Sequence[str]) -> list[dict[str, Any]] | None:
        """Every result for *cites*, or ``None`` when any is missing.

        All-or-nothing on purpose: a partial hit still needs a network call, and
        serving half a lookup from cache would silently drop the rest.
        """
        found: list[dict[str, Any]] = []
        for cite in cites:
            item = self.get(cite)
            if item is None:
                return None
            found.append(item)
        return found

    def put_many(self, items: Iterable[dict[str, Any]]) -> None:
        """Store successful results under every citation form they answer to."""
        now = time.time()
        with self._lock:
            for item in items:
                keys = {str(item.get("citation") or "")}
                keys.update(str(n) for n in (item.get("normalized_citations") or []))
                keys.discard("")
                for key in keys:
                    self._data[key] = {"item": item, "cached_at": now}
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            payload = json.dumps(self._data, indent=1)
        # Write-then-rename: a crash mid-write must not leave a corrupt cache
        # that reads as "these citations do not exist".
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


@dataclass
class RateBudget:
    """Rolling-window token bucket over per-minute, per-hour, and per-day caps.

    **Process-local by design.** The clock is :func:`time.monotonic`, whose epoch
    is the process start, so the day window cannot survive a restart and the
    125/day cap is enforced per process, not per calendar day. This is a
    deliberate choice, not an oversight: persisting the budget would need a
    durable store, and this module is deployed as a single long-lived backend
    process where a restart is rare and CourtListener enforces its own
    server-side limit as the real backstop. If the daily cap ever has to hold
    across restarts, `_history` needs to move to a persistent store keyed on wall
    time — see REMEDIATION for the decision record.
    """

    per_minute: int = DEFAULT_PER_MINUTE
    per_hour: int = DEFAULT_PER_HOUR
    per_day: int = DEFAULT_PER_DAY
    # A deque, not a list: `_prune` trims from the left on every check, and
    # `list.pop(0)` is O(n) in the number of retained timestamps.
    _history: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _prune(self, now: float) -> None:
        cutoff = now - 24 * 60 * 60
        # append-only and sorted by time; trim from the left in O(1) per item
        while self._history and self._history[0] < cutoff:
            self._history.popleft()

    def check(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._prune(now)
            minute_count = sum(1.0 for t in self._history if t > now - 60)
            hour_count = sum(1.0 for t in self._history if t > now - 3600)
            day_count = len(self._history)
            if (
                minute_count >= self.per_minute
                or hour_count >= self.per_hour
                or day_count >= self.per_day
            ):
                raise BudgetExhausted(
                    f"CourtListener rate budget exhausted: {minute_count}/{self.per_minute} "
                    f"min, {hour_count}/{self.per_hour} hr, {day_count}/{self.per_day} day"
                )

    def spend(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._history.append(now)

    def counts(self, now: float | None = None) -> Mapping[str, int]:
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._prune(now)
            return {
                "minute": sum(1 for t in self._history if t > now - 60),
                "hour": sum(1 for t in self._history if t > now - 3600),
                "day": len(self._history),
            }


@dataclass
class VerificationResult:
    """Outcome of verifying one position against CourtListener."""

    position_side: str
    citations: list[CitationSlot] = field(default_factory=list)
    error: str = ""


class CourtListenerClient:
    """One POST per position to /api/rest/v4/citation-lookup/."""

    def __init__(
        self,
        token: str,
        base_url: str = COURTLISTENER_BASE_URL,
        budget: RateBudget | None = None,
        cache: ContentCache | None = None,
        http: httpx.Client | None = None,
        cite_cache: PersistentCiteCache | None = None,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.budget = budget or RateBudget()
        self.cache = cache or ContentCache()
        self.http = http or httpx.Client()
        #: Optional cross-process cache. Without it every run re-verifies the
        #: same authorities and burns the daily quota again.
        self.cite_cache = cite_cache

    def lookup(
        self,
        text_block: str,
        ledger: Any | None = None,
        cites: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return citation-lookup response, spending budget only on real network calls."""

        cached = self.cache.get(text_block)
        if cached is not None:
            if ledger is not None:
                ledger.cache_hits += 1
            return cast(dict[str, Any], cached)

        # A persistent per-citation cache can answer the whole block without a
        # call, which is what makes a repeat run cost no quota.
        if self.cite_cache is not None and cites:
            hit = self.cite_cache.get_all(cites)
            if hit is not None:
                if ledger is not None:
                    ledger.cache_hits += 1
                return {"results": hit}

        self.budget.check()

        try:
            resp = self.http.post(
                f"{self.base_url}/citation-lookup/",
                headers={"Authorization": f"Token {self.token}"},
                data={"text": text_block},
                timeout=30.0,
            )
        except httpx.TimeoutException:
            result: dict[str, Any] = {"error": "timeout", "results": []}
            return result
        except httpx.RequestError as exc:
            return {"error": f"request_error: {exc}", "results": []}

        if resp.status_code == 429:
            return {"error": "rate_limited", "results": []}
        if resp.status_code >= 500:
            return {"error": f"server_error:{resp.status_code}", "results": []}
        if resp.status_code == 401:
            return {"error": "unauthorized", "results": []}

        try:
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            return {"error": f"http_error:{exc.response.status_code}", "results": []}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"decode_error: {exc}", "results": []}

        self.budget.spend()
        self.cache.set(text_block, data)
        if self.cite_cache is not None:
            results = data if isinstance(data, list) else data.get("results", [])
            self.cite_cache.put_many(r for r in results if isinstance(r, dict))
        if ledger is not None:
            ledger.calls_spent += 1
        return cast(dict[str, Any], data)


def _resolves(match: dict[str, Any] | None) -> tuple[bool, str]:
    """Does this citation-lookup result confirm the citation?

    ``200`` is a single match. ``300`` means CourtListener returned several
    clusters, which is usually *duplicate records of one opinion* rather than an
    ambiguous citation — it holds two entries for Sorrell, Rice, West Virginia
    v. EPA, AOSI II and Humanitarian Law Project, among others. Treating 300 as
    unverified made five valid Supreme Court authorities permanently
    unverifiable, which reads as a fabricated citation rather than a quirk of
    the upstream database.

    A 300 is accepted only when the clusters are the *same* opinion. When they
    are genuinely different cases the citation really is ambiguous, and guessing
    which one was meant is exactly the error the gate exists to catch.

    Sameness is decided on the filing date first and the case name second. Name
    matching alone is too brittle: CourtListener stores AOSI II twice as
    "Agency for Int'l Dev. v. Alliance for Open Soc'y Int'l, Inc." and
    "Agency for Int'l Development v. Alliance for Open Society" — one opinion in
    two abbreviation styles. Two genuinely distinct cases sharing a citation and
    a filing date is vanishingly unlikely by comparison.
    """
    if not match or not match.get("clusters"):
        return False, ""
    status = match.get("status")
    if status == 200:
        return True, ""
    if status != 300:
        return False, ""

    clusters = match["clusters"]
    note = f" ({len(clusters)} duplicate cluster(s) for one opinion)"

    dates = {str(c.get("date_filed", "")).strip() for c in clusters}
    dates.discard("")
    if len(dates) == 1:
        return True, note

    names = {" ".join(str(c.get("case_name", "")).lower().split()) for c in clusters}
    names.discard("")
    if len(names) == 1:
        return True, note
    return False, ""


def verify_position(
    position: Position,
    client: CourtListenerClient,
    ledger: Any | None = None,
) -> VerificationResult:
    """Verify every filled citation in a position with one POST to CourtListener."""

    result = VerificationResult(position_side=position.side)
    filled = [slot for slot in position.propositions if slot.normalized_cite]
    if not filled:
        result.citations = list(position.propositions)
        return result

    # The endpoint parses a free-text block; batch every normalized cite in it.
    block = " ".join(slot.normalized_cite for slot in filled)
    data = client.lookup(
        block, ledger=ledger, cites=[slot.normalized_cite for slot in filled]
    )

    if isinstance(data, list):
        data = {"results": data}

    if data.get("error"):
        result.error = data["error"]
        for slot in position.propositions:
            new_slot = slot.model_copy()
            if new_slot.normalized_cite:
                new_slot.status = SlotStatus.NOT_FOUND
                new_slot.note = f"verification failed: {data['error']}"
            result.citations.append(new_slot)
        return result

    found: dict[str, dict[str, Any]] = {}
    for item in data.get("results", []):
        citation = item.get("citation") or ""
        normalized = item.get("normalized_citations") or [citation]
        for norm in normalized:
            found[norm] = item
        if citation:
            found[citation] = item

    # Results are matched back to slots by string key. When CourtListener returns
    # fewer results than were sent, or normalises a cite differently from the way
    # it was submitted, the slot falls to NOT_FOUND — and without this record the
    # note ("CourtListener status: no_match") gives no way to tell a genuinely
    # unknown citation from a key-matching failure on our side.
    _LOG.debug(
        "citation-lookup for %s: sent block %r; returned keys %r",
        position.side,
        block,
        sorted(found),
    )

    for slot in position.propositions:
        new_slot = slot.model_copy()
        if not new_slot.normalized_cite:
            result.citations.append(new_slot)
            continue

        match = found.get(new_slot.normalized_cite)
        # Try a whitespace-normalized fallback for tolerant matching.
        if match is None:
            compact = " ".join(new_slot.normalized_cite.split())
            match = found.get(compact)

        resolved, why = _resolves(match)
        if resolved:
            clusters = match["clusters"]  # type: ignore[index]
            new_slot.status = SlotStatus.VERIFIED
            new_slot.cluster_id = str(clusters[0].get("id", ""))
            new_slot.note = f"resolved by CourtListener v4 citation-lookup{why}"
        else:
            new_slot.status = SlotStatus.NOT_FOUND
            if match is None:
                # Name the mismatch explicitly rather than saying "no_match".
                _LOG.info(
                    "no result keyed on %r for %s; returned keys were %r",
                    new_slot.normalized_cite,
                    position.side,
                    sorted(found),
                )
                new_slot.note = (
                    f"CourtListener returned no result keyed on "
                    f"{new_slot.normalized_cite!r} "
                    f"({len(found)} key(s) returned for {len(filled)} cite(s) sent)"
                )
            else:
                new_slot.note = f"CourtListener status: {match.get('status')}"
        result.citations.append(new_slot)

    return result
