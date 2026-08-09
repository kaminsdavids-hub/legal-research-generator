"""CourtListener v4 citation-lookup client with token-bucket rate budget."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.budget = budget or RateBudget()
        self.cache = cache or ContentCache()
        self.http = http or httpx.Client()

    def lookup(
        self,
        text_block: str,
        ledger: Any | None = None,
    ) -> dict[str, Any]:
        """Return citation-lookup response, spending budget only on real network calls."""

        cached = self.cache.get(text_block)
        if cached is not None:
            if ledger is not None:
                ledger.cache_hits += 1
            return cast(dict[str, Any], cached)

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

    A 300 is accepted only when every returned cluster names the same case. When
    the clusters name *different* cases the citation really is ambiguous, and
    guessing which one was meant is exactly the error the gate exists to catch.
    """
    if not match or not match.get("clusters"):
        return False, ""
    status = match.get("status")
    if status == 200:
        return True, ""
    if status != 300:
        return False, ""

    names = {
        " ".join(str(c.get("case_name", "")).lower().split())
        for c in match["clusters"]
    }
    names.discard("")
    if len(names) == 1:
        return True, f" ({len(match['clusters'])} duplicate cluster(s) for one opinion)"
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
    data = client.lookup(block, ledger=ledger)

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
