"""Cache-only citation lookup: what a missing token does and does not block.

All 34 case cites in ``data/corpus/openweights.jsonl`` were already cached from
earlier paid runs, and a tokenless eval consulted none of them — the client was
built inside an ``if token:`` branch, so every slot stayed NOT_FOUND and the
citation-integrity gate failed every question. A frontier comparison run that
way would have spent real money measuring a metric pinned at zero by a missing
credential.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.dialectic.verification import (
    CourtListenerClient,
    PersistentCiteCache,
    RateBudget,
)

CITE = "209 F.3d 481"
RESULT = {
    "citation": CITE,
    "status": 200,
    "clusters": [{"id": 1, "case_name": "Junger v. Daley"}],
}


class _ExplodingHTTP:
    """Any network call is a test failure: cache-only must not reach out."""

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("a tokenless client must not make a request")


@pytest.fixture
def cache(tmp_path: Path) -> PersistentCiteCache:
    path = tmp_path / "courtlistener.json"
    # Same shape as evals/.cache/courtlistener.json: {cite: {item, cached_at}}.
    path.write_text(json.dumps({CITE: {"item": RESULT, "cached_at": 4102444800.0}}))
    return PersistentCiteCache(path)


def _client(cache: PersistentCiteCache, token: str = "") -> CourtListenerClient:
    return CourtListenerClient(
        token=token, budget=RateBudget(), cache=None, http=_ExplodingHTTP(), cite_cache=cache
    )


def test_a_cached_cite_resolves_without_a_token(cache: PersistentCiteCache) -> None:
    data = _client(cache).lookup(f"See {CITE}.", cites=[CITE])

    assert "error" not in data
    assert data["results"] == [RESULT]


def test_an_uncached_cite_is_refused_rather_than_guessed(cache: PersistentCiteCache) -> None:
    """Not silently empty: "no_token" says why the slot cannot be verified."""

    data = _client(cache).lookup("See 550 U.S. 544.", cites=["550 U.S. 544"])

    assert data["error"] == "no_token"
    assert data["results"] == []


def test_cache_only_spends_no_budget(cache: PersistentCiteCache) -> None:
    budget = RateBudget()
    client = CourtListenerClient(
        token="", budget=budget, cache=None, http=_ExplodingHTTP(), cite_cache=cache
    )
    before = getattr(budget, "spent", 0)

    client.lookup("See 550 U.S. 544.", cites=["550 U.S. 544"])

    assert getattr(budget, "spent", 0) == before


def test_a_token_still_takes_the_network_path(cache: PersistentCiteCache) -> None:
    """The guard is the missing token, not the cache: with a token an uncached
    cite must still be looked up."""

    with pytest.raises(AssertionError, match="must not make a request"):
        _client(cache, token="real-token").lookup("See 550 U.S. 544.", cites=["550 U.S. 544"])


def test_the_eval_builds_a_client_even_without_a_token() -> None:
    """The defect was structural: no token meant no client, so the cache that
    could have answered was never consulted."""

    source = Path("evals/run_eval.py").read_text(encoding="utf-8")
    build = source.index("cite_cache = PersistentCiteCache(args.cite_cache)")
    guard = source.index("if token:", build)
    # The cache and client are constructed before the token is branched on.
    assert build < guard
    assert "CACHE-ONLY" in source
