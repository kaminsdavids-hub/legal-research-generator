"""The shared-key gate on /api/*.

It had no test. That mattered on 2026-08-16: setting ``LRG_API_KEY`` so the
backend could be exposed through a Tailscale Funnel turned twelve tests in
``test_api.py`` red without a line of source changing, and the failures read as
``KeyError: 'session_id'`` because a 401 body has no session in it. Nothing
described the gate's intended behaviour, so nothing distinguished "the gate
works" from "the suite is misconfigured".

The other API tests now pin ``LRG_API_KEY=""`` because they send no key. This
file is where the key is set deliberately, so the gate is exercised rather than
inherited.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

KEY = "test-key-not-a-real-secret"


@pytest.fixture
def gated(monkeypatch) -> TestClient:
    """A client against an app that requires the key.

    ``ApiKeyMiddleware`` is installed at import with the key read at that
    moment, so the module has to be reloaded after the environment is set --
    monkeypatching afterwards would leave the already-bound middleware alone and
    the test would pass against a gate that was never on.
    """
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_LLM_WARMUP_ENABLED", "false")
    monkeypatch.setenv("LRG_API_KEY", KEY)

    from legal_research.config import reset_settings

    reset_settings()
    module = importlib.reload(importlib.import_module("legal_research.api.app"))
    yield TestClient(module.app)

    # Leave the process as it was found: the app is a module-level singleton and
    # a gated one would follow this test into every file that imports it.
    reset_settings()
    importlib.reload(module)


def test_health_stays_open(gated: TestClient) -> None:
    """Netlify and any uptime check hit this without credentials."""
    assert gated.get("/api/health").status_code == 200


def test_a_protected_route_refuses_no_key(gated: TestClient) -> None:
    assert gated.get("/api/config").status_code == 401


def test_a_wrong_key_is_refused(gated: TestClient) -> None:
    """Rejected, not ignored — a wrong key must not fall through to open."""
    assert gated.get("/api/config", headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": KEY},
        {"Authorization": f"Bearer {KEY}"},
    ],
    ids=["x-api-key", "bearer"],
)
def test_both_header_forms_are_accepted(gated: TestClient, headers: dict) -> None:
    """auth.py documents both: Bearer for proxies, X-API-Key for a browser fetch,
    the second because refusing it pushes people to put the key in a query
    string where it lands in every access log."""
    assert gated.get("/api/config", headers=headers).status_code == 200


def test_options_is_exempt(gated: TestClient) -> None:
    """The CORS preflight carries no credentials by design.

    Gating it would fail every cross-origin request at the preflight with a CORS
    error, hiding the real cause -- a missing key on the request that follows.
    """
    resp = gated.options(
        "/api/config",
        headers={
            "Origin": "http://localhost:3001",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code != 401


def test_an_unknown_route_is_still_gated(gated: TestClient) -> None:
    """404 must not leak past the gate: whether a route exists is not public."""
    assert gated.get("/api/does-not-exist").status_code == 401


def test_the_gate_is_off_when_no_key_is_set(monkeypatch) -> None:
    """Empty key means open, which is the documented loopback-only posture.

    Worth pinning because it is a footgun: `if not self.key` in auth.py silently
    passes everything through, and that is safe only while the backend is bound
    to loopback. A future change that made an empty key mean "deny" would be a
    surprise; one that made a *set* key mean "allow" would be a breach.
    """
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_LLM_WARMUP_ENABLED", "false")
    monkeypatch.setenv("LRG_API_KEY", "")

    from legal_research.config import reset_settings

    reset_settings()
    module = importlib.reload(importlib.import_module("legal_research.api.app"))
    try:
        assert TestClient(module.app).get("/api/config").status_code == 200
    finally:
        reset_settings()
        importlib.reload(module)
