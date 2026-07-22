"""API smoke tests using FastAPI's TestClient (spec §3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # The app is configured for live Ollama by default; tests must stay on the
    # deterministic mock backend so they run without GPU or network.
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    from legal_research.api.app import app

    return TestClient(app)


def test_health_and_config(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["llm_mode"] == "mock"
    assert "not legal advice" in cfg["disclaimer"].lower()


def test_session_lifecycle(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "Test"}).json()["session_id"]

    bb = client.post(
        f"/api/sessions/{sid}/brainstorm", json={"message": "Section 10(b) and scienter."}
    ).json()
    assert bb["brainstorm"], "brainstorm turn recorded"

    bb = client.post(f"/api/sessions/{sid}/ideate", json={"seed": "aiding and abetting"}).json()
    assert len(bb["ideas"]) >= 2

    idea_id = bb["ideas"][0]["id"]
    bb = client.post(
        f"/api/sessions/{sid}/ideas/select",
        json={"selections": [{"idea_id": idea_id, "priority": 0}]},
    ).json()
    assert any(i["status"] == "keep" for i in bb["ideas"])

    client.post(f"/api/sessions/{sid}/outline")
    bb = client.get(f"/api/sessions/{sid}").json()
    assert bb["outline"]


def test_run_all_endpoint(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "E2E"}).json()["session_id"]
    resp = client.post(
        f"/api/sessions/{sid}/run-all",
        json={"idea": "Section 10(b) requires scienter and bars aiding-and-abetting suits."},
    ).json()
    assert resp["shippable"] is True
    assert resp["steps"]

    report = client.get(f"/api/sessions/{sid}/report").json()["markdown"]
    assert "Citation Verification Report" in report

    preview = client.get(f"/api/sessions/{sid}/preview").json()["html"]
    assert "<!DOCTYPE html>" in preview


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/sessions/does-not-exist").status_code == 404
