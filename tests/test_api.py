"""API smoke tests using FastAPI's TestClient (spec §3)."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

import legal_research.multi_chat as multi_chat_module


def _count_words(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


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
    assert cfg["saul_model"]
    assert cfg["hermes3_model"]
    assert cfg["multi_chat_models"]
    assert cfg["multi_chat_verifiers"]
    assert isinstance(cfg["grammar_chain_enabled"], bool)
    assert cfg["grammar_chain_roles"]
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

    bb = client.get(f"/api/sessions/{sid}").json()
    assert bb["outline"]
    assert any(section["content"].strip() for section in bb["outline"])

    report = client.get(f"/api/sessions/{sid}/report").json()["markdown"]
    assert "Citation Verification Report" in report

    preview = client.get(f"/api/sessions/{sid}/preview").json()["html"]
    assert "<!DOCTYPE html>" in preview


def test_socratic_paragraph_revision_endpoint(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "Socratic"}).json()["session_id"]
    client.post(
        f"/api/sessions/{sid}/brainstorm", json={"message": "Section 10(b) and scienter."}
    )
    client.post(f"/api/sessions/{sid}/ideate", json={"seed": "aiding and abetting"})
    client.post(f"/api/sessions/{sid}/outline")
    client.post(f"/api/sessions/{sid}/research")
    client.post(f"/api/sessions/{sid}/draft")

    bb = client.get(f"/api/sessions/{sid}").json()
    first = next(s for s in bb["outline"] if s["content"])
    before = first["content"]

    resp = client.post(
        f"/api/sessions/{sid}/revise/socratic",
        json={
            "section_id": first["id"],
            "paragraph_index": 0,
            "message": "Stress-test this paragraph and propose one stronger doctrinal move.",
            "history": [],
            "apply_revision": True,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["assistant"]
    assert payload["applied"] is True
    assert isinstance(payload["cycle_break_triggered"], bool)
    assert isinstance(payload["novelty_score"], float)
    assert isinstance(payload["rewrite_delta_score"], float)

    updated = client.get(f"/api/sessions/{sid}").json()["outline"]
    after = next(s for s in updated if s["id"] == first["id"])["content"]
    assert after
    assert after != before


def test_socratic_paragraph_revision_mode_endpoint(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "Socratic Mode"}).json()["session_id"]
    client.post(
        f"/api/sessions/{sid}/brainstorm", json={"message": "Section 10(b) and scienter."}
    )
    client.post(f"/api/sessions/{sid}/ideate", json={"seed": "aiding and abetting"})
    client.post(f"/api/sessions/{sid}/outline")
    client.post(f"/api/sessions/{sid}/research")
    client.post(f"/api/sessions/{sid}/draft")

    bb = client.get(f"/api/sessions/{sid}").json()
    first = next(s for s in bb["outline"] if s["content"])

    resp = client.post(
        f"/api/sessions/{sid}/revise/socratic",
        json={
            "section_id": first["id"],
            "paragraph_index": 0,
            "message": "Expand this paragraph with deeper analysis and practical implications.",
            "history": [],
            "apply_revision": True,
            "mode": "expand_analysis",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["assistant"]
    assert payload["applied"] is True
    assert payload["mode"] == "expand_analysis"


def test_draft_essay_endpoint(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "Essay 2k"}).json()["session_id"]
    client.post(
        f"/api/sessions/{sid}/brainstorm", json={"message": "Section 10(b) and scienter."}
    )
    client.post(f"/api/sessions/{sid}/ideate", json={"seed": "aiding and abetting"})
    client.post(f"/api/sessions/{sid}/outline")
    client.post(f"/api/sessions/{sid}/research")

    resp = client.post(f"/api/sessions/{sid}/draft/essay", json={"target_words": 2000})
    assert resp.status_code == 200
    bb = resp.json()
    draft_text = "\n\n".join(s["content"] for s in bb["outline"] if s["content"])
    assert _count_words(draft_text) >= 1200


def test_multi_chat_endpoint(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"title": "Chat"}).json()["session_id"]
    resp = client.post(
        f"/api/sessions/{sid}/multi-chat",
        json={
            "message": "What is the narrowest way to frame scienter under Rule 10b-5?",
            "history": [{"role": "user", "content": "Prior context."}],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["final_answer"]
    assert len(payload["model_answers"]) == 5
    assert len(payload["verifiers"]) == 2


def test_run_all_recovers_when_draft_stage_fails(client: TestClient, monkeypatch) -> None:
    api_app_module = importlib.import_module("legal_research.api.app")

    def _failing_draft(bb, target_words=None):  # noqa: ANN001, ARG001
        raise RuntimeError("simulated draft failure")

    monkeypatch.setattr(api_app_module.pipeline, "draft", _failing_draft)

    sid = client.post("/api/sessions", json={"title": "Draft Failure Recovery"}).json()["session_id"]
    response = client.post(
        f"/api/sessions/{sid}/run-all",
        json={"idea": "Scienter under Rule 10b-5 and PSLRA pleading"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["steps"]
    assert any("fallback" in step["summary"].lower() for step in payload["steps"])

    bb = client.get(f"/api/sessions/{sid}").json()
    assert bb["outline"]
    assert any(section["content"].strip() for section in bb["outline"])


def test_run_all_terminal_error_is_sanitized(client: TestClient, monkeypatch) -> None:
    api_app_module = importlib.import_module("legal_research.api.app")

    def _fatal_run_all(*args, **kwargs):  # noqa: ANN001, ARG001
        raise RuntimeError("secret RuntimeError details should not be exposed")

    monkeypatch.setattr(api_app_module.pipeline, "run_all", _fatal_run_all)

    sid = client.post("/api/sessions", json={"title": "Run All Fatal"}).json()["session_id"]
    response = client.post(
        f"/api/sessions/{sid}/run-all",
        json={"idea": "Section 10(b) scienter standards"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"].lower()
    assert "run-all unavailable" in detail
    assert "before fallback recovery" in detail
    assert "runtimeerror" not in detail
    assert "secret" not in detail


def test_multi_chat_uses_rescue_when_panel_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "30")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_SYNTHESIS_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_VERIFIER_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_RESCUE_TIMEOUT_SECONDS", "5")

    from legal_research.config import Settings, reset_settings
    from legal_research.multi_chat import MultiModelChat

    reset_settings()

    class _FakeLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float) -> None:
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            if self.name == "panel_rescue":
                return (
                    "The controlling legal rule requires structured scienter analysis under "
                    "the governing standard, with element-by-element evaluation of record "
                    "facts and inferential proof. The factual matrix should isolate timeline "
                    "evidence, internal statements, and motive indicators while distinguishing "
                    "what is established from what remains uncertain.\n\n"
                    "A complete answer must also test the strongest counterargument that the "
                    "record supports only negligence, then explain why competing inferences do "
                    "or do not satisfy the controlling doctrinal threshold."
                )
            raise RuntimeError("simulated upstream failure")

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _FakeLLM)

    chat = MultiModelChat(Settings())
    result = chat.chat("What is scienter under Rule 10b-5?", history=[])
    assert "controlling legal rule" in result.final_answer.lower()
    assert "element" in result.final_answer.lower()
    assert "counterargument" in result.final_answer.lower()


def test_multi_chat_guarantees_analytic_fallback_when_all_models_fail(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "1")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("LRG_MULTI_CHAT_SYNTHESIS_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_VERIFIER_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("LRG_MULTI_CHAT_RESCUE_TIMEOUT_SECONDS", "1")

    from legal_research.config import Settings, reset_settings
    from legal_research.multi_chat import MultiModelChat

    reset_settings()

    class _AlwaysFailLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float) -> None:
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            raise RuntimeError("simulated outage")

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _AlwaysFailLLM)

    chat = MultiModelChat(Settings())
    result = chat.chat("What is scienter under Rule 10b-5?", history=[])
    assert "degraded-mode analytical answer" in result.final_answer.lower()
    assert "controlling legal rule" in result.final_answer.lower()


def test_multi_chat_recovers_readtimeout_for_every_model(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "30")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_SYNTHESIS_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_VERIFIER_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("LRG_MULTI_CHAT_RESCUE_TIMEOUT_SECONDS", "5")

    from legal_research.config import Settings, reset_settings
    from legal_research.multi_chat import MultiModelChat

    reset_settings()

    class ReadTimeout(Exception):
        pass

    class _TimeoutLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float) -> None:
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            raise ReadTimeout("simulated timeout")

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _TimeoutLLM)

    chat = MultiModelChat(Settings())
    result = chat.chat("What is the controlling scienter standard under Rule 10b-5?", history=[])

    assert result.final_answer
    assert "readtimeout" not in result.final_answer.lower()
    assert all("(unavailable: readtimeout)" not in ans.content.lower() for ans in result.model_answers)
    assert all("readtimeout" not in v.verdict.lower() for v in result.verifiers)


def test_multi_chat_rewrites_weak_panel_answers(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")

    from legal_research.config import Settings, reset_settings
    from legal_research.multi_chat import MultiModelChat

    reset_settings()

    class _WeakThenStrongLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float) -> None:
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            system = messages[0].content.lower() if messages else ""
            if "rewriting a legal panel draft" in system:
                return (
                    "The controlling rule requires element-by-element analysis of scienter, "
                    "materiality, and causal connection under governing precedent. The record "
                    "evidence should be separated into direct proof and inferential proof, with "
                    "specific allegations tied to each doctrinal element.\n\n"
                    "On the factual account, internal statements, timeline evidence, and disclosure "
                    "practice can support scienter inferences, but alternative explanations must be "
                    "tested against the same record facts.\n\n"
                    "The strongest counterargument is that the evidence shows negligence rather than "
                    "intent. The dispositive uncertainty is whether the most probative facts support "
                    "conscious disregard under the controlling standard."
                )
            if self.name in {"panel_synth", "panel_rescue"}:
                return "This is a complex issue and it depends on many factors overall."
            if "legal-analysis verifier" in system:
                return "RISK: medium\nCONCERN: generic fallback test path.\nSUGGESTION: review manually."
            return "It depends. More information is needed."

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _WeakThenStrongLLM)

    chat = MultiModelChat(Settings())
    result = chat.chat("Assess scienter under Rule 10b-5 given conflicting internal communications.", history=[])

    assert result.final_answer
    assert "controlling rule" in result.final_answer.lower()
    assert "element" in result.final_answer.lower()
    assert "counterargument" in result.final_answer.lower()
    assert "it depends" not in result.final_answer.lower()


def test_multi_chat_uses_deterministic_fallback_when_quality_rewrite_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")

    from legal_research.config import Settings, reset_settings
    from legal_research.multi_chat import MultiModelChat

    reset_settings()

    class _AlwaysWeakLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float) -> None:
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            system = messages[0].content.lower() if messages else ""
            if "legal-analysis verifier" in system:
                return "RISK: medium\nCONCERN: weak test path.\nSUGGESTION: review manually."
            return "This is a complex issue and it depends on many factors overall."

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _AlwaysWeakLLM)

    chat = MultiModelChat(Settings())
    result = chat.chat("Analyze factual predicates for scienter under Rule 10b-5.", history=[])

    lowered = result.final_answer.lower()
    assert (
        "degraded-mode analytical answer" in lowered
        or "low-substance rewrite fallback" in lowered
    )
    assert "controlling legal rule" in lowered or "controlling legal standard" in lowered


def test_retrieval_smoke_endpoint_disabled_by_default(client: TestClient) -> None:
    resp = client.post("/api/retrieval/smoke", json={"query": "scienter under 10b-5", "k": 3})
    assert resp.status_code == 404
    assert "debug endpoints" in resp.json()["detail"]


def test_retrieval_smoke_endpoint_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_DEBUG_ENDPOINTS_ENABLED", "true")
    from legal_research.config import reset_settings

    reset_settings()
    from legal_research.api.app import app

    test_client = TestClient(app)
    resp = test_client.post(
        "/api/retrieval/smoke",
        json={"query": "scienter pleading requirements under Rule 10b-5", "k": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["retriever_mode"]
    assert payload["embed_model"]
    assert payload["retriever_impl"]
    assert payload["hits"]
    assert payload["hits"][0]["record_id"]


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/sessions/does-not-exist").status_code == 404
