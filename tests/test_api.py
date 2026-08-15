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


# --------------------------------------------------------------------------- #
# The API key gate
# --------------------------------------------------------------------------- #
def _keyed_client(monkeypatch, key: str = "s3cret-for-tests") -> tuple[TestClient, str]:
    """A client against an app configured with a key.

    The middleware is bound when `app.py` is imported, so the setting has to be
    in place before the module object exists — hence the reload rather than
    patching an attribute afterwards. Testing the gate any other way would test
    a wiring the process never actually uses.
    """

    import importlib
    import sys

    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_API_KEY", key)
    from legal_research.config import reset_settings

    reset_settings()
    # sys.modules, not `import ... as`: the package re-exports the FastAPI
    # instance as `legal_research.api.app`, so the dotted name resolves to the
    # object rather than the module and reload() gets handed an app.
    import legal_research.api.app  # noqa: F401  (ensures it is in sys.modules)

    app_module = importlib.reload(sys.modules["legal_research.api.app"])
    return TestClient(app_module.app), key


def test_an_unkeyed_request_is_refused(monkeypatch) -> None:
    """Every /api route was open to anyone who could reach the host. The gate is
    the thing that stopped being true."""

    client, _ = _keyed_client(monkeypatch)

    resp = client.post("/api/sessions", json={"topic": "probe"})

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_both_header_forms_are_accepted(monkeypatch) -> None:
    """Bearer is what proxies already send; X-API-Key is what people paste. The
    alternative to accepting the second is that they put the key in a query
    string, where it lands in every access log."""

    client, key = _keyed_client(monkeypatch)

    assert client.get("/api/config", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/api/config", headers={"X-API-Key": key}).status_code == 200
    assert client.get("/api/config", headers={"X-API-Key": "wrong"}).status_code == 401
    # A bare token with no scheme is not a Bearer credential.
    assert client.get("/api/config", headers={"Authorization": key}).status_code == 401


def test_health_answers_without_a_key_but_config_does_not(monkeypatch) -> None:
    """A liveness probe that needs a secret is one nobody wires up. /api/config
    is not in the same class: it lists every model the host is running."""

    client, key = _keyed_client(monkeypatch)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/config").status_code == 401


def test_cors_preflight_is_not_gated(monkeypatch) -> None:
    """Browsers send preflight without credentials by design. Gating OPTIONS
    would surface a missing key as an opaque CORS error instead of a 401."""

    client, _ = _keyed_client(monkeypatch)

    resp = client.options(
        "/api/sessions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.status_code < 400


def test_an_empty_key_leaves_the_api_open(monkeypatch) -> None:
    """Off by default, because a mandatory secret on a loopback dev server is
    friction with no benefit. The startup warning is what keeps it from being a
    silent choice."""

    client, _ = _keyed_client(monkeypatch, key="")

    assert client.get("/api/config").status_code == 200


def test_the_startup_warning_fires_only_where_it_matters() -> None:
    """Binding to loopback is not proof of safety — a Funnel or reverse proxy
    publishes a localhost port to the internet, which is exactly how this API
    came to be publicly reachable. The check catches the obvious case and its
    docstring admits the rest."""

    from legal_research.api.auth import warn_if_unprotected

    said: list[str] = []
    assert warn_if_unprotected("0.0.0.0", "", said.append) is True
    assert "LRG_API_KEY" in said[0]
    assert warn_if_unprotected("0.0.0.0", "a-key", said.append) is False
    assert warn_if_unprotected("127.0.0.1", "", said.append) is False


# --------------------------------------------------------------------------- #
# Slow steps as jobs
# --------------------------------------------------------------------------- #
def test_submitting_a_step_returns_at_once_with_an_id(client: TestClient) -> None:
    """202, not 200: the work has been accepted and has not been done. A 200
    would tell every cache and client library that this holds a result."""

    session = client.post("/api/sessions", json={"title": "async"}).json()

    resp = client.post(f"/api/sessions/{session['session_id']}/jobs", json={"step": "ideate"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] and body["step"] == "ideate"
    assert body["state"] in {"running", "succeeded"}


def test_polling_reaches_a_terminal_state(client: TestClient) -> None:
    """The contract the proxy depends on: a client that saw `running` must
    eventually see something else, and each poll must be fast."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs", json={"step": "ideate"}
    ).json()["job_id"]

    assert _jobs.get(job_id).wait(timeout=120), "step did not finish"

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] in {"succeeded", "failed"}
    # The result is not inlined; the client fetches the session once, on success.
    assert "blackboard" not in body


def test_a_failing_step_is_reported_not_lost(client: TestClient) -> None:
    """A job that vanished or hung would leave the client polling forever with
    nothing to show a user. The step failed; the server did not."""

    from legal_research.api.app import _jobs
    from legal_research.api.jobs import JobState

    def boom() -> None:
        raise RuntimeError("the model server is down")

    job = _jobs.submit("session-x", "draft", boom)
    assert job.wait(timeout=10)

    assert job.state is JobState.FAILED
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["state"] == "failed"
    assert "the model server is down" in body["error"]
    # The message, not our stack frames.
    assert "Traceback" not in body["error"]


def test_a_second_step_on_one_session_is_refused(client: TestClient) -> None:
    """Steps mutate a shared blackboard in place, so two running against one
    session would interleave writes. Refused rather than queued: queueing hides
    from the caller that their step has not started."""

    import threading

    from legal_research.api.app import _jobs
    from legal_research.api.jobs import SessionBusy

    release = threading.Event()
    first = _jobs.submit("busy-session", "draft", release.wait)
    try:
        with pytest.raises(SessionBusy) as caught:
            _jobs.submit("busy-session", "verify", lambda: None)
        assert caught.value.job_id == first.id
    finally:
        release.set()
        first.wait(timeout=10)

    # Once it finishes the session is free again.
    assert _jobs.submit("busy-session", "verify", lambda: None).wait(timeout=10)


def test_the_busy_session_surfaces_as_409(client: TestClient) -> None:
    """Not 429. This is not rate limiting, it is a statement that the session is
    in a state where a second step cannot start."""

    import threading

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async"}).json()
    release = threading.Event()
    held = _jobs.submit(session["session_id"], "draft", release.wait)
    try:
        resp = client.post(f"/api/sessions/{session['session_id']}/jobs", json={"step": "verify"})
        assert resp.status_code == 409
        assert held.id in resp.json()["detail"]
    finally:
        release.set()
        held.wait(timeout=10)


def test_an_unknown_step_names_the_permitted_set(client: TestClient) -> None:
    """An allow-list, not getattr(pipeline, step): turning a path segment into
    an attribute lookup on a live object would let a caller reach anything the
    pipeline exposes."""

    session = client.post("/api/sessions", json={"title": "async"}).json()

    resp = client.post(
        f"/api/sessions/{session['session_id']}/jobs", json={"step": "_seed_fallback_ideas"}
    )

    assert resp.status_code == 400
    assert "research" in resp.json()["detail"]


def test_polling_an_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/nope").status_code == 404


def test_run_all_as_a_job_carries_what_the_blackboard_cannot(client: TestClient) -> None:
    """run-all produces a per-agent step log and a shippable verdict, neither of
    which is in the blackboard. As a synchronous route those came back in the
    response; as a job they would be lost unless the job carries them."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async run"}).json()
    submitted = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "run-all", "idea": "open weights and the EAR", "title": "async run"},
    )

    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300), "run-all did not finish"

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "succeeded", body["error"]
    assert body["result"]["steps"], "the per-agent log is the point of the summary"
    assert isinstance(body["result"]["shippable"], bool)
    # Still not the blackboard: a poller would re-download it every 2s.
    assert "blackboard" not in body["result"]


def test_the_run_all_job_advances_the_stored_session(client: TestClient) -> None:
    """The result the client fetches once, on success. The job mutates the same
    session the client already knows about, so there is nothing to reconcile."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async run"}).json()
    assert session["outline"] == []

    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "run-all", "idea": "open weights and the EAR", "title": "async run"},
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    after = client.get(f"/api/sessions/{session['session_id']}").json()
    assert after["session_id"] == session["session_id"]
    assert after["outline"], "the run should have advanced the session"


def test_a_plain_step_sets_no_result(client: TestClient) -> None:
    """Only steps producing something outside the blackboard set one. For
    everything else the result *is* the blackboard, and a duplicate would
    invite the client to read a stale copy."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs", json={"step": "ideate"}
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=120)

    assert client.get(f"/api/jobs/{job_id}").json()["result"] is None


def test_run_all_is_named_among_the_permitted_steps(client: TestClient) -> None:
    session = client.post("/api/sessions", json={"title": "async"}).json()

    detail = client.post(
        f"/api/sessions/{session['session_id']}/jobs", json={"step": "nope"}
    ).json()["detail"]

    assert "run-all" in detail


def test_multi_chat_and_dialectic_run_as_jobs(client: TestClient) -> None:
    """Both answer a question rather than advancing the manuscript, so the whole
    response is the job's result — there is no blackboard to fetch afterwards."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "chat"}).json()
    sid = session["session_id"]

    for step, marker in (("multi-chat", "final_answer"), ("dialectic", "cruxes")):
        submitted = client.post(
            f"/api/sessions/{sid}/jobs", json={"step": step, "message": "does the EAR reach weights?"}
        )
        assert submitted.status_code == 202, step
        job_id = submitted.json()["job_id"]
        assert _jobs.get(job_id).wait(timeout=300), f"{step} did not finish"

        body = client.get(f"/api/jobs/{job_id}").json()
        assert body["state"] == "succeeded", f"{step}: {body['error']}"
        assert marker in (body["result"] or {}), f"{step} lost its response shape"


def test_two_conversations_can_run_at_once(client: TestClient) -> None:
    """The correction this conversion forced. Multi-chat and dialectic never
    touch the blackboard, so putting them under the writers' lock would take
    away something a user can already do — hold two conversations — to prevent a
    corruption they cannot cause."""

    import threading

    from legal_research.api.app import _jobs

    release = threading.Event()
    first = _jobs.submit("chatty", "multi-chat", release.wait, mutates=False)
    try:
        second = _jobs.submit("chatty", "dialectic", lambda: None, mutates=False)
        assert second.wait(timeout=10), "the second conversation was blocked by the first"
        # And a read-only job in flight does not block a writer either.
        writer = _jobs.submit("chatty", "draft", lambda: None)
        assert writer.wait(timeout=10)
    finally:
        release.set()
        first.wait(timeout=10)


def test_a_writing_step_is_still_exclusive(client: TestClient) -> None:
    """The exemption is narrow: steps that mutate the blackboard keep the lock."""

    import threading

    from legal_research.api.app import _jobs
    from legal_research.api.jobs import SessionBusy

    release = threading.Event()
    held = _jobs.submit("writer-session", "draft", release.wait)
    try:
        with pytest.raises(SessionBusy):
            _jobs.submit("writer-session", "verify", lambda: None)
    finally:
        release.set()
        held.wait(timeout=10)


def test_an_empty_message_is_refused_before_a_job_starts(client: TestClient) -> None:
    """The synchronous route rejects an empty question with a 400. Submitting a
    job that is certain to fail would turn that into a poll and a failure state,
    which is a worse way to learn the same thing."""

    session = client.post("/api/sessions", json={"title": "chat"}).json()

    resp = client.post(
        f"/api/sessions/{session['session_id']}/jobs", json={"step": "dialectic", "message": "  "}
    )

    assert resp.status_code == 400
    assert "must not be empty" in resp.json()["detail"]
