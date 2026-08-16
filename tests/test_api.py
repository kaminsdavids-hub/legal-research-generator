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
    from legal_research.config import get_settings

    assert len(payload["model_answers"]) == len(get_settings().multi_chat_panel_members)
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

    job = _jobs.submit("session-x", "draft", lambda _job: boom())
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
    first = _jobs.submit("busy-session", "draft", lambda _job: release.wait())
    try:
        with pytest.raises(SessionBusy) as caught:
            _jobs.submit("busy-session", "verify", lambda _job: None)
        assert caught.value.job_id == first.id
    finally:
        release.set()
        first.wait(timeout=10)

    # Once it finishes the session is free again.
    assert _jobs.submit("busy-session", "verify", lambda _job: None).wait(timeout=10)


def test_the_busy_session_surfaces_as_409(client: TestClient) -> None:
    """Not 429. This is not rate limiting, it is a statement that the session is
    in a state where a second step cannot start."""

    import threading

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "async"}).json()
    release = threading.Event()
    held = _jobs.submit(session["session_id"], "draft", lambda _job: release.wait())
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
    first = _jobs.submit("chatty", "multi-chat", lambda _job: release.wait(), mutates=False)
    try:
        second = _jobs.submit("chatty", "dialectic", lambda _job: None, mutates=False)
        assert second.wait(timeout=10), "the second conversation was blocked by the first"
        # And a read-only job in flight does not block a writer either.
        writer = _jobs.submit("chatty", "draft", lambda _job: None)
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
    held = _jobs.submit("writer-session", "draft", lambda _job: release.wait())
    try:
        with pytest.raises(SessionBusy):
            _jobs.submit("writer-session", "verify", lambda _job: None)
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


def test_socratic_exclusivity_is_decided_by_the_request_not_the_step() -> None:
    """The last conversion broke the rule the previous one established. The same
    step name is read-only when it only answers and exclusive when it applies
    the revision, so a name-based set cannot express it: it would either lock
    out concurrent questions that harm nothing, or let a revision land while a
    draft rewrites the same section."""

    from legal_research.api.app import _mutates
    from legal_research.api.schemas import JobRequest

    assert _mutates(JobRequest(step="socratic", apply_revision=False)) is False
    assert _mutates(JobRequest(step="socratic", apply_revision=True)) is True
    assert _mutates(JobRequest(step="multi-chat")) is False
    assert _mutates(JobRequest(step="dialectic")) is False
    assert _mutates(JobRequest(step="draft")) is True
    assert _mutates(JobRequest(step="run-all")) is True


def test_an_applying_socratic_job_takes_the_session_lock() -> None:
    """The consequence that matters: a revision that writes must not run beside
    a draft rewriting the same manuscript."""

    import threading

    from legal_research.api.app import _jobs
    from legal_research.api.jobs import SessionBusy

    release = threading.Event()
    held = _jobs.submit("socratic-session", "draft", lambda _job: release.wait())
    try:
        # Answering only: allowed alongside the draft.
        asking = _jobs.submit("socratic-session", "socratic", lambda _job: None, mutates=False)
        assert asking.wait(timeout=10)
        # Applying: refused, because it writes.
        with pytest.raises(SessionBusy):
            _jobs.submit("socratic-session", "socratic", lambda _job: None, mutates=True)
    finally:
        release.set()
        held.wait(timeout=10)


def test_an_unknown_section_is_404_before_a_job_starts(client: TestClient) -> None:
    """The synchronous route answers an unknown section with a 404. Submitting a
    job certain to fail would demote that to a poll and a failure state."""

    session = client.post("/api/sessions", json={"title": "socratic"}).json()

    resp = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "socratic", "message": "is this paragraph doing work?",
              "section_id": "no-such-section"},
    )

    assert resp.status_code == 404
    assert "no-such-section" in resp.json()["detail"]


def test_socratic_runs_as_a_job_and_returns_its_own_response(client: TestClient) -> None:
    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "socratic"}).json()
    sid = session["session_id"]
    # A run is needed before there is a paragraph to interrogate.
    run = client.post(
        f"/api/sessions/{sid}/jobs",
        json={"step": "run-all", "idea": "open weights and the EAR", "title": "socratic"},
    ).json()["job_id"]
    assert _jobs.get(run).wait(timeout=300)
    section_id = client.get(f"/api/sessions/{sid}").json()["outline"][0]["id"]

    submitted = client.post(
        f"/api/sessions/{sid}/jobs",
        json={"step": "socratic", "message": "what work is this paragraph doing?",
              "section_id": section_id},
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "succeeded", body["error"]
    assert body["result"]["section_id"] == section_id
    assert body["result"]["applied"] is False


# --------------------------------------------------------------------------- #
# Pushing a job's outcome instead of making the client ask
# --------------------------------------------------------------------------- #
def _sse_events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data). Comment frames are skipped, which
    is what a real client does with a heartbeat."""

    import json as _json

    out = []
    for block in body.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if not lines:
            continue
        event = next((ln[len("event: ") :] for ln in lines if ln.startswith("event: ")), "")
        data = next((ln[len("data: ") :] for ln in lines if ln.startswith("data: ")), "")
        if event and data:
            out.append((event, _json.loads(data)))
    return out


def test_the_stream_reports_a_job_that_already_finished(client: TestClient) -> None:
    """State first, before any waiting. A client that connects after the job
    ended must still be told, or it waits forever for an event that has already
    happened."""

    from legal_research.api.app import _jobs

    job = _jobs.submit("stream-session", "ideate", lambda _job: {"ok": True})
    assert job.wait(timeout=30)

    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(resp.iter_text()))

    assert [name for name, _ in events] == ["state", "done"]
    assert events[-1][1]["state"] == "succeeded"
    assert events[-1][1]["result"] == {"ok": True}


def test_the_stream_delivers_a_result_the_client_never_asked_twice_for(
    client: TestClient,
) -> None:
    """The point of the route: one connection, and the answer arrives the moment
    it exists rather than at the next poll."""

    import threading

    from legal_research.api.app import _jobs

    release = threading.Event()
    job = _jobs.submit("stream-session-2", "multi-chat",
                       lambda _job: release.wait() or {"answer": "x"}, mutates=False)
    threading.Timer(0.3, release.set).start()

    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        events = _sse_events("".join(resp.iter_text()))

    assert events[0][1]["state"] == "running"
    assert events[-1][0] == "done"
    assert events[-1][1]["state"] == "succeeded"


def test_a_failure_arrives_on_the_stream_too(client: TestClient) -> None:
    """A stream that ended silently would leave the client unable to tell a
    failure from a dropped connection."""

    from legal_research.api.app import _jobs

    def boom() -> None:
        raise RuntimeError("panel unreachable")

    job = _jobs.submit("stream-session-3", "dialectic", lambda _job: boom(), mutates=False)
    assert job.wait(timeout=30)

    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        events = _sse_events("".join(resp.iter_text()))

    assert events[-1][1]["state"] == "failed"
    assert "panel unreachable" in events[-1][1]["error"]


def test_the_stream_and_the_poll_describe_a_job_identically(client: TestClient) -> None:
    """One definition of what a client is told, so the two routes cannot drift
    into describing the same job differently."""

    from legal_research.api.app import _jobs

    job = _jobs.submit("stream-session-4", "ideate", lambda _job: {"ok": 1})
    assert job.wait(timeout=30)

    polled = client.get(f"/api/jobs/{job.id}").json()
    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        streamed = _sse_events("".join(resp.iter_text()))[-1][1]

    assert polled == streamed


def test_streaming_an_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_the_panel_reports_each_model_as_it_finishes(client: TestClient) -> None:
    """The change worth making. A five-model panel takes tens of seconds and
    returns nothing until the last one is done; without per-model events a
    caller can only show a spinner and hope."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "panel"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "multi-chat", "message": "does the EAR reach open weights?"},
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    events = client.get(f"/api/jobs/{job_id}").json()["events"]
    kinds = [e["event"] for e in events]

    assert kinds[0] == "panel_started"
    assert "synthesising" in kinds
    from legal_research.config import get_settings

    answered = [e for e in events if e["event"] == "model_answered"]
    # One per panel member, whoever they are: the membership is configuration
    # and moves with the hardware, but the correspondence is the invariant.
    assert len(answered) == len(get_settings().multi_chat_panel_members)
    assert {e["name"] for e in answered} == set(get_settings().multi_chat_panel_members)
    for event in answered:
        assert event["model"] and isinstance(event["seconds"], float)
        assert isinstance(event["answered"], bool)


def test_progress_reaches_the_stream_as_it_happens(client: TestClient) -> None:
    """Events are pushed while the job runs, not bundled into the terminal
    frame — otherwise this would be polling with extra steps."""

    import threading

    from legal_research.api.app import _jobs

    release = threading.Event()

    def work(job) -> dict:  # noqa: ANN001 - the store passes the job in
        job.emit({"event": "model_answered", "model": "first"})
        release.wait(5)
        job.emit({"event": "model_answered", "model": "second"})
        return {"ok": True}

    job = _jobs.submit("progress-session", "multi-chat", work, mutates=False)
    threading.Timer(0.3, release.set).start()

    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        events = _sse_events("".join(resp.iter_text()))

    progress = [data for name, data in events if name == "progress"]
    assert [p["model"] for p in progress] == ["first", "second"]
    assert events[-1][0] == "done"


def test_a_late_subscriber_sees_the_whole_run(client: TestClient) -> None:
    """Progress already recorded goes out with the opening frame, so connecting
    late shows the run and not just its tail."""

    from legal_research.api.app import _jobs

    def work(job) -> dict:  # noqa: ANN001
        job.emit({"event": "model_answered", "model": "early"})
        return {"ok": True}

    job = _jobs.submit("late-session", "multi-chat", work, mutates=False)
    assert job.wait(timeout=30)

    with client.stream("GET", f"/api/jobs/{job.id}/events") as resp:
        events = _sse_events("".join(resp.iter_text()))

    # Replayed as progress, so a client handles one kind of frame either way.
    assert events[0][0] == "state"
    progress = [data for name, data in events if name == "progress"]
    assert [p["model"] for p in progress] == ["early"]


def test_a_callback_that_raises_cannot_break_the_exchange() -> None:
    """Reporting on the work must not be able to destroy the work."""

    from legal_research.config import Settings
    from legal_research.multi_chat import MultiModelChat

    engine = MultiModelChat(Settings(llm_mode="mock", retriever_mode="mock"))

    def hostile(_event: dict) -> None:
        raise RuntimeError("subscriber exploded")

    result = engine.chat("does the EAR reach open weights?", on_event=hostile)

    assert result.final_answer, "the exchange completed despite the callback"


def test_the_synchronous_route_reports_nothing(client: TestClient) -> None:
    """It returns once, at the end, so it has nowhere to put progress and the
    engine skips the reporting entirely."""

    session = client.post("/api/sessions", json={"title": "sync"}).json()

    resp = client.post(
        f"/api/sessions/{session['session_id']}/multi-chat", json={"message": "hello"}
    )

    assert resp.status_code == 200
    assert "events" not in resp.json()


def test_work_is_always_called_with_its_job() -> None:
    """One signature, always. Accepting either shape and inferring which was
    passed read `threading.Event.wait` as wanting the job, called it with the
    job as its timeout, and quietly turned the session lock off — four tests
    caught it, and the fix was to stop guessing."""

    from legal_research.api.app import _jobs

    received: list[object] = []
    job = _jobs.submit("signature-session", "ideate", lambda j: received.append(j) or {"ok": 1})
    assert job.wait(timeout=10)

    assert received == [job]


def test_the_dialectic_reports_each_stage(client: TestClient) -> None:
    """Unlike the multi-chat panel this pipeline is sequential and its stages are
    heterogeneous — a model call, a network round-trip to CourtListener, an NLI
    pass — so a spinner cannot distinguish a slow debate from a hung one."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "dialectic"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "dialectic", "message": "does the EAR reach open weights?"},
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "succeeded", body["error"]
    kinds = [e["event"] for e in body["events"]]

    # The pipeline's real order: both sides generated, then retrieval,
    # verification, cruxes, synthesis.
    assert kinds.index("position_generated") < kinds.index("retrieved")
    assert kinds.index("retrieved") < kinds.index("verified")
    assert kinds.index("verified") < kinds.index("cruxes_extracted")
    assert kinds.index("cruxes_extracted") < kinds.index("synthesising")

    sides = [e["side"] for e in body["events"] if e["event"] == "position_generated"]
    assert sides == ["thesis", "antithesis"]


def test_an_empty_crux_table_is_still_reported(client: TestClient) -> None:
    """Zero cruxes is a real outcome, not a failure. Reporting nothing would let
    a reader assume the stage had not run."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "dialectic"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "dialectic", "message": "does the EAR reach open weights?"},
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    events = client.get(f"/api/jobs/{job_id}").json()["events"]
    (cruxes,) = [e for e in events if e["event"] == "cruxes_extracted"]

    assert isinstance(cruxes["count"], int)
    # When it is zero the engine's own note says why, and it travels with it.
    if cruxes["count"] == 0:
        assert cruxes["note"]


def test_a_hostile_subscriber_cannot_break_a_dialectic_turn() -> None:
    """Same discipline as the panel: reporting on the work must not be able to
    destroy the work."""

    from legal_research.api.app import _get_dialectic

    def hostile(_event: dict) -> None:
        raise RuntimeError("subscriber exploded")

    turn = _get_dialectic().chat("does the EAR reach open weights?", on_event=hostile)

    assert turn.question


def test_run_all_reports_every_stage_including_the_fallbacks(client: TestClient) -> None:
    """The longest silence in the application: nine agents in sequence, minutes
    of work, nothing until the end. Degraded stages are reported too — a run
    where most agents fell back is exactly the run a caller needs to see, and
    reporting only the healthy path would make a limping run look like a fast
    one."""

    from legal_research.api.app import _jobs

    session = client.post("/api/sessions", json={"title": "instrumented"}).json()
    job_id = client.post(
        f"/api/sessions/{session['session_id']}/jobs",
        json={"step": "run-all", "idea": "open weights and the EAR", "title": "instrumented"},
    ).json()["job_id"]
    assert _jobs.get(job_id).wait(timeout=300)

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "succeeded", body["error"]
    events = body["events"]

    assert events, "run-all reported nothing"
    assert all(e["event"] == "step_completed" for e in events)
    # One event per step in the summary, in the same order, no more and no less.
    assert [e["agent"] for e in events] == [s["agent"] for s in body["result"]["steps"]]
    # Index is 1-based and monotonic, so a display can show "step 4 of n".
    assert [e["index"] for e in events] == list(range(1, len(events) + 1))
    assert all(isinstance(e["degraded"], bool) for e in events)


def test_a_degraded_stage_is_marked_as_such(client: TestClient) -> None:
    """`_degraded_step` marks its payload, and the flag travels. A progress
    display that conflated a fallback with a success would report steady
    progress through a collapsing run."""

    from legal_research.config import Settings
    from legal_research.pipeline import LegalResearchPipeline

    pipe = LegalResearchPipeline(Settings(llm_mode="mock", retriever_mode="mock"))
    seen: list[dict] = []

    # Force the first stage to fall back.
    pipe.brainstorm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[method-assign]
    pipe.run_all("open weights", title="degraded", bb=pipe.new_session("degraded"), on_event=seen.append)

    assert seen[0]["degraded"] is True
    assert any(e["degraded"] is False for e in seen), "later stages still succeeded"


def test_a_hostile_subscriber_cannot_break_a_run() -> None:
    """Third time this discipline appears; it holds here too."""

    from legal_research.config import Settings
    from legal_research.pipeline import LegalResearchPipeline

    pipe = LegalResearchPipeline(Settings(llm_mode="mock", retriever_mode="mock"))

    def hostile(_event: dict) -> None:
        raise RuntimeError("subscriber exploded")

    result = pipe.run_all("open weights", title="hostile", bb=pipe.new_session("hostile"), on_event=hostile)

    assert result.steps, "the run completed despite the callback"


def test_every_step_the_page_offers_is_one_the_backend_accepts() -> None:
    """The page's buttons and the backend's allow-list are two lists that must
    agree, in different languages, in different files. A button naming a step
    the backend rejects fails at the click, which is the worst place to find
    out."""

    import re
    from pathlib import Path

    from legal_research.api.app import _JOB_STEPS, QUESTION_STEPS, RUN_ALL

    page = Path("frontend/app/page.tsx").read_text(encoding="utf-8")
    offered = re.findall(r'key: "([a-z-]+)"', page)
    accepted = set(_JOB_STEPS) | {RUN_ALL} | QUESTION_STEPS | {"socratic"}

    assert offered, "no steps found in the page; the pattern must have changed"
    assert set(offered) <= accepted, f"page offers {set(offered) - accepted}"


def test_the_client_names_every_step_the_backend_accepts() -> None:
    """And the other direction: JobStep in the client is the type that stops a
    typo reaching the network, so it has to list what the backend allows."""

    from pathlib import Path

    from legal_research.api.app import _JOB_STEPS, QUESTION_STEPS, RUN_ALL

    api_ts = Path("frontend/lib/api.ts").read_text(encoding="utf-8")
    declared = api_ts.split("export type JobStep =")[1].split(";")[0]

    for step in {*_JOB_STEPS, RUN_ALL, *QUESTION_STEPS, "socratic"}:
        assert f'"{step}"' in declared, f"JobStep omits {step!r}"


def test_a_substituted_panel_answer_is_not_reported_as_an_answer() -> None:
    """Found by running it. Every one of five models timed out, each was replaced
    with canned text, and the progress strip rendered five successes — because
    `answered` was `bool(content)` and a substitute is not empty. That is the
    conflation the run-all recorder avoids, reintroduced one module over."""

    from legal_research.multi_chat import MultiModelChat

    assert MultiModelChat.answered("The EAR reaches published weights because…")
    for substitute in (
        "Degraded panel answer (gpt-oss:20b; timeout recovered): For 'x', begin with…",
        "Degraded panel answer (gemma4:latest; low-substance rewrite fallback): …",
        "(skipped: model jury time budget exceeded)",
        "(no response)",
        "(unavailable: ReadTimeout)",
        "   ",
    ):
        assert not MultiModelChat.answered(substitute), substitute


def test_the_panel_membership_is_configuration_not_code() -> None:
    """Panel size is a hardware question — five concurrent generations on one GPU
    cost each of them 2.3-2.6x their solo time — so it has to be answerable
    without editing the engine."""

    from legal_research.config import Settings
    from legal_research.multi_chat import MultiModelChat

    three = MultiModelChat(
        Settings(llm_mode="mock", multi_chat_panel_members=["gpt_oss", "apertus", "hermes3"])
    )

    assert [name for name, _ in three._panel_specs()] == ["gpt_oss", "apertus", "hermes3"]
    # Every model stays configured; membership decides only who is asked.
    assert three._settings.multi_chat_gemma4_model


def test_an_unknown_panel_member_raises_rather_than_being_skipped() -> None:
    """A panel quietly one member short is a panel whose disagreement measure is
    reading a different jury than the author thinks, and nothing downstream would
    notice."""

    import pytest as _pytest

    from legal_research.config import Settings
    from legal_research.multi_chat import MultiModelChat

    engine = MultiModelChat(
        Settings(llm_mode="mock", multi_chat_panel_members=["gpt_oss", "gpt_5_turbo"])
    )

    with _pytest.raises(ValueError, match="unknown panel member"):
        engine._panel_specs()

    empty = MultiModelChat(Settings(llm_mode="mock", multi_chat_panel_members=[]))
    with _pytest.raises(ValueError, match="no members"):
        empty._panel_specs()


def test_the_shipped_panel_is_the_four_that_answer() -> None:
    """Pinned to the measurement, 2026-08-16, and to two different reasons.

    gemma4 is in because it was only ever a latency casualty: with
    MAX_LOADED_MODELS=8 and NUM_PARALLEL=1 all seven warmed models stay
    resident, and it answers at 75.0s against a 110s cap.

    nemotron is out because it is a substance casualty, which residency cannot
    fix. Exercised through the API with all seven resident and no contention it
    returned "Degraded panel answer (... low-substance rewrite fallback)" --
    filler that never mentions the question's subject -- while comfortably
    inside the cap. An earlier probe got a real answer from it, so it is
    unreliable rather than uniformly bad, which is worse for a panel whose
    value is independent reads.

    So do not "fix" a regression here by raising the timeout: the model that
    was dropped was never short of time."""

    from legal_research.config import Settings, reset_settings

    reset_settings()

    settings = Settings()
    assert settings.multi_chat_panel_members == [
        "gpt_oss",
        "gemma4",
        "apertus",
        "hermes3",
    ]
    # nemotron stays configured -- the module warms every panel slot regardless
    # of membership, and the rescue and verifier paths still reach it.
    assert settings.multi_chat_nemotron_model == "nemotron-3-nano:4b"
    # The margin is the point: 110s cleared the slowest by 29s. A timeout raised
    # to paper over a slow panel would silently reintroduce the canned-text
    # substitution this panel size was originally cut to avoid.
    assert settings.multi_chat_per_model_timeout_seconds == 110.0
