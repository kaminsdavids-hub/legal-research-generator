"""Latency-budget scheduling for the model jury.

The panel used to run sequentially against stage caps that over-subscribed the
total latency budget (5x per-model + synthesis + citation repair + 2x verifier
exceeds ``multi_chat_max_latency_seconds``), so the panel consumed the whole
budget and every later stage fell through to its canned "time budget exceeded"
text. These tests pin the arithmetic that prevents that.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

import legal_research.multi_chat as multi_chat_module
from legal_research.config import Settings, reset_settings
from legal_research.multi_chat import _MAX_RESERVE_FRACTION, MultiModelChat

#: The panel these tests reason about, pinned rather than read from the ambient
#: configuration. `.env` now ships a three-model panel because five concurrent
#: generations do not fit this GPU's budget, and a test that quietly followed
#: that would be asserting facts about the deployment instead of the code -- and
#: would change its meaning again the next time the hardware does.
PANEL_MEMBERS = ["gpt_oss", "gemma4", "apertus", "nemotron", "hermes3"]
PANEL_SIZE = len(PANEL_MEMBERS)

ANALYSIS = (
    "The controlling rule requires scienter, and the doctrine breaks into elements "
    "that must each be satisfied on the record. The first element addresses falsity, "
    "and the factual predicates are the internal documents.\n\n"
    "Applying the standard element by element, the record establishes awareness while "
    "the inference of intent remains contested. The strongest counterargument is that "
    "the record shows negligence rather than deception.\n\n"
    "The outcome-determinative uncertainty is whether the documents show actual "
    "awareness at the time of the statement."
)


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_GROUNDING_ENABLED", "false")
    monkeypatch.setenv("LRG_MULTI_CHAT_PANEL_MEMBERS", json.dumps(PANEL_MEMBERS))
    reset_settings()
    return Settings()


def _install_llm(
    monkeypatch: pytest.MonkeyPatch, handler=None, hold_seconds: float = 0.0
) -> tuple[list[dict], dict[str, int]]:
    """Install a fake LLM recording each call plus the peak in-flight count.

    ``hold_seconds`` keeps each call open briefly. Without it a single worker
    thread can drain the whole queue before another starts, which would make any
    concurrency assertion depend on scheduling luck; peak in-flight is the
    deterministic signal.
    """

    calls: list[dict] = []
    lock = threading.Lock()
    gauge = {"in_flight": 0, "peak": 0}

    class _FakeLLM:
        def __init__(
            self, name: str, base_url: str, model: str, api_key: str, timeout: float
        ) -> None:
            self.name = name
            self.timeout = timeout

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            with lock:
                calls.append(
                    {
                        "name": self.name,
                        "timeout": self.timeout,
                        "thread": threading.current_thread().name,
                    }
                )
                gauge["in_flight"] += 1
                gauge["peak"] = max(gauge["peak"], gauge["in_flight"])
            try:
                if hold_seconds:
                    time.sleep(hold_seconds)
                return ANALYSIS if handler is None else handler(self.name)
            finally:
                with lock:
                    gauge["in_flight"] -= 1

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _FakeLLM)
    return calls, gauge


# --------------------------------------------------------------------------- #
# Reserve arithmetic
# --------------------------------------------------------------------------- #
def test_reserve_leaves_the_panel_a_usable_share(settings: Settings) -> None:
    chat = MultiModelChat(settings)
    reserve = chat._downstream_reserve_seconds()
    total = settings.multi_chat_max_latency_seconds

    assert 0 < reserve < total
    assert reserve <= total * _MAX_RESERVE_FRACTION + 1e-9


def test_reserve_is_clamped_when_stage_caps_exceed_the_whole_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short budget must still leave panel time, not skip the panel entirely."""

    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "60")
    reset_settings()
    short = Settings()
    chat = MultiModelChat(short)

    # Caps alone (45 + 40 + 2x18) far exceed the 60s budget.
    uncapped = (
        short.multi_chat_synthesis_timeout_seconds
        + short.multi_chat_citation_repair_timeout_seconds
        + 2 * short.multi_chat_per_verifier_timeout_seconds
    )
    assert uncapped > short.multi_chat_max_latency_seconds

    reserve = chat._downstream_reserve_seconds()
    assert reserve < short.multi_chat_max_latency_seconds
    wall, per_model = chat._panel_budget(PANEL_SIZE)
    assert wall > 0
    assert per_model >= 5.0


def test_panel_budget_never_oversubscribes_the_total(settings: Settings) -> None:
    chat = MultiModelChat(settings)
    wall, per_model = chat._panel_budget(PANEL_SIZE)
    reserve = chat._downstream_reserve_seconds()

    # The panel's wall budget plus the reserve fits inside the total.
    assert wall + reserve <= settings.multi_chat_max_latency_seconds + 1e-9

    # Sequential worst case across concurrency waves also fits the wall budget.
    concurrency = min(settings.multi_chat_panel_concurrency, PANEL_SIZE)
    waves = -(-PANEL_SIZE // concurrency)
    assert per_model * waves <= wall + 1e-9


def test_per_model_timeout_is_capped_by_the_configured_maximum(settings: Settings) -> None:
    _, per_model = MultiModelChat(settings)._panel_budget(PANEL_SIZE)

    assert per_model <= settings.multi_chat_per_model_timeout_seconds


def test_stage_timeout_respects_the_reserve() -> None:
    # 100s left, a 90s cap, but 60s must survive for later stages.
    assert MultiModelChat._stage_timeout_seconds(100.0, 90.0, reserve_seconds=60.0) == 40.0
    # Never returns a nonsensically small or negative timeout.
    assert MultiModelChat._stage_timeout_seconds(10.0, 90.0, reserve_seconds=60.0) == 5.0
    # Backwards compatible with no reserve.
    assert MultiModelChat._stage_timeout_seconds(100.0, 90.0) == 90.0


# --------------------------------------------------------------------------- #
# Panel scheduling
# --------------------------------------------------------------------------- #
def test_panel_runs_concurrently(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """The panel used to be a plain sequential for-loop over five models."""

    _, gauge = _install_llm(monkeypatch, hold_seconds=0.05)

    MultiModelChat(settings).chat("What does scienter require?", history=[])

    assert gauge["peak"] >= 2, "panel calls never overlapped"
    assert gauge["peak"] <= settings.multi_chat_panel_concurrency


def test_panel_answers_stay_in_configured_order(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Concurrency must not reorder the panel: the UI labels answers by position."""

    _install_llm(monkeypatch, hold_seconds=0.02)

    result = MultiModelChat(settings).chat("What does scienter require?", history=[])

    assert [a.model for a in result.model_answers] == [
        settings.multi_chat_gpt_oss_model,
        settings.multi_chat_gemma4_model,
        settings.multi_chat_apertus_model,
        settings.multi_chat_nemotron_model,
        settings.multi_chat_hermes3_model,
    ]


def test_sequential_panel_when_concurrency_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_GROUNDING_ENABLED", "false")
    monkeypatch.setenv("LRG_MULTI_CHAT_PANEL_CONCURRENCY", "1")
    reset_settings()
    serial_settings = Settings()
    _, gauge = _install_llm(monkeypatch, hold_seconds=0.02)

    MultiModelChat(serial_settings).chat("What does scienter require?", history=[])

    assert gauge["peak"] == 1


def test_every_panel_model_is_queried_and_synthesis_still_runs(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """The regression: the panel used to starve synthesis of any time at all."""

    calls, _ = _install_llm(monkeypatch)

    result = MultiModelChat(settings).chat("What does scienter require?", history=[])

    names = [c["name"] for c in calls]
    for expected in ("gpt_oss", "gemma4", "apertus", "nemotron", "hermes3"):
        assert expected in names
    assert "panel_synth" in names
    assert result.final_answer
    assert "time budget exceeded" not in result.final_answer


def test_panel_clients_get_the_derived_timeout_not_the_raw_cap(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls, _ = _install_llm(monkeypatch)

    chat = MultiModelChat(settings)
    _, per_model = chat._panel_budget(PANEL_SIZE)
    chat.chat("What does scienter require?", history=[])

    panel_timeouts = [c["timeout"] for c in calls if c["name"] == "gpt_oss"]
    assert panel_timeouts
    assert panel_timeouts[0] <= per_model + 1e-9


def test_constrained_hardware_profile_still_runs_the_whole_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The .env.example "constrained hardware" profile is the likely deployment.

    Its caps are the most over-subscribed of any profile (a 90s wanted reserve
    against a 120s total), so it is the one most prone to the original failure.
    """

    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_GROUNDING_ENABLED", "false")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "120")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_MODEL_TIMEOUT_SECONDS", "25")
    monkeypatch.setenv("LRG_MULTI_CHAT_SYNTHESIS_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("LRG_MULTI_CHAT_PER_VERIFIER_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LRG_MULTI_CHAT_PANEL_MEMBERS", json.dumps(PANEL_MEMBERS))
    reset_settings()
    constrained = Settings()
    calls, _ = _install_llm(monkeypatch)

    chat = MultiModelChat(constrained)
    wall, per_model = chat._panel_budget(PANEL_SIZE)
    assert per_model >= 5.0
    assert wall + chat._downstream_reserve_seconds() <= 120.0 + 1e-9

    result = chat.chat("What does scienter require?", history=[])

    names = [c["name"] for c in calls]
    for expected in (*PANEL_MEMBERS, "panel_synth"):
        assert expected in names
    assert len(result.verifiers) == 2
    for verifier in result.verifiers:
        assert "time budget was exceeded" not in verifier.verdict
