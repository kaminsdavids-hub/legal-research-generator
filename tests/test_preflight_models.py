"""Lineup preflight and per-role endpoints. Offline: no network."""

from __future__ import annotations

import pytest

from evals.preflight_models import ROLES, build_parser, families_ok, main, resolve
from modules.dialectic.roles import detect_family


class _Settings:
    """Only the attributes the preflight reads."""

    def __init__(self, **kwargs: object) -> None:
        self.dialectic_base_url = "http://127.0.0.1:11434/v1"
        self.llm_api_key = "local"
        self.dialectic_thesis_model = "saul:7b-instruct-v1"
        self.dialectic_antithesis_model = "llama3.1:8b"
        self.dialectic_synthesis_model = "gemma3:4b"
        self.dialectic_nli_model = "nemotron-3-nano:4b"
        for role in ROLES:
            setattr(self, f"dialectic_{role}_base_url", "")
            setattr(self, f"dialectic_{role}_api_key", "")
        for key, value in kwargs.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------- #
# Frontier families
# --------------------------------------------------------------------------- #
def test_frontier_models_are_recognised_rather_than_unknown() -> None:
    """Every hosted model mapped to 'unknown' before, and because two unknowns
    collide by design, no frontier lineup could satisfy the guard at all.
    """
    assert detect_family("claude-opus-4-20250101") == "claude"
    assert detect_family("gemini-2.0-pro") == "gemini"
    assert detect_family("mistral-large-latest") == "mistral"
    assert detect_family("qwen2.5-72b-instruct") == "qwen"
    assert detect_family("deepseek-v3") == "deepseek"


def test_a_frontier_lineup_can_now_satisfy_the_distinctness_guard() -> None:
    settings = _Settings(
        dialectic_thesis_model="claude-opus-4",
        dialectic_antithesis_model="gpt-4o",
        dialectic_synthesis_model="gemini-2.0-pro",
        dialectic_nli_model="mistral-large",
    )
    assert families_ok([resolve(settings, r) for r in ROLES]) == []


def test_two_models_from_one_frontier_family_still_collide() -> None:
    """Different sizes of the same model are the correlated errors the guard
    exists to prevent.
    """
    settings = _Settings(
        dialectic_thesis_model="claude-opus-4",
        dialectic_antithesis_model="claude-sonnet-4",
    )
    problems = families_ok([resolve(settings, r) for r in ROLES])
    assert any("claude" in p for p in problems)


def test_two_unrecognised_models_collide_deliberately() -> None:
    """Guessing that two unfamiliar names differ would let correlated models
    debate each other, which is the one thing this guard is for.
    """
    settings = _Settings(
        dialectic_thesis_model="brand-new-model-a",
        dialectic_antithesis_model="brand-new-model-b",
    )
    problems = families_ok([resolve(settings, r) for r in ROLES])
    assert any("unknown" in p for p in problems)


# --------------------------------------------------------------------------- #
# Per-role endpoints
# --------------------------------------------------------------------------- #
def test_a_role_without_an_override_uses_the_shared_endpoint() -> None:
    check = resolve(_Settings(), "thesis")
    assert check.endpoint == "http://127.0.0.1:11434/v1"
    assert check.key_source == "LRG_LLM_API_KEY"


def test_a_role_can_point_at_its_own_endpoint() -> None:
    """One shared base URL made a mixed local/hosted lineup impossible."""
    settings = _Settings(
        dialectic_thesis_model="claude-opus-4",
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
        dialectic_thesis_api_key="configured-elsewhere",
    )
    thesis = resolve(settings, "thesis")
    antithesis = resolve(settings, "antithesis")
    assert thesis.endpoint == "https://api.anthropic.com/v1"
    assert antithesis.endpoint == "http://127.0.0.1:11434/v1", "others stay local"
    assert thesis.key_source == "LRG_DIALECTIC_THESIS_API_KEY"


def test_the_preflight_never_prints_a_key(capsys) -> None:  # type: ignore[no-untyped-def]
    """A credential read back to a terminal ends up in scrollback."""
    secret = "sk-ant-do-not-print-this"
    settings = _Settings(
        dialectic_thesis_model="claude-opus-4",
        dialectic_thesis_api_key=secret,
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
    )
    check = resolve(settings, "thesis")
    print(f"{check.role} {check.model} {check.endpoint} {check.key_source}")
    assert secret not in capsys.readouterr().out


def test_the_service_builder_reads_the_per_role_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The setting is useless if `build_dialectic_chat` ignores it."""
    seen: list[dict[str, str]] = []

    import modules.dialectic.service as service

    class _Fake:
        def __init__(self, **kwargs: str) -> None:
            seen.append(kwargs)
            self.name = kwargs["name"]

    monkeypatch.setattr(
        "legal_research.llm.openai_compat.OpenAICompatLLM", _Fake, raising=False
    )
    monkeypatch.setattr(service, "_build_retriever", lambda s: None)

    settings = _Settings(
        dialectic_thesis_model="claude-opus-4",
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
        llm_mode="openai",
        dialectic_timeout_seconds=30.0,
        courtlistener_token="",
    )
    service.build_dialectic_chat(settings)

    thesis = next(k for k in seen if k["model"] == "claude-opus-4")
    others = [k for k in seen if k["model"] != "claude-opus-4"]
    assert thesis["base_url"] == "https://api.anthropic.com/v1"
    assert all(k["base_url"] == "http://127.0.0.1:11434/v1" for k in others)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def test_a_collision_fails_before_any_model_is_called(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """The whole point is to fail in seconds rather than after paid calls."""
    import evals.preflight_models as mod

    called: list[str] = []
    monkeypatch.setattr(mod, "probe", lambda c, s: called.append(c.role))
    monkeypatch.setattr(
        "legal_research.config.get_settings",
        lambda: _Settings(
            dialectic_thesis_model="claude-opus-4",
            dialectic_antithesis_model="claude-sonnet-4",
        ),
    )
    assert main([]) == 1
    assert not called, "no model may be called once the lineup is invalid"
    assert "family collision" in capsys.readouterr().out


def test_no_probe_makes_no_calls(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import evals.preflight_models as mod

    called: list[str] = []
    monkeypatch.setattr(mod, "probe", lambda c, s: called.append(c.role))
    monkeypatch.setattr("legal_research.config.get_settings", lambda: _Settings())
    assert main(["--no-probe"]) == 0
    assert not called
    assert "skipped endpoint probes" in capsys.readouterr().out


def test_an_unknown_family_gets_told_to_add_a_case_not_loosen_the_guard(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "legal_research.config.get_settings",
        lambda: _Settings(
            dialectic_thesis_model="mystery-a", dialectic_antithesis_model="mystery-b"
        ),
    )
    main([])
    assert "detect_family" in capsys.readouterr().out


def test_the_parser_offers_a_call_free_mode() -> None:
    assert build_parser().parse_args(["--no-probe"]).no_probe is True
