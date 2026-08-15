"""Per-role endpoints for the dialectic lineup, and the control arm they enable.

The rule under test is that a role served from somewhere other than this machine
must carry its own credential. Falling back to the local placeholder key would
either fail at the provider with an error that reads like a network problem, or
succeed against something that ignores the header — and then report a "frontier"
number that no frontier model produced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from legal_research.config import Settings
from modules.dialectic.roles import detect_family
from modules.dialectic.service import is_local_endpoint, resolve_api_key, resolve_role

ROOT = Path(__file__).resolve().parents[1]


def _settings(**overrides) -> Settings:
    return Settings(llm_mode="mock", retriever_mode="mock", **overrides)


def test_roles_fall_back_to_the_shared_endpoint() -> None:
    """An all-local lineup must resolve exactly as it did before overrides existed."""

    s = _settings(dialectic_base_url="http://127.0.0.1:11434/v1")
    for role in ("thesis", "antithesis", "synthesis", "nli"):
        spec = resolve_role(s, role)
        assert spec.base_url == "http://127.0.0.1:11434/v1"
        assert spec.model
        assert spec.family != "unknown"


def test_a_per_role_override_wins() -> None:
    s = _settings(
        dialectic_base_url="http://127.0.0.1:11434/v1",
        dialectic_thesis_model="claude-sonnet-4-5",
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
    )
    thesis = resolve_role(s, "thesis")
    antithesis = resolve_role(s, "antithesis")

    assert thesis.base_url == "https://api.anthropic.com/v1"
    assert thesis.family == "claude"
    assert antithesis.base_url == "http://127.0.0.1:11434/v1"


def test_a_remote_role_without_its_own_key_is_refused() -> None:
    s = _settings(
        dialectic_thesis_model="claude-sonnet-4-5",
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
    )
    with pytest.raises(ValueError, match="LRG_DIALECTIC_THESIS_API_KEY"):
        resolve_api_key(s, resolve_role(s, "thesis"))


def test_a_remote_role_with_a_key_resolves_to_it() -> None:
    s = _settings(
        dialectic_thesis_model="claude-sonnet-4-5",
        dialectic_thesis_base_url="https://api.anthropic.com/v1",
        dialectic_thesis_api_key="sk-test",
    )
    assert resolve_api_key(s, resolve_role(s, "thesis")) == "sk-test"


def test_a_local_role_uses_the_shared_key() -> None:
    s = _settings(llm_api_key="local-not-secret")
    assert resolve_api_key(s, resolve_role(s, "nli")) == "local-not-secret"


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:11434/v1", True),
        ("https://api.anthropic.com/v1", False),
        ("", False),
    ],
)
def test_endpoint_locality(url: str, local: bool) -> None:
    assert is_local_endpoint(url) is local


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("claude-sonnet-4-5", "claude"),
        ("deepseek-v4-pro", "deepseek"),
        ("glm-5.2", "glm"),
        # A distill carries the base weights' failure modes, so it must resolve
        # to the base family or two "independent" roles could share one.
        ("deepseek-r1-distill-llama-70b", "llama"),
        ("saul:7b-instruct-v1", "saul"),
    ],
)
def test_family_detection(model: str, family: str) -> None:
    assert detect_family(model) == family


# --------------------------------------------------------------------------- #
# Preflight — a lineup that cannot be valid should cost nothing to reject
# --------------------------------------------------------------------------- #
def _preflight():
    spec = importlib.util.spec_from_file_location(
        "preflight_models", ROOT / "scripts" / "preflight_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["preflight_models"] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_accepts_a_frontier_thesis_with_no_probe(capsys) -> None:
    preflight = _preflight()
    argv = [
        "preflight_models.py",
        "--no-probe",
        "--thesis",
        "claude-sonnet-4-5",
        "--thesis-base-url",
        "https://api.anthropic.com/v1",
        "--thesis-api-key-env",
        "LRG_DIALECTIC_THESIS_API_KEY",
    ]
    sys.argv, saved = argv, sys.argv
    try:
        assert preflight.main() == 0
    finally:
        sys.argv = saved
    out = capsys.readouterr().out
    assert "REMOTE https://api.anthropic.com/v1" in out
    assert "no calls made" in out


def test_preflight_rejects_a_shared_family() -> None:
    preflight = _preflight()
    complaints = preflight.check_families(
        preflight._specs(
            preflight.build_parser().parse_args(["--synthesis", "llama3.1:70b"])
        )
    )
    assert len(complaints) == 1
    assert "shares family" in complaints[0]


def test_preflight_reports_an_unrecognised_family_rather_than_passing_it() -> None:
    preflight = _preflight()
    complaints = preflight.check_families(
        preflight._specs(preflight.build_parser().parse_args(["--thesis", "some-new-model:9b"]))
    )
    assert len(complaints) == 1
    assert "unrecognised" in complaints[0]


# --------------------------------------------------------------------------- #
# A 400 must name the field, not send the operator back for another round trip
# --------------------------------------------------------------------------- #
def test_preflight_narrows_a_rejected_payload_to_the_offending_field() -> None:
    """A live frontier preflight returned "400 Bad Request" with the body
    discarded, so the run could not proceed without a human guessing which
    decoding parameter the provider refused."""

    preflight = _preflight()

    class _RejectsSeed:
        """Accepts the request only once `seed` is absent."""

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, config=None):  # noqa: ANN001 - test double
            self.calls += 1
            if getattr(config, "seed", None) is not None:
                raise RuntimeError("400 Bad Request: unsupported parameter 'seed'")
            return "ready"

    client = _RejectsSeed()
    culprit = preflight._narrow_payload(client, 30.0)

    assert "seed" in culprit
    assert "accepted once" in culprit
    assert client.calls == 1  # stopped at the first payload that worked


def test_preflight_says_so_when_no_decoding_field_is_to_blame() -> None:
    """And names no cause of its own. An earlier version concluded "the model
    name or endpoint path is wrong"; the real 400 was an empty credit balance,
    so a confident wrong diagnosis printed directly under the provider's
    correct one."""

    class _AlwaysRefuses:
        def chat(self, messages, config=None):  # noqa: ANN001 - test double
            raise RuntimeError(
                "400: Your credit balance is too low to access the Anthropic API."
            )

    culprit = _preflight()._narrow_payload(_AlwaysRefuses(), 30.0)

    assert "no decoding parameter is to blame" in culprit
    assert "model name" not in culprit
    assert "endpoint path" not in culprit
