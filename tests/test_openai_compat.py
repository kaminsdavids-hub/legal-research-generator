"""Response parsing for the OpenAI-compatible client.

Reasoning models make the response shape less forgiving than it looks: the answer
and the chain-of-thought arrive in different fields, and a model that spends its
whole token budget thinking returns no answer at all. Both must degrade to an
empty string, because every caller's emptiness check depends on it.
"""

from __future__ import annotations

import httpx
import pytest

from legal_research.llm.base import ChatMessage, GenerationConfig
from legal_research.llm.openai_compat import OpenAICompatLLM, _message_content


def _response(message: dict, finish_reason: str = "stop") -> dict:
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


# --------------------------------------------------------------------------- #
# The "None" bug
# --------------------------------------------------------------------------- #
def test_null_content_becomes_empty_string_not_the_word_none() -> None:
    """A reasoning-only response must not surface as the literal text "None".

    vLLM and the hosted providers return ``content: null`` when the model emits
    only reasoning. Coercing that with ``str()`` yields "None", which is truthy,
    passes every ``if not content`` guard, and lands in the manuscript as text.
    """

    assert _message_content(_response({"role": "assistant", "content": None})) == ""


def test_reasoning_only_response_with_length_finish_is_empty() -> None:
    payload = _response(
        {"role": "assistant", "content": "", "reasoning": "thinking at length..."},
        finish_reason="length",
    )

    assert _message_content(payload) == ""


def test_missing_content_key_is_empty() -> None:
    assert _message_content(_response({"role": "assistant"})) == ""


def test_malformed_payloads_are_empty_not_exceptions() -> None:
    assert _message_content({}) == ""
    assert _message_content({"choices": []}) == ""
    assert _message_content({"choices": "nope"}) == ""
    assert _message_content({"choices": [None]}) == ""
    assert _message_content({"choices": [{"message": "nope"}]}) == ""


# --------------------------------------------------------------------------- #
# Chain-of-thought must never reach the caller
# --------------------------------------------------------------------------- #
def test_inline_think_block_is_stripped() -> None:
    payload = _response(
        {
            "role": "assistant",
            "content": "<think>The user wants doctrine. Let me plan.</think>The rule requires scienter.",
        }
    )

    assert _message_content(payload) == "The rule requires scienter."


def test_multiline_and_multiple_think_blocks_are_stripped() -> None:
    payload = _response(
        {
            "role": "assistant",
            "content": "<think>\nline one\nline two\n</think>Answer A.<THINK>more</THINK> Answer B.",
        }
    )

    result = _message_content(payload)
    assert "line one" not in result
    assert "more" not in result
    assert "Answer A." in result
    assert "Answer B." in result


def test_sibling_reasoning_field_is_never_returned() -> None:
    payload = _response(
        {"role": "assistant", "content": "The rule requires scienter.", "reasoning": "SECRET PLAN"}
    )

    assert _message_content(payload) == "The rule requires scienter."


def test_normal_content_is_returned_unchanged() -> None:
    payload = _response({"role": "assistant", "content": "  The rule requires scienter.  "})

    assert _message_content(payload) == "The rule requires scienter."


# --------------------------------------------------------------------------- #
# End to end through chat()
# --------------------------------------------------------------------------- #
def test_chat_returns_empty_string_for_reasoning_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                {"role": "assistant", "content": None, "reasoning": "..."},
                finish_reason="length",
            ),
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):  # noqa: ANN001, ANN202
        kwargs["transport"] = transport
        kwargs.pop("timeout", None)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("legal_research.llm.openai_compat.httpx.Client", _factory)

    client = OpenAICompatLLM("panel", "http://llm.test/v1", "some-model")
    out = client.chat([ChatMessage("user", "hi")], GenerationConfig(max_tokens=200))

    assert out == ""


def test_chat_strips_think_block_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                {"role": "assistant", "content": "<think>plan</think>Scienter is required."}
            ),
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):  # noqa: ANN001, ANN202
        kwargs["transport"] = transport
        kwargs.pop("timeout", None)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("legal_research.llm.openai_compat.httpx.Client", _factory)

    client = OpenAICompatLLM("panel", "http://llm.test/v1", "some-model")

    assert client.chat([ChatMessage("user", "hi")]) == "Scienter is required."


def test_an_error_status_carries_the_provider_explanation() -> None:
    """A frontier preflight failed with "400 Bad Request" and the reply naming
    the offending parameter was discarded, leaving nothing to act on."""

    import httpx
    import pytest

    from legal_research.llm.base import ChatMessage
    from legal_research.llm.openai_compat import OpenAICompatLLM

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "temperature and top_p may not both be set",
                            "param": "top_p", "type": "invalid_request_error"}},
        )

    client = OpenAICompatLLM(
        name="frontier", base_url="https://api.example.com/v1", model="some-model:1"
    )
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    try:
        httpx.Client = lambda **kw: real_client(transport=transport, **kw)  # type: ignore[misc]
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            client.chat([ChatMessage("user", "hi")])
    finally:
        httpx.Client = real_client  # type: ignore[misc]

    message = str(excinfo.value)
    assert "400" in message
    assert "temperature and top_p may not both be set" in message
    assert "some-model:1" in message


# --------------------------------------------------------------------------- #
# Remote endpoints get only the fields every provider accepts
# --------------------------------------------------------------------------- #
def _payload_for(base_url: str, **kwargs):
    from legal_research.llm.base import ChatMessage, DecodingPolicy
    from legal_research.llm.openai_compat import OpenAICompatLLM

    client = OpenAICompatLLM(name="r", base_url=base_url, model="m:1", **kwargs)
    return client._payload([ChatMessage("user", "hi")], DecodingPolicy.COLD.config, stream=False)


def test_a_remote_payload_is_full_by_default() -> None:
    """A frontier preflight's 400 was suspected to be these fields and was not
    — it was an empty credit balance. Dropping them by default would cost the
    mechanism critic server-enforced JSON on GLM and DeepSeek, both configured
    here, to guard against a fault nobody has reproduced."""

    payload = _payload_for("https://api.anthropic.com/v1")

    assert payload["seed"] == 7
    assert payload["top_p"] == 1.0
    assert payload["temperature"] == 0.0


def test_a_remote_payload_can_be_narrowed_on_request() -> None:
    """For an endpoint observed to reject a field."""

    payload = _payload_for("https://api.anthropic.com/v1", conservative_remote=True)

    assert set(payload) == {"model", "messages", "max_tokens", "temperature", "stream"}
    for field in ("seed", "top_p", "keep_alive", "response_format"):
        assert field not in payload


def test_a_local_payload_is_unchanged() -> None:
    """Ollama takes the full set, and seed is what makes COLD reproducible."""

    payload = _payload_for("http://127.0.0.1:11434/v1")

    assert payload["seed"] == 7
    assert payload["top_p"] == 1.0
    assert payload["temperature"] == 0.0


def test_local_and_remote_send_the_same_request() -> None:
    """A role must not become a different experiment by moving hosts."""

    local = _payload_for("http://127.0.0.1:11434/v1")
    remote = _payload_for("https://api.anthropic.com/v1")

    assert {k: v for k, v in local.items() if k != "model"} == {
        k: v for k, v in remote.items() if k != "model"
    }


def test_response_format_survives_remotely_unless_narrowing_is_asked_for() -> None:
    from dataclasses import replace

    from legal_research.llm.base import ChatMessage, DecodingPolicy
    from legal_research.llm.openai_compat import OpenAICompatLLM

    config = replace(DecodingPolicy.COLD.config, response_format="json_object")
    message = [ChatMessage("user", "hi")]

    local = OpenAICompatLLM(name="l", base_url="http://localhost:11434/v1", model="m:1")
    remote = OpenAICompatLLM(name="r", base_url="https://api.example.com/v1", model="m:1")

    narrowed = OpenAICompatLLM(
        name="n", base_url="https://api.example.com/v1", model="m:1", conservative_remote=True
    )
    expected = {"type": "json_object"}
    assert local._payload(message, config, stream=False)["response_format"] == expected
    assert remote._payload(message, config, stream=False)["response_format"] == expected
    assert "response_format" not in narrowed._payload(message, config, stream=False)


def test_one_locality_rule_serves_every_caller() -> None:
    from legal_research.llm.base import is_local_endpoint
    from legal_research.llm.pool import _is_local_endpoint
    from modules.dialectic.service import is_local_endpoint as service_rule

    for url in ("http://127.0.0.1:11434/v1", "https://api.anthropic.com/v1", ""):
        assert is_local_endpoint(url) == _is_local_endpoint(url) == service_rule(url)
