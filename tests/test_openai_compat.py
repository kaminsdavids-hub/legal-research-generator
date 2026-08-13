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
