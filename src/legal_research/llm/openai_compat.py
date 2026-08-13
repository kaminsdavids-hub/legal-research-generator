"""OpenAI-compatible chat client (Ollama, vLLM, llama.cpp, or a hosted provider).

Used only when ``LRG_LLM_MODE=openai``. Every expert role points at a process on
the same host (the DGX Spark) unless one of the opt-in hosted providers -- GLM-5.2
or DeepSeek-V4-Pro, which are too large to serve locally -- has been enabled and
mapped onto that role.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator

import httpx

from .base import ChatMessage, GenerationConfig, LLMClient

# Reasoning models split their output: the answer goes to `content` and the
# chain-of-thought to a sibling field (`reasoning` on Ollama, `reasoning_content`
# on vLLM). Some templates instead inline the thinking in `content` wrapped in
# <think> tags. Strip those defensively -- chain-of-thought must never reach a
# manuscript or a panel answer.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _message_content(data: dict[str, object]) -> str:
    """Extract assistant text from an OpenAI-compatible response.

    Returns ``""`` -- never the string ``"None"`` -- when the model produced no
    visible answer. A reasoning model that spends its entire ``max_tokens`` budget
    thinking returns ``content: null`` (vLLM, GLM, DeepSeek) or ``content: ""``
    (Ollama) with ``finish_reason: "length"``; naively coercing that through
    ``str()`` yields a four-character string that passes every ``if not content``
    guard downstream and lands in the output as literal text.
    """

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        return ""
    return _THINK_TAG_RE.sub("", str(content)).strip()


class OpenAICompatLLM(LLMClient):
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "local-not-secret",
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def _payload(
        self, messages: list[ChatMessage], config: GenerationConfig | None, stream: bool
    ) -> dict[str, object]:
        cfg = config or GenerationConfig()
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "stream": stream,
        }
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        if cfg.keep_alive is not None:
            payload["keep_alive"] = cfg.keep_alive
        return payload

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, config, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
        return _message_content(data)

    def warm_up(self, prompt: str = ".") -> None:
        """Send a tiny completion to force the local model into memory.

        This is a best-effort call: failures are swallowed because a model that
        cannot be reached at startup may still become reachable by the time it is
        actually needed.
        """

        with contextlib.suppress(Exception):
            self.chat(
                [ChatMessage("user", prompt)],
                GenerationConfig(
                    temperature=0.0, top_p=1.0, max_tokens=1, seed=7, keep_alive="-1m"
                ),
            )

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> Iterator[str]:
        with (
            httpx.Client(timeout=self._timeout) as client,
            client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, config, stream=True),
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
