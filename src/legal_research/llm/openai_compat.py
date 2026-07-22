"""OpenAI-compatible client for locally served models (Ollama, vLLM, llama.cpp).

Used only when ``LRG_LLM_MODE=openai``. Never contacts a cloud provider — the base
URL points at a process running on the same host (the DGX Spark).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from .base import ChatMessage, GenerationConfig, LLMClient


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
        return payload

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> str:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, config, stream=False),
            )
            resp.raise_for_status()
            data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> Iterator[str]:
        with httpx.Client(timeout=self._timeout) as client, client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=self._payload(messages, config, stream=True),
        ) as resp:
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
