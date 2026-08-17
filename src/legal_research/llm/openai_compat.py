"""OpenAI-compatible chat client (Ollama, vLLM, llama.cpp, or a hosted provider).

Used only when ``LRG_LLM_MODE=openai``. Every expert role points at a process on
the same host (the DGX Spark) unless one of the opt-in hosted providers -- GLM-5.2
or DeepSeek-V4-Pro, which are too large to serve locally -- has been enabled and
mapped onto that role.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Iterator

import httpx

from .base import ChatMessage, GenerationConfig, LLMClient, is_local_endpoint

logger = logging.getLogger(__name__)

# Reasoning models split their output: the answer goes to `content` and the
# chain-of-thought to a sibling field (`reasoning` on Ollama, `reasoning_content`
# on vLLM). Some templates instead inline the thinking in `content` wrapped in
# <think> tags. Strip those defensively -- chain-of-thought must never reach a
# manuscript or a panel answer.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: What every OpenAI-compatible endpoint accepts. Everything else this client
#: sends is either Ollama-specific or optional in the spec, and a hosted
#: provider is entitled to reject an unknown field rather than ignore it.
#:
#: The subset a provider is least likely to reject: ``seed`` is an OpenAI
#: extension, ``keep_alive`` is an Ollama field, ``top_p`` alongside
#: ``temperature`` is discouraged by some providers, and ``response_format`` is
#: optional in the spec.
#:
#: **Opt-in, because the fault it was written for did not exist.** A frontier
#: preflight returned ``400 Bad Request`` from Anthropic's compatibility
#: endpoint while four local roles answered; these fields were the suspect and
#: were dropped by default. Surfacing the response body showed the real cause --
#: an empty credit balance -- and the narrowing probe then refused identically
#: with every one of them removed.
#:
#: Dropping them by default cost something real: ``response_format`` is how the
#: mechanism critic gets server-enforced JSON, and GLM and DeepSeek are both
#: configured in this repo, so a remote critic would have lost it silently to
#: guard against a problem nobody has reproduced. The full payload is therefore
#: the default again, and this is here for an endpoint observed to need it.
_PORTABLE_FIELDS = frozenset({"model", "messages", "max_tokens", "temperature", "stream"})


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


def _raise_for_status(resp: httpx.Response, model: str) -> None:
    """Raise on an error status, with the provider's explanation attached.

    ``raise_for_status`` reports the status line and drops the body, and the
    body is where every OpenAI-compatible provider says *what was wrong*. That
    cost a frontier control run: the thesis role came back
    ``HTTPStatusError: Client error '400 Bad Request'`` and the reply naming the
    offending parameter was discarded, leaving a failure that could not be acted
    on without another paid attempt. A 400 with the field named is one edit; a
    400 without it is a guess.
    """

    if resp.status_code < 400:
        return
    detail = " ".join(resp.text.split())[:400] or "(empty response body)"
    raise httpx.HTTPStatusError(
        f"{resp.status_code} from {resp.request.url} for model {model!r}: {detail}",
        request=resp.request,
        response=resp,
    )


class OpenAICompatLLM(LLMClient):
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "local-not-secret",
        timeout: float = 120.0,
        conservative_remote: bool = False,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        #: Send only :data:`_PORTABLE_FIELDS` to an off-machine endpoint. Off by
        #: default: every role gets the same request wherever it is served, so a
        #: remote run is not quietly a different experiment from a local one.
        #: Turn it on for an endpoint observed to reject a field -- the error
        #: body says which, and the preflight narrows it.
        self._conservative_remote = conservative_remote
        self._warned_conservative = False

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
        if cfg.response_format is not None:
            payload["response_format"] = {"type": cfg.response_format}

        if self._conservative_remote and not is_local_endpoint(self._base_url):
            dropped = sorted(set(payload) - _PORTABLE_FIELDS)
            if dropped and not self._warned_conservative:
                self._warned_conservative = True
                logger.info(
                    "%s is served from %s: sending only the portable request "
                    "fields and dropping %s. Pass conservative_remote=False if "
                    "this provider accepts them.",
                    self._model,
                    self._base_url,
                    ", ".join(dropped),
                )
            payload = {k: v for k, v in payload.items() if k in _PORTABLE_FIELDS}
        return payload

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, config, stream=False),
            )
            _raise_for_status(resp, self._model)
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
            # `read()` first: a streamed error response has no body until it is
            # read, and the body is the part worth reporting.
            if resp.status_code >= 400:
                resp.read()
                _raise_for_status(resp, self._model)
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
