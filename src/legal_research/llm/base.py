"""Core LLM types shared by every backend implementation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


class DecodingPolicy(str, Enum):
    """Decoding presets the router chooses per task.

    * ``COLD``     — deterministic, low temperature. Used for legal doctrine where
      citations are forced through RAG and invention is unacceptable.
    * ``BALANCED`` — moderate temperature for synthesis / prose.
    * ``HOT``      — high temperature for divergent brainstorming / ideation.
    """

    COLD = "cold"
    BALANCED = "balanced"
    HOT = "hot"

    @property
    def config(self) -> GenerationConfig:
        if self is DecodingPolicy.COLD:
            return GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=1024, seed=7)
        if self is DecodingPolicy.HOT:
            return GenerationConfig(temperature=1.0, top_p=0.98, max_tokens=768, seed=None)
        return GenerationConfig(temperature=0.6, top_p=0.95, max_tokens=1536, seed=None)


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


def is_local_endpoint(base_url: str) -> bool:
    """Whether the endpoint is served from this machine.

    The one implementation. There were two — ``llm.pool._is_local_endpoint`` and
    ``modules.dialectic.service.is_local_endpoint`` — and the answer now decides
    more than a log line: it decides which request fields are sent, so two
    copies could disagree about whether a run is local and send a payload the
    provider rejects.
    """

    url = (base_url or "").lower()
    return any(host in url for host in ("127.0.0.1", "localhost", "::1"))


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    seed: int | None = None
    keep_alive: str | None = None
    #: Ask the server to constrain decoding to valid JSON ("json_object"). Only
    #: callers that parse the reply as JSON should set it.
    #:
    #: Prompting for JSON is not the same as getting it. A critic asked to quote
    #: legal prose emits the quote's own punctuation into a string it never
    #: escapes, and the reply fails to parse on the one section whose text
    #: contained a quotation mark. Constrained decoding removes that failure at
    #: the source; backends that do not support the field ignore it, so the
    #: tolerant parsing on the caller's side stays necessary either way.
    response_format: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal OpenAI-compatible chat interface implemented by every backend."""

    name: str

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> str: ...

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> Iterator[str]: ...
