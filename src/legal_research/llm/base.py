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


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    seed: int | None = None


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
