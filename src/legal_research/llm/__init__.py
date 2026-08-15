"""LLM backend abstraction: a small OpenAI-compatible client protocol plus a
deterministic mock used for offline/CI runs."""

from __future__ import annotations

from .base import ChatMessage, DecodingPolicy, GenerationConfig, LLMClient
from .mock import MockLLM
from .openai_compat import OpenAICompatLLM
from .pool import ExpertPool, build_expert_pool

__all__ = [
    "ChatMessage",
    "DecodingPolicy",
    "GenerationConfig",
    "LLMClient",
    "MockLLM",
    "OpenAICompatLLM",
    "ExpertPool",
    "build_expert_pool",
]
