"""Typed application configuration, loaded from environment / ``.env``."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExpertEndpoint(BaseSettings):
    """A single OpenAI-compatible local endpoint for one expert model."""

    base_url: str
    model: str


class Settings(BaseSettings):
    """Global settings. All values are local; no cloud services are contacted."""

    model_config = SettingsConfigDict(
        env_prefix="LRG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000

    # "mock" | "openai"
    llm_mode: str = "openai"
    llm_api_key: str = "local-not-secret"

    # Defaults target a local Ollama OpenAI-compatible endpoint. Override via .env
    # or environment variables per role. No cloud endpoints are used.
    router_base_url: str = "http://127.0.0.1:11434/v1"
    router_model: str = "llama3.1:8b"

    saul_base_url: str = "http://127.0.0.1:11434/v1"
    saul_model: str = "saul:7b-instruct-v1"

    finance_base_url: str = "http://127.0.0.1:11434/v1"
    finance_model: str = "llama3.1:8b"

    writer_base_url: str = "http://127.0.0.1:11434/v1"
    writer_model: str = "qwen2.5-coder:7b"

    # "mock" | "faiss"
    retriever_mode: str = "mock"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    corpus_path: str = "data/corpus/sample_corpus.jsonl"

    # Verifier support test: how it decides a source actually supports a claim.
    #   "lexical"   — content-token recall (default; deterministic, zero deps, CI).
    #   "embedding" — sentence-transformers cosine similarity (needs gpu extra).
    #   "nli"       — cross-encoder entailment P(passage ⊨ claim) (needs gpu extra).
    # The semantic options fall back to lexical if their deps are missing.
    support_scorer: str = "lexical"
    support_threshold: float | None = None
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # "html" | "latex"
    pdf_renderer: str = "html"

    # "bluebook" | "alwd" | "oscola"
    citation_style: str = "bluebook"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings (used by tests)."""

    global _settings
    _settings = None
