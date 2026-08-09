"""Typed application configuration, loaded from environment / ``.env``."""

from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExpertEndpoint(BaseSettings):
    """A single OpenAI-compatible local endpoint for one expert model."""

    base_url: str
    model: str


class Settings(BaseSettings):
    """Global settings.

    Every expert endpoint is local by default; the only exceptions are the
    opt-in hosted providers at the bottom of this class, which are disabled
    unless explicitly enabled and credentialed.
    """

    model_config = SettingsConfigDict(
        env_prefix="LRG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # The hosted-provider fields carry explicit aliases; keep their plain
        # field names usable for direct construction.
        populate_by_name=True,
    )

    host: str = "127.0.0.1"
    port: int = 8000
    debug_endpoints_enabled: bool = False

    # "mock" | "openai"
    llm_mode: str = "openai"
    llm_api_key: str = "local-not-secret"
    # Local, CPU/Metal-served reasoning models (e.g. Nemotron Nano's chain-of-thought)
    # can take well over a minute per call; keep this generous.
    llm_timeout_seconds: float = 300.0

    # Warm-up each local Ollama model on startup with a tiny completion. This pays the
    # first-load cost once and, combined with OLLAMA_MAX_LOADED_MODELS, keeps models
    # resident across the pipeline/multi-chat lifecycle instead of reloading per stage.
    llm_warmup_enabled: bool = False
    llm_warmup_prompt: str = "Hello."

    # Mirrors OLLAMA_MAX_LOADED_MODELS for the local Ollama server. Setting this in the
    # backend process is informational; the value must also be exported in the Ollama
    # server's environment to keep more than the default number of models resident.
    #
    # Two server-side settings dominate Model Jury latency, and neither can be
    # controlled from here -- the OpenAI-compatible endpoint has no field for
    # either, so they must be set in the Ollama *service* environment:
    #
    #   OLLAMA_MAX_LOADED_MODELS  Defaults to 3. One jury query needs 8 model
    #                             activations (5 panel + synthesis + 2 verifiers),
    #                             so at the default every call evicts and reloads
    #                             models, and panel members time out.
    #   OLLAMA_CONTEXT_LENGTH     Defaults to 4k/32k/256k *chosen by available
    #                             memory*. On a 128 GB Spark it picks the top tier,
    #                             so each model allocates a huge KV cache:
    #                             hermes3:8b measured at 30.4 GB resident by
    #                             default versus 6.0 GB at 8192. Capping this is
    #                             what makes co-residency possible at all.
    ollama_max_loaded_models: int | None = Field(
        default=None,
        validation_alias=AliasChoices("LRG_OLLAMA_MAX_LOADED_MODELS", "OLLAMA_MAX_LOADED_MODELS"),
    )

    # Defaults target a local Ollama OpenAI-compatible endpoint. Override via .env
    # or environment variables per role. No cloud endpoints are used.
    saul_base_url: str = "http://127.0.0.1:11434/v1"
    saul_model: str = "saul:7b-instruct-v1"

    writer_base_url: str = "http://127.0.0.1:11434/v1"
    writer_model: str = "llama3.1:8b"

    # Gemma 3 — the general-purpose analyst and the router's tie-break classifier.
    gemma_base_url: str = "http://127.0.0.1:11434/v1"
    gemma_model: str = "gemma3:4b"

    # Hermes role — reasoning/ideation engine. Backed by NVIDIA Nemotron 3 Nano
    # (strong reasoning/tool-use, small footprint) in this workspace.
    hermes_base_url: str = "http://127.0.0.1:11434/v1"
    hermes_model: str = "nemotron-3-nano:4b"

    # Hermes-3 role — final grammar/fluency pass in the post-draft chain.
    hermes3_base_url: str = "http://127.0.0.1:11434/v1"
    hermes3_model: str = "hermes3:8b"

    # Secondary chat module: multi-model response panel + verifier pair.
    multi_chat_base_url: str = "http://127.0.0.1:11434/v1"
    multi_chat_gpt_oss_model: str = "gpt-oss:20b"
    multi_chat_gemma4_model: str = "gemma4:latest"
    multi_chat_apertus_model: str = "apertus:latest"
    multi_chat_nemotron_model: str = "nemotron-3-nano:4b"
    multi_chat_hermes3_model: str = "hermes3:8b"
    multi_chat_verifier_gemma3_model: str = "gemma3:4b"
    multi_chat_verifier_saul_model: str = "saul:7b-instruct-v1"
    multi_chat_max_latency_seconds: float = 260.0
    # Upper bound per panel model. The effective per-model timeout is derived from
    # the panel's share of the latency budget (see MultiModelChat._panel_budget)
    # and is capped by this value, so raising it never starves synthesis.
    #
    # 30s was too tight for the reasoning models on the panel: gpt-oss:20b needs
    # ~20s warm and alone for a 1.5k-token analysis, before any cold load of its
    # 13 GB of weights. A timeout there costs the panel two calls, because the
    # timeout path retries.
    multi_chat_per_model_timeout_seconds: float = 75.0
    # Panel members queried at once. Ollama serializes requests that need the same
    # model, so this mostly buys overlap across *different* models -- and each
    # concurrent model must be co-resident, so raising it on a memory-tight host
    # trades wall time for swapping. 2 is safe with the default 5-model panel;
    # raise it once the host has headroom.
    multi_chat_panel_concurrency: int = 2
    multi_chat_synthesis_timeout_seconds: float = 45.0
    multi_chat_per_verifier_timeout_seconds: float = 18.0
    multi_chat_rescue_model: str = "gpt-oss:20b"
    multi_chat_rescue_timeout_seconds: float = 45.0

    # Citation grounding for the chat panel. When enabled, panel prompts receive a
    # retrieved authority packet and every asserted citation is audited against the
    # corpus; unverifiable authorities trigger a repair pass and a reader warning.
    multi_chat_grounding_enabled: bool = True
    multi_chat_grounding_k: int = 6
    # None => inherit the support scorer's own threshold.
    multi_chat_grounding_min_support: float | None = None
    multi_chat_citation_repair_timeout_seconds: float = 40.0

    # Dialectic chat module: thesis / antithesis / synthesis on three DISTINCT
    # base model families, plus a fourth model for the NLI contradiction pass.
    # Same-family debaters have correlated errors and produce agreement dressed
    # as debate, so the engine refuses to start when two roles share a family.
    dialectic_base_url: str = "http://127.0.0.1:11434/v1"
    # NOT gpt-oss:20b: it is reasoning-only and returns an empty content channel
    # through the OpenAI-compatible endpoint, so every turn fails to parse.
    #
    # Thesis is saul (a legal-domain model) rather than hermes3, which now serves
    # as the eval judge. Of the locally available models, hermes3 was the only one
    # that could separate weak work from strong against the scoring rubric --
    # saul rated a response of pure failure placeholders 8.0/10. The judge is a
    # fifth role and must not share a family with any debater, so the two swapped.
    # Eval and production are kept on the same debate configuration deliberately:
    # an eval that scores a lineup you do not ship measures the wrong system.
    dialectic_thesis_model: str = "saul:7b-instruct-v1"
    dialectic_antithesis_model: str = "llama3.1:8b"
    dialectic_synthesis_model: str = "gemma3:4b"
    dialectic_nli_model: str = "nemotron-3-nano:4b"
    dialectic_timeout_seconds: float = 120.0
    # CourtListener v4 citation-lookup. Slots stay NOT_FOUND without a token;
    # the module never marks a slot VERIFIED on an unverified path.
    courtlistener_token: str = Field(
        default="",
        validation_alias=AliasChoices("LRG_COURTLISTENER_TOKEN", "COURTLISTENER_TOKEN"),
    )

    # Post-draft grammar correction chain. Roles run in order and are expected to
    # preserve citation token IDs exactly.
    grammar_chain_enabled: bool = True
    grammar_chain_roles: str = "hermes,gemma,hermes3"
    grammar_chain_fail_open: bool = True

    # ------------------------------------------------------------------ #
    # Hosted (cloud) OpenAI-compatible providers -- opt in, off by default.
    #
    # GLM-5.2 (753B) and DeepSeek-V4-Pro (1.6T) are too large to serve from the
    # 128 GB Spark, so they are reachable only as remote APIs. Enabling either
    # one sends prompt text off the local machine, which is why they are gated
    # behind an explicit flag plus a non-empty key.
    #
    # These read the same unprefixed env names as the patent-generator stack
    # (``GLM_API_KEY``, ``DEEPSEEK_API_KEY``, ...) so one shell environment
    # configures both, while still honouring the ``LRG_``-prefixed form.
    glm_enable: bool = Field(
        default=False, validation_alias=AliasChoices("LRG_GLM_ENABLE", "GLM_ENABLE")
    )
    glm_base_url: str = Field(
        default="https://api.z.ai/api/paas/v4",
        validation_alias=AliasChoices("LRG_GLM_BASE_URL", "GLM_BASE_URL"),
    )
    glm_model: str = Field(
        default="glm-5.2", validation_alias=AliasChoices("LRG_GLM_MODEL", "GLM_MODEL")
    )
    glm_api_key: str = Field(
        default="", validation_alias=AliasChoices("LRG_GLM_API_KEY", "GLM_API_KEY")
    )

    deepseek_enable: bool = Field(
        default=False,
        validation_alias=AliasChoices("LRG_DEEPSEEK_ENABLE", "DEEPSEEK_ENABLE"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("LRG_DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    deepseek_model: str = Field(
        default="deepseek-v4-pro",
        validation_alias=AliasChoices("LRG_DEEPSEEK_MODEL", "DEEPSEEK_MODEL"),
    )
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LRG_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    )

    # Optional reassignment of existing expert roles onto a hosted provider,
    # e.g. "saul=deepseek,writer=glm,hermes=deepseek". Roles left out keep their
    # local endpoint. Entries naming a disabled provider are ignored.
    remote_role_map: str = Field(
        default="", validation_alias=AliasChoices("LRG_REMOTE_ROLE_MAP", "REMOTE_ROLE_MAP")
    )

    # "mock" | "faiss"
    retriever_mode: str = "faiss"
    # Default dense retrieval model for the Spark stack.
    embed_model: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    corpus_path: str = "data/corpus/sample_corpus.jsonl"

    # Verifier support test: how it decides a source actually supports a claim.
    #   "lexical"   — content-token recall (deterministic, zero deps, CI).
    #   "embedding" — sentence-transformers cosine similarity (needs gpu extra).
    #   "nli"       — cross-encoder entailment P(passage ⊨ claim) (needs gpu extra).
    # The semantic options fall back to lexical if their deps are missing.
    support_scorer: str = "nli"
    support_threshold: float | None = None
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # "html" | "latex"
    pdf_renderer: str = "html"

    # "bluebook" | "alwd" | "oscola"
    citation_style: str = "bluebook"

    # Long-form manuscript drafting targets.
    manuscript_target_min_words: int = 7500
    manuscript_target_max_words: int = 10000
    draft_paragraph_target_words: int = 220
    draft_max_paragraphs_per_section: int = 30

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://legal-research-generator-app.netlify.app",
            "https://legal-research-generator-app-932.netlify.app",
        ]
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    global _settings
    if _settings is None:
        _settings = Settings()
        if _settings.ollama_max_loaded_models is not None:
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(_settings.ollama_max_loaded_models)
    return _settings


def reset_settings() -> None:
    """Clear the cached settings (used by tests)."""

    global _settings
    _settings = None
