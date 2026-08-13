"""The pluggable expert pool.

Experts are addressed by logical role — ``saul``, ``writer``, ``gemma``,
``hermes`` and ``hermes3`` — so the rest of the system never hard-codes a model.
In mock mode all roles resolve to a :class:`MockLLM`; in ``openai`` mode each
resolves to a locally served OpenAI-compatible endpoint.

``gemma`` (Gemma 3) is the general-purpose analyst and the router's tie-break
classifier; ``hermes`` (NVIDIA Nemotron 3 Nano) is the reasoning/ideation engine;
``hermes3`` runs the final grammar/fluency pass in the post-draft chain.

Two further roles -- ``glm`` (GLM-5.2) and ``deepseek`` (DeepSeek-V4-Pro) -- are
hosted APIs rather than local servers, since neither fits on the Spark. They
appear in the pool only when enabled and credentialed, and can additionally take
over any local role via ``LRG_REMOTE_ROLE_MAP``.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass

from ..config import Settings, get_settings
from .base import LLMClient
from .mock import MockLLM
from .openai_compat import OpenAICompatLLM

SAUL = "saul"
WRITER = "writer"
GEMMA = "gemma"
HERMES = "hermes"
HERMES3 = "hermes3"
GLM = "glm"
DEEPSEEK = "deepseek"

EXPERTS = (SAUL, WRITER, GEMMA, HERMES, HERMES3)
REMOTE_EXPERTS = (GLM, DEEPSEEK)


@dataclass
class ExpertPool:
    """Holds one :class:`LLMClient` per logical expert role."""

    clients: dict[str, LLMClient]

    def get(self, role: str) -> LLMClient:
        try:
            return self.clients[role]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"unknown expert role {role!r}; known roles: {sorted(self.clients)}"
            ) from exc

    @property
    def roles(self) -> list[str]:
        return sorted(self.clients)


def _is_local_endpoint(base_url: str) -> bool:
    """Whether the endpoint is the same machine as the backend."""

    url = base_url.lower()
    return any(host in url for host in ("127.0.0.1", "localhost", "::1"))


def build_expert_pool(settings: Settings | None = None) -> ExpertPool:
    s = settings or get_settings()
    if s.ollama_max_loaded_models is not None:
        os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(s.ollama_max_loaded_models)

    if s.llm_mode == "mock":
        return ExpertPool({role: MockLLM(role) for role in EXPERTS})

    if s.llm_mode == "openai":
        clients: dict[str, LLMClient] = {
            SAUL: OpenAICompatLLM(
                SAUL, s.saul_base_url, s.saul_model, s.llm_api_key, s.llm_timeout_seconds
            ),
            WRITER: OpenAICompatLLM(
                WRITER, s.writer_base_url, s.writer_model, s.llm_api_key, s.llm_timeout_seconds
            ),
            GEMMA: OpenAICompatLLM(
                GEMMA, s.gemma_base_url, s.gemma_model, s.llm_api_key, s.llm_timeout_seconds
            ),
            HERMES: OpenAICompatLLM(
                HERMES, s.hermes_base_url, s.hermes_model, s.llm_api_key, s.llm_timeout_seconds
            ),
            HERMES3: OpenAICompatLLM(
                HERMES3,
                s.hermes3_base_url,
                s.hermes3_model,
                s.llm_api_key,
                s.llm_timeout_seconds,
            ),
        }
        clients.update(_build_remote_clients(s))
        _apply_remote_role_map(clients, s)
        pool = ExpertPool(clients)

        if s.llm_warmup_enabled:

            def _warm_pool() -> None:
                for client in pool.clients.values():
                    if isinstance(client, OpenAICompatLLM) and _is_local_endpoint(
                        client._base_url
                    ):
                        with contextlib.suppress(Exception):
                            client.warm_up(s.llm_warmup_prompt)

            threading.Thread(target=_warm_pool, daemon=True).start()

        return pool

    raise ValueError(f"unknown LRG_LLM_MODE {s.llm_mode!r} (expected 'mock' or 'openai')")


def _build_remote_clients(s: Settings) -> dict[str, LLMClient]:
    """Clients for each hosted provider that is both enabled and credentialed.

    A blank key disables the provider instead of raising: a half-filled ``.env``
    should degrade to the local-only stack, not break startup.
    """

    remote: dict[str, LLMClient] = {}
    if s.glm_enable and s.glm_api_key.strip():
        remote[GLM] = OpenAICompatLLM(
            GLM, s.glm_base_url, s.glm_model, s.glm_api_key, s.llm_timeout_seconds
        )
    if s.deepseek_enable and s.deepseek_api_key.strip():
        remote[DEEPSEEK] = OpenAICompatLLM(
            DEEPSEEK,
            s.deepseek_base_url,
            s.deepseek_model,
            s.deepseek_api_key,
            s.llm_timeout_seconds,
        )
    return remote


def _apply_remote_role_map(clients: dict[str, LLMClient], s: Settings) -> None:
    """Point local roles at a hosted provider per ``remote_role_map``.

    Unknown roles and providers that are not active are skipped, so the mapping
    can be left in place while a provider is toggled off.
    """

    for entry in s.remote_role_map.split(","):
        role, _, provider = entry.partition("=")
        role, provider = role.strip(), provider.strip()
        if role in EXPERTS and provider in REMOTE_EXPERTS and provider in clients:
            clients[role] = clients[provider]
