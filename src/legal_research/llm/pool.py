"""The pluggable expert pool.

Experts are addressed by logical role — ``router``, ``saul``, ``finance``,
``writer`` — so the rest of the system never hard-codes a model. In mock mode all
roles resolve to a :class:`MockLLM`; in ``openai`` mode each resolves to a locally
served OpenAI-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from .base import LLMClient
from .mock import MockLLM
from .openai_compat import OpenAICompatLLM

ROUTER = "router"
SAUL = "saul"
FINANCE = "finance"
WRITER = "writer"

EXPERTS = (ROUTER, SAUL, FINANCE, WRITER)


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


def build_expert_pool(settings: Settings | None = None) -> ExpertPool:
    s = settings or get_settings()
    if s.llm_mode == "mock":
        return ExpertPool({role: MockLLM(role) for role in EXPERTS})

    if s.llm_mode == "openai":
        clients: dict[str, LLMClient] = {
            ROUTER: OpenAICompatLLM(ROUTER, s.router_base_url, s.router_model, s.llm_api_key),
            SAUL: OpenAICompatLLM(SAUL, s.saul_base_url, s.saul_model, s.llm_api_key),
            FINANCE: OpenAICompatLLM(FINANCE, s.finance_base_url, s.finance_model, s.llm_api_key),
            WRITER: OpenAICompatLLM(WRITER, s.writer_base_url, s.writer_model, s.llm_api_key),
        }
        return ExpertPool(clients)

    raise ValueError(f"unknown LRG_LLM_MODE {s.llm_mode!r} (expected 'mock' or 'openai')")
