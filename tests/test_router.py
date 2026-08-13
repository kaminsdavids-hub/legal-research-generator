"""Router classification (spec §1)."""

from __future__ import annotations

from collections.abc import Iterator

from legal_research.agents.finance_analyst import FinanceAnalyst
from legal_research.agents.ideator import Ideator
from legal_research.llm.base import ChatMessage, DecodingPolicy, GenerationConfig
from legal_research.llm.mock import MockLLM
from legal_research.llm.pool import (
    EXPERTS,
    GEMMA,
    HERMES,
    HERMES3,
    SAUL,
    WRITER,
    ExpertPool,
    build_expert_pool,
)
from legal_research.router import RequestKind, Router


def _router() -> Router:
    return Router(build_expert_pool())


class _SpyLLM:
    """Records how many times it is asked to break a tie and returns a fixed label."""

    def __init__(self, name: str, label: str = "finance") -> None:
        self.name = name
        self._label = label
        self.calls = 0

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> str:
        self.calls += 1
        return self._label

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> Iterator[str]:
        yield self.chat(messages, config)


def test_legal_routes_to_saul_cold_with_forced_citation() -> None:
    decision = _router().route("Analyze the holding and statutory basis of the doctrine.")
    assert decision.kind is RequestKind.LEGAL
    assert decision.expert == SAUL
    assert decision.policy is DecodingPolicy.COLD
    assert decision.forced_citation is True


def test_finance_routes_to_general_gemma() -> None:
    decision = _router().route("Explain Basel capital and bank leverage requirements.")
    assert decision.kind is RequestKind.FINANCE
    assert decision.expert == GEMMA


def test_brainstorm_routes_hot() -> None:
    decision = _router().route("Let's brainstorm a novel angle and idea for the thesis.")
    assert decision.kind is RequestKind.BRAINSTORM
    assert decision.policy is DecodingPolicy.HOT


def test_default_is_synthesis() -> None:
    decision = _router().route("Please tie these threads together into a coherent whole.")
    assert decision.kind is RequestKind.SYNTHESIS
    assert decision.expert == WRITER


def test_pool_exposes_gemma_and_hermes_roles() -> None:
    assert GEMMA in EXPERTS and HERMES in EXPERTS
    pool = ExpertPool({role: MockLLM(role) for role in EXPERTS})
    assert GEMMA in pool.roles
    assert HERMES in pool.roles
    assert HERMES3 in pool.roles


def test_retired_roles_are_absent() -> None:
    pool_roles = build_expert_pool().roles
    for retired in ("finance", "router"):
        assert retired not in EXPERTS
        assert retired not in pool_roles


def test_tie_break_invokes_gemma_role() -> None:
    # One legal term ("court") and one finance term ("swap") -> a genuine tie that
    # forces the router to consult the Gemma-3 classifier.
    clients: dict = {role: MockLLM(role) for role in EXPERTS}
    spy = _SpyLLM(GEMMA, label="finance")
    clients[GEMMA] = spy
    decision = Router(ExpertPool(clients)).route("The court reviewed the swap.")
    assert spy.calls == 1
    assert decision.kind is RequestKind.FINANCE
    assert "tie broken" in decision.rationale


def test_ideator_runs_on_hermes_role() -> None:
    assert Ideator.expert_role == HERMES


def test_finance_analyst_runs_on_general_gemma_role() -> None:
    assert FinanceAnalyst.expert_role == GEMMA
