"""Router classification (spec §1)."""

from __future__ import annotations

from legal_research.llm.base import DecodingPolicy
from legal_research.llm.pool import FINANCE, SAUL, WRITER, build_expert_pool
from legal_research.router import RequestKind, Router


def _router() -> Router:
    return Router(build_expert_pool())


def test_legal_routes_to_saul_cold_with_forced_citation() -> None:
    decision = _router().route("Analyze the holding and statutory basis of the doctrine.")
    assert decision.kind is RequestKind.LEGAL
    assert decision.expert == SAUL
    assert decision.policy is DecodingPolicy.COLD
    assert decision.forced_citation is True


def test_finance_routes_to_finance_model() -> None:
    decision = _router().route("Explain Basel capital and bank leverage requirements.")
    assert decision.kind is RequestKind.FINANCE
    assert decision.expert == FINANCE


def test_heavy_quant_finance_routes_to_writer_engine() -> None:
    decision = _router().route(
        "Build a quantitative model to calculate the pricing and valuation of the swap."
    )
    assert decision.expert == WRITER


def test_brainstorm_routes_hot() -> None:
    decision = _router().route("Let's brainstorm a novel angle and idea for the thesis.")
    assert decision.kind is RequestKind.BRAINSTORM
    assert decision.policy is DecodingPolicy.HOT


def test_default_is_synthesis() -> None:
    decision = _router().route("Please tie these threads together into a coherent whole.")
    assert decision.kind is RequestKind.SYNTHESIS
    assert decision.expert == WRITER
