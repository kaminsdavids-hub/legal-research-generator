"""Request router (spec §1).

A small, auditable classifier decides which expert handles a request and with which
decoding policy. Rule-based first (deterministic and inspectable); the Gemma-3
router model is only consulted to break genuine ties. Legal requests are forced
cold with mandatory citation-RAG; brainstorming runs hot on the writer engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .llm.base import ChatMessage, DecodingPolicy
from .llm.pool import FINANCE, SAUL, WRITER, ExpertPool


class RequestKind(str, Enum):
    LEGAL = "legal"
    FINANCE = "finance"
    SYNTHESIS = "synthesis"
    BRAINSTORM = "brainstorm"


@dataclass(frozen=True)
class RoutingDecision:
    kind: RequestKind
    expert: str
    policy: DecodingPolicy
    forced_citation: bool
    rationale: str


_LEGAL = {
    "holding", "held", "statute", "statutory", "precedent", "doctrine", "circuit",
    "court", "plaintiff", "defendant", "ruling", "case", "constitutional", "liability",
    "remedy", "jurisdiction", "regulation", "reg", "usc", "cfr",
}
_FINANCE = {
    "basel", "capital", "collateral", "swap", "derivative", "liquidity", "bank",
    "banking", "securitization", "securitisation", "leverage", "margin", "hedge",
    "instrument", "counterparty", "solvency",
}
_HEAVY_QUANT = {
    "model", "regression", "pricing", "quantitative", "calculate", "optimization",
    "stochastic", "var", "monte", "valuation",
}
_BRAINSTORM = {
    "brainstorm", "ideate", "idea", "angle", "novel", "novelty", "framing",
    "thesis", "explore", "what if",
}


def _score(text: str, vocab: set[str]) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for term in vocab if f" {term}" in lowered or f"{term} " in lowered)


class Router:
    def __init__(self, pool: ExpertPool) -> None:
        self._pool = pool

    def route(self, text: str, hint: RequestKind | None = None) -> RoutingDecision:
        if hint is not None:
            return self._decision_for(hint, text, rationale=f"explicit hint={hint.value}")

        scores = {
            RequestKind.BRAINSTORM: _score(text, _BRAINSTORM),
            RequestKind.LEGAL: _score(text, _LEGAL),
            RequestKind.FINANCE: _score(text, _FINANCE),
        }
        best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])

        if best_score == 0:
            return self._decision_for(RequestKind.SYNTHESIS, text, rationale="no domain signal")

        # Tie between top two non-zero categories -> ask the router model.
        ranked = sorted(scores.values(), reverse=True)
        if len(ranked) >= 2 and ranked[0] == ranked[1] and ranked[0] > 0:
            best_kind = self._break_tie(text, scores)
            rationale = "tie broken by gemma3 router"
        else:
            rationale = f"keyword score={best_score} for {best_kind.value}"

        return self._decision_for(best_kind, text, rationale=rationale)

    def _decision_for(self, kind: RequestKind, text: str, rationale: str) -> RoutingDecision:
        if kind is RequestKind.LEGAL:
            return RoutingDecision(kind, SAUL, DecodingPolicy.COLD, True, rationale)
        if kind is RequestKind.FINANCE:
            # Heavy quantitative finance is better handled by the reasoning writer
            # engine; domain finance models are weaker reasoners (spec §1).
            if _score(text, _HEAVY_QUANT) >= 2:
                return RoutingDecision(
                    RequestKind.FINANCE, WRITER, DecodingPolicy.BALANCED, False,
                    rationale + " + heavy-quant -> writer engine",
                )
            return RoutingDecision(kind, FINANCE, DecodingPolicy.BALANCED, False, rationale)
        if kind is RequestKind.BRAINSTORM:
            return RoutingDecision(kind, WRITER, DecodingPolicy.HOT, False, rationale)
        return RoutingDecision(RequestKind.SYNTHESIS, WRITER, DecodingPolicy.BALANCED, False, rationale)

    def _break_tie(self, text: str, scores: dict[RequestKind, int]) -> RequestKind:
        client = self._pool.get("router")
        messages = [
            ChatMessage("system", "TASK: route\nReturn exactly one label: legal, finance, brainstorm, or synthesis."),
            ChatMessage("user", text),
        ]
        label = client.chat(messages, DecodingPolicy.COLD.config).strip().lower()
        for kind in RequestKind:
            if kind.value in label:
                return kind
        # Fall back to the highest keyword score, stable by enum order.
        return max(scores.items(), key=lambda kv: kv[1])[0]
