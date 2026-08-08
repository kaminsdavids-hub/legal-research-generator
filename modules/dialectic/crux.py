"""Crux extraction and weight-aware partition."""

from __future__ import annotations

from typing import Literal

from .models import CitationSlot, Crux, Position, SlotStatus, Weight
from .nli import NLIEvaluator


class PrecedenceRule:
    """Weight-aware precedence rule for the authority/fact/open partition.

    The precedence order is:

    1. ``controlling`` > ``persuasive`` > ``supporting`` > ``contra`` (always).
    2. Within the same weight, a ``verified`` slot outranks an unverified slot.

    A verified ``persuasive`` cite from another circuit therefore does NOT
    outrank an unverified but correct reading of ``controlling`` precedent.
    Verification only breaks ties at the same weight.
    """

    WEIGHT_ORDER = {
        Weight.CONTROLLING: 4,
        Weight.PERSUASIVE: 3,
        Weight.SUPPORTING: 2,
        Weight.CONTRA: 1,
    }

    def rank(self, slot: CitationSlot) -> tuple[int, int]:
        weight_score = self.WEIGHT_ORDER.get(slot.weight, 0)
        verified_score = 1 if slot.status == SlotStatus.VERIFIED else 0
        return (weight_score, verified_score)

    def outranks(self, a: CitationSlot, b: CitationSlot) -> bool:
        return self.rank(a) > self.rank(b)

    @staticmethod
    def _looks_factual(text: str) -> bool:
        t = text.lower()
        markers = (
            "record shows",
            "facts show",
            "evidence",
            "document",
            "witness",
            "timeline",
            "factual",
            "occurred",
            "conduct",
            "party",
            "plaintiff",
            "defendant",
        )
        return any(m in t for m in markers)

    def classify(self, crux: Crux) -> tuple[str, Literal["thesis", "antithesis", "none"]]:
        thesis = crux.thesis_prop
        antithesis = crux.antithesis_prop

        # Authority-resolvable only when one side strictly outranks the other.
        if self.outranks(thesis, antithesis):
            return "resolvable by authority", "thesis"
        if self.outranks(antithesis, thesis):
            return "resolvable by authority", "antithesis"

        # Same-rank contradictions about record facts are factual; everything else is open.
        if self._looks_factual(thesis.proposition) and self._looks_factual(
            antithesis.proposition
        ):
            return "resolvable by fact", "none"
        return "open", "none"


class CruxExtractor:
    """Extract outcome-bearing contradictions via a separate NLI pass."""

    def __init__(self, nli: NLIEvaluator | None = None, precedence: PrecedenceRule | None = None) -> None:
        self.nli = nli or NLIEvaluator()
        self.precedence = precedence or PrecedenceRule()

    @staticmethod
    def _outcome_bearing(slot: CitationSlot) -> bool:
        """Outcome-bearing propositions carry controlling or persuasive weight."""
        return slot.weight in (Weight.CONTROLLING, Weight.PERSUASIVE)

    def extract(self, thesis: Position, antithesis: Position) -> list[Crux]:
        """Extract contradictions across the full cross-product of propositions.

        Weight does NOT gate extraction. The weight is self-declared by each
        debater about its own argument, so gating on it let a modest model turn
        the whole crux engine off by assigning itself `supporting` — the module's
        central feature disabled by the very output it was meant to analyse.

        A contradiction between two `supporting` propositions is still a
        contradiction; it simply is not resolvable by authority.
        :meth:`PrecedenceRule.classify` already returns ``open`` for exactly that
        case, and :attr:`Crux.outcome_bearing` preserves the distinction so the
        UI can still rank.
        """
        cruxes: list[Crux] = []
        for t in thesis.propositions:
            for a in antithesis.propositions:
                rel, source = self.nli.relate(a.proposition, t.proposition)
                if rel != "contradiction":
                    continue
                crux = Crux(
                    thesis_prop=t,
                    antithesis_prop=a,
                    negates=True,
                    outcome_bearing=self._outcome_bearing(t) and self._outcome_bearing(a),
                    nli_source=source,
                )
                crux.partition, crux.winner = self.precedence.classify(crux)
                cruxes.append(crux)
        # Outcome-bearing cruxes first so the UI ranks without re-deriving weight.
        cruxes.sort(key=lambda c: not c.outcome_bearing)
        return cruxes
