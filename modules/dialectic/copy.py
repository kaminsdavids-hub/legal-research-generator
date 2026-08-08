"""Copy serializers for the dialectic turn.

All three granularities must preserve [UNSUPPORTED] markers so a reader can never
paste unverified authority into a filing without seeing the flag.
"""

from __future__ import annotations

from .models import CitationSlot, Crux, DialecticTurn, Position, SlotStatus


def _slot_marker(slot: CitationSlot) -> str:
    """Render the authority marker for a single proposition.

    A verified slot carrying a citation is the *only* state that renders clean.
    Everything else — including a verified slot with an empty cite, which is a
    partially-populated record rather than a resolved one — keeps a visible
    marker. A copy path that silently drops the marker is the single most
    dangerous defect in this module.
    """
    if slot.status == SlotStatus.VERIFIED:
        if slot.normalized_cite:
            return f" [{slot.normalized_cite}]"
        # Verified but no cite: nothing to paste, so never render clean.
        return (
            f" [UNSUPPORTED: {slot.weight} authority for {slot.proposition}"
            f" — marked verified with no citation; treat as unresolved]"
        )
    # Unfilled, proposed-but-unverified, or failed slots stay visible and flagged.
    marker = f" [UNSUPPORTED: {slot.weight} authority for {slot.proposition}]"
    if slot.status == SlotStatus.PROPOSED and slot.normalized_cite:
        # A candidate exists but no lookup confirmed it. Show the candidate and
        # the fact that it is unconfirmed; an unverified candidate must never
        # read as clean authority.
        marker += f" (proposed, unverified: {slot.normalized_cite})"
    if slot.note:
        marker += f" ({slot.note})"
    return marker


def _render_proposition(slot: CitationSlot, number: int) -> str:
    return f"  {number}. {slot.proposition}{_slot_marker(slot)}"


def _render_position(position: Position) -> str:
    lines = [f"{position.side.upper()} ({position.model})", ""]
    for idx, slot in enumerate(position.propositions, start=1):
        lines.append(_render_proposition(slot, idx))
    return "\n".join(lines)


def _render_crux(crux: Crux, number: int) -> str:
    rank = "outcome-bearing" if crux.outcome_bearing else "not outcome-bearing"
    # `nli_source` is shown so a downgrade from the NLI model to the offline
    # heuristic is legible in the copy, not buried in the JSON.
    lines = [
        f"{number}. {crux.partition} (winner: {crux.winner}; {rank}; nli: {crux.nli_source})",
        f"   THESIS: {crux.thesis_prop.proposition}{_slot_marker(crux.thesis_prop)}",
        f"   ANTITHESIS: {crux.antithesis_prop.proposition}{_slot_marker(crux.antithesis_prop)}",
    ]
    return "\n".join(lines)


def copy_exchange(turn: DialecticTurn) -> str:
    """Serialize the whole dialectic turn: both positions, synthesis, and cruxes."""
    parts = [
        f"QUESTION: {turn.question}",
        "",
        _render_position(turn.thesis),
        "",
        _render_position(turn.antithesis),
        "",
        f"SYNTHESIS:\n{turn.synthesis}",
    ]
    if turn.cruxes:
        parts.extend(["", "CRUX TABLE:", ""])
        parts.extend(_render_crux(c, i) for i, c in enumerate(turn.cruxes, start=1))
    else:
        parts.extend(["", copy_crux_table(turn)])
    parts.append(f"\nVerification calls spent: {turn.ledger.calls_spent}")
    if turn.regenerated:
        # A turn that burned regeneration attempts looks identical to a clean one
        # unless the count is stated. The per-slot `note` says why.
        parts.append(f"Regeneration attempts spent: {turn.regenerated}")
    return "\n".join(parts)


def copy_position(turn: DialecticTurn, side: str) -> str:
    """Serialize one side alone."""
    position = turn.antithesis if side == "antithesis" else turn.thesis
    return _render_position(position)


def copy_crux_table(turn: DialecticTurn) -> str:
    """Serialize the crux partition alone."""
    if not turn.cruxes:
        # An empty table renders the reason, never a bare "(none)": the reader
        # cannot otherwise tell "the sides agree" from "the extractor is off".
        reason = turn.crux_note or "no contradiction found between the two positions"
        return f"CRUX TABLE: (none) — {reason}"
    lines = ["CRUX TABLE:"]
    for i, crux in enumerate(turn.cruxes, start=1):
        lines.append(_render_crux(crux, i))
    return "\n".join(lines)
