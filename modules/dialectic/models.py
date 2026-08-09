"""Domain models for the dialectic chat module."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Marker written into a slot's note when its authority is no longer operative.
#: Shared between the producer (the corpus adapter) and the renderer, so a
#: rescinded rule cannot render as clean authority just because it verified.
NOT_OPERATIVE = "NOT CURRENTLY OPERATIVE"


class Weight(StrEnum):
    """Weight of authority attached to a proposition's citation slot.

    A real enum, not ``class Weight(str)`` with class attributes: the latter
    typed :attr:`CitationSlot.weight` as a bare ``str``, so any string at all
    validated and ``extra="forbid"`` gave no protection.
    """

    CONTROLLING = "controlling"
    PERSUASIVE = "persuasive"
    SUPPORTING = "supporting"
    CONTRA = "contra"


class SlotStatus(StrEnum):
    """Resolution status of a citation slot.

    ``PENDING`` and ``PROPOSED`` are deliberately distinct: ``pending`` means
    nothing has been proposed yet, ``proposed`` means a candidate citation is
    awaiting lookup. Collapsing them hid the fact that a slot carrying an
    unconfirmed candidate is not the same as one carrying nothing.
    """

    PENDING = "pending"
    PROPOSED = "proposed"
    NOT_FOUND = "not_found"
    VERIFIED = "verified"


class CitationSlot(BaseModel):
    """A single proposition and the authority slot attached to it."""

    model_config = ConfigDict(extra="forbid")

    proposition: str
    court_hint: str = ""
    weight: Weight = Weight.SUPPORTING
    status: SlotStatus = SlotStatus.PENDING
    cluster_id: str = ""
    normalized_cite: str = ""
    note: str = ""


class Position(BaseModel):
    """One side of a dialectic exchange."""

    model_config = ConfigDict(extra="forbid")

    side: Literal["thesis", "antithesis"]
    model: str
    family: str
    propositions: list[CitationSlot] = Field(default_factory=list)
    raw: str = ""


class Crux(BaseModel):
    """A direct contradiction between a thesis and an antithesis proposition.

    Not every crux is outcome-bearing. A contradiction between two ``supporting``
    propositions is still a contradiction; it simply is not resolvable by
    authority. :attr:`outcome_bearing` records which kind this is so the UI can
    rank, instead of the extractor discarding the pair outright.
    """

    model_config = ConfigDict(extra="forbid")

    thesis_prop: CitationSlot
    antithesis_prop: CitationSlot
    negates: bool = False
    partition: str = ""
    winner: Literal["thesis", "antithesis", "none"] = "none"
    #: True when both sides carry controlling or persuasive weight.
    outcome_bearing: bool = False
    #: Which NLI path produced this relation. A silent downgrade from the model
    #: to the heuristic must be visible in the output, not invisible.
    nli_source: Literal["model", "heuristic"] = "heuristic"


class BudgetLedger(BaseModel):
    """Spending record for one dialectic exchange."""

    model_config = ConfigDict(extra="forbid")

    calls_spent: int = 0
    cache_hits: int = 0
    exhausted: bool = False
    notes: list[str] = Field(default_factory=list)


class DialecticTurn(BaseModel):
    """A complete dialectic turn: thesis, antithesis, synthesis, and cruxes."""

    model_config = ConfigDict(extra="forbid")

    question: str
    thesis: Position
    antithesis: Position
    synthesis: str = ""
    cruxes: list[Crux] = Field(default_factory=list)
    ledger: BudgetLedger = Field(default_factory=BudgetLedger)
    calls_spent: int = 0
    #: Total regeneration attempts spent across both positions.
    regenerated: int = 0
    #: Why the crux table is empty, when it is. An empty table with no
    #: explanation is indistinguishable from a broken extractor.
    crux_note: str = ""
