"""Domain models shared across the blackboard, agents, citations and API.

These are Pydantic models so the FastAPI layer can (de)serialize them directly and
so the frontend receives a stable, typed contract.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IdeaStatus(str, Enum):
    CANDIDATE = "candidate"
    KEEP = "keep"
    PARK = "park"
    DISCARD = "discard"


class SectionStatus(str, Enum):
    IDEA = "idea"
    DRAFTED = "drafted"
    CITED = "cited"
    VERIFIED = "verified"


class SourceType(str, Enum):
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    SECONDARY = "secondary"


class CiteStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    REMOVED = "removed"


class Relation(str, Enum):
    SUPPORTING = "supporting"
    CONTRARY = "contrary"


class BrainstormRole(str, Enum):
    SCHOLAR = "scholar"
    INTERVIEWER = "interviewer"
    IDEATOR = "ideator"


class Idea(BaseModel):
    id: str
    text: str
    angle: str = ""
    novelty_note: str = ""
    status: IdeaStatus = IdeaStatus.CANDIDATE
    priority: int = 0


class OutlineSection(BaseModel):
    id: str
    title: str
    status: SectionStatus = SectionStatus.IDEA
    idea_ids: list[str] = Field(default_factory=list)
    content: str = ""
    citation_ids: list[str] = Field(default_factory=list)


class RetrievedPassage(BaseModel):
    record_id: str
    score: float
    text: str
    locator: str = ""


class Authority(BaseModel):
    record_id: str
    citation: str
    relation: Relation
    proposition: str
    passage: str


class Citation(BaseModel):
    id: str
    record_id: str
    proposition: str
    quote: str | None = None
    pin_cite: str | None = None
    status: CiteStatus = CiteStatus.PENDING
    supporting_passage: str = ""
    note: str = ""
    # Provenance flag: True only if the authority came from the retriever. A cite
    # that a model tried to assert from its own weights is marked False and blocked.
    from_retrieval: bool = False


class VerificationResult(BaseModel):
    citation_id: str
    record_id: str
    status: CiteStatus
    reason: str
    supporting_passage: str = ""


class EditRecord(BaseModel):
    id: str
    section_id: str
    before: str
    after: str
    note: str = ""
    author: str = "editor"


class BrainstormTurn(BaseModel):
    role: BrainstormRole
    content: str


class NoveltyAssessment(BaseModel):
    contribution: str
    distinguished_from: list[str] = Field(default_factory=list)
    score: float = 0.0
    grounded: bool = False


class FootnoteRef(BaseModel):
    number: int
    text: str
    citation_id: str
