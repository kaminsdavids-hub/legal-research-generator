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
    #: The one-sentence assertion this idea makes, as opposed to the topic
    #: `text` names. Retrieval and verification need something a source can
    #: support: "Analyze how weights are treated" cannot be entailed by
    #: anything, while "Weights are published information under the EAR" can.
    #: Empty when the generator produced only a topic (REMEDIATION §25).
    claim: str = ""
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
    #: "entailed" | "rule_support" | "none". A verified citation is not one
    #: thing: entailment means the passage makes the proposition true, while
    #: rule support means it states the rule the proposition applies and the
    #: application step is the author's. Merging them would hide the weaker
    #: claim behind the stronger word.
    relation: str = "none"


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


class MechanismFinding(BaseModel):
    """One passage whose argument rests on an operation it never states.

    Produced by :mod:`legal_research.mechanism`. ``resolved`` records whether the
    gate's repair pass actually fixed it — verified by re-scanning the rewrite,
    not by the rewriting model's own account of its work. Unresolved findings
    stay on the blackboard and in the verification report: a defect the gate
    could not repair is a fact about the manuscript, not a stage that failed.
    """

    section_id: str
    section_title: str = ""
    paragraph_index: int = 0
    #: "BLACK_BOX" (no structure anywhere near) or "UNDER_SPECIFIED" (too thin).
    severity: str
    term: str = ""
    reason: str = ""
    excerpt: str = ""
    #: "lexical" (deterministic scan) or "semantic" (LLM critic).
    source: str = "lexical"
    resolved: bool = False
