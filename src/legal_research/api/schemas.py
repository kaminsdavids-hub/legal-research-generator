"""Request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import IdeaStatus


class ConfigResponse(BaseModel):
    debug_endpoints_enabled: bool
    llm_mode: str
    retriever_mode: str
    embed_model: str
    support_scorer: str
    saul_model: str
    writer_model: str
    gemma_model: str
    hermes_model: str
    hermes3_model: str
    multi_chat_models: dict[str, str]
    multi_chat_verifiers: dict[str, str]
    dialectic_models: dict[str, str] = Field(default_factory=dict)
    grammar_chain_enabled: bool
    grammar_chain_roles: list[str]
    pdf_renderer: str
    citation_style: str
    manuscript_target_min_words: int
    manuscript_target_max_words: int
    disclaimer: str


class CreateSessionRequest(BaseModel):
    title: str = "Untitled Research Paper"


class BrainstormRequest(BaseModel):
    message: str | None = None


class MultiChatTurn(BaseModel):
    role: str
    content: str


class MultiChatRequest(BaseModel):
    message: str
    history: list[MultiChatTurn] = Field(default_factory=list)


class MultiChatModelAnswer(BaseModel):
    model: str
    content: str


class MultiChatVerifierResult(BaseModel):
    model: str
    verdict: str


class MultiChatCitationFinding(BaseModel):
    citation: str
    kind: str
    status: str
    detail: str
    record_id: str = ""
    support: float = 0.0


class MultiChatGrounding(BaseModel):
    available: bool = True
    summary: str = ""
    note: str = ""
    authorities: list[str] = Field(default_factory=list)
    findings: list[MultiChatCitationFinding] = Field(default_factory=list)
    verified_count: int = 0
    unverified_count: int = 0
    unconfirmed_count: int = 0


class MultiChatResponse(BaseModel):
    final_answer: str
    model_answers: list[MultiChatModelAnswer]
    verifiers: list[MultiChatVerifierResult]
    grounding: MultiChatGrounding = Field(default_factory=MultiChatGrounding)


class DialecticRequest(BaseModel):
    message: str


class DialecticSlot(BaseModel):
    proposition: str
    court_hint: str = ""
    weight: str = "supporting"
    status: str = "pending"
    cluster_id: str = ""
    normalized_cite: str = ""
    note: str = ""


class DialecticPosition(BaseModel):
    side: str
    model: str
    family: str
    propositions: list[DialecticSlot] = Field(default_factory=list)


class DialecticCrux(BaseModel):
    thesis_prop: DialecticSlot
    antithesis_prop: DialecticSlot
    negates: bool = False
    partition: str = ""
    winner: str = "none"
    #: True when both sides carry controlling or persuasive weight. A crux
    #: between two `supporting` propositions is still a contradiction; it is
    #: just not resolvable by authority. Lets a client rank without re-deriving
    #: weight.
    outcome_bearing: bool = False
    #: Which NLI path produced this relation, "model" or "heuristic". A silent
    #: downgrade to the offline heuristic changes what the label means, so it
    #: travels with the label rather than being inferred.
    nli_source: str = "heuristic"


class DialecticResponse(BaseModel):
    question: str
    thesis: DialecticPosition
    antithesis: DialecticPosition
    synthesis: str = ""
    cruxes: list[DialecticCrux] = Field(default_factory=list)
    calls_spent: int = 0
    #: Regeneration attempts spent across both positions and the synthesis. A
    #: turn that burned its budget is otherwise indistinguishable from a clean
    #: first pass; the per-slot `note` says why.
    regenerated: int = 0
    #: Why the crux table is empty, when it is. An empty table with no
    #: explanation is indistinguishable from a broken extractor.
    crux_note: str = ""
    #: Pre-rendered copy payloads; the serializers preserve [UNSUPPORTED] markers.
    copy_exchange: str = ""
    copy_thesis: str = ""
    copy_antithesis: str = ""
    copy_crux_table: str = ""


class RetrievalSmokeRequest(BaseModel):
    query: str
    k: int = 3


class RetrievalSmokeHit(BaseModel):
    record_id: str
    score: float
    locator: str
    text: str


class RetrievalSmokeResponse(BaseModel):
    retriever_mode: str
    embed_model: str
    retriever_impl: str
    hits: list[RetrievalSmokeHit]


class IdeateRequest(BaseModel):
    seed: str | None = None


class SelectionItem(BaseModel):
    idea_id: str
    priority: int = 0


class SelectIdeasRequest(BaseModel):
    selections: list[SelectionItem] = Field(default_factory=list)


class IdeaStatusUpdate(BaseModel):
    status: IdeaStatus
    priority: int | None = None


class ReviseRequest(BaseModel):
    section_id: str
    instruction: str


class DraftEssayRequest(BaseModel):
    target_words: int = 2000


class SocraticTurn(BaseModel):
    role: str
    content: str


SocraticMode = Literal[
    "strengthen_doctrine",
    "expand_analysis",
    "counter_rebuttal",
    "policy_implications",
    "comparative_framework",
]


class SocraticReviseRequest(BaseModel):
    section_id: str
    paragraph_index: int = 0
    message: str
    history: list[SocraticTurn] = Field(default_factory=list)
    apply_revision: bool = False
    mode: SocraticMode = "strengthen_doctrine"


class SocraticReviseResponse(BaseModel):
    section_id: str
    paragraph_index: int
    assistant: str
    suggested_revision: str = ""
    applied: bool = False
    mode: SocraticMode = "strengthen_doctrine"
    cycle_break_triggered: bool = False
    novelty_score: float = 0.0
    rewrite_delta_score: float = 0.0


class RunAllRequest(BaseModel):
    idea: str
    title: str = "Untitled Research Paper"
    max_ideas: int = 3


class RenderPdfRequest(BaseModel):
    author: str = "Anonymous Scholar"
    date: str = ""


class StepInfo(BaseModel):
    agent: str
    runtime: str
    summary: str


class RunAllResponse(BaseModel):
    session_id: str
    steps: list[StepInfo]
    shippable: bool


class JobRequest(BaseModel):
    #: Which pipeline step to run. Validated against an explicit allow-list in
    #: the route, not against this type, so an unknown value gets a 400 naming
    #: the permitted set rather than a 422 with a schema dump.
    step: str
    #: Arguments for the steps that take them. Carried here rather than on a
    #: route per step so that a client, and any proxy in front of it, have one
    #: submission shape to learn. Each field names the step that reads it;
    #: everything else ignores them.
    #:
    #: run-all:
    idea: str = ""
    title: str = "Untitled Research Paper"
    max_ideas: int = 3
    #: multi-chat and dialectic:
    message: str = ""
    #: multi-chat only:
    history: list[MultiChatTurn] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    session_id: str
    step: str
    state: str
    #: Non-empty only when the step failed. The message, not the traceback.
    error: str = ""
    #: A small terminal summary, present only on success and only for steps that
    #: produce something outside the blackboard. `run-all` sets it to the
    #: per-agent step log and the shippable verdict; every other step leaves it
    #: null, because their result *is* the blackboard.
    result: dict[str, Any] | None = None
