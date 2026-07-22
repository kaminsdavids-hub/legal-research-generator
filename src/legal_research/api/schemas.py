"""Request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import IdeaStatus


class ConfigResponse(BaseModel):
    llm_mode: str
    retriever_mode: str
    pdf_renderer: str
    citation_style: str
    disclaimer: str


class CreateSessionRequest(BaseModel):
    title: str = "Untitled Research Paper"


class BrainstormRequest(BaseModel):
    message: str | None = None


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
