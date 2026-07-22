"""Agent base class and shared execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..blackboard import Blackboard
from ..citations.bluebook import CitationFormatter
from ..citations.corpus import Corpus
from ..citations.retriever import Retriever
from ..citations.verifier import CitationGuard, CitationVerifier
from ..config import Settings
from ..llm.pool import ExpertPool
from ..router import Router


@dataclass
class AgentContext:
    """Everything an agent needs to do its job, all local and injectable/mockable."""

    settings: Settings
    pool: ExpertPool
    router: Router
    retriever: Retriever
    corpus: Corpus
    formatter: CitationFormatter
    guard: CitationGuard
    verifier: CitationVerifier
    blackboard: Blackboard


class AgentResult(BaseModel):
    agent: str
    runtime: str = "openhands"
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    """Base class for the nine cooperating agents."""

    #: Human-readable agent name.
    name: str = "agent"
    #: Logical expert role this agent primarily routes to.
    expert_role: str = "writer"

    @abstractmethod
    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        """Read from and write to the blackboard; return a structured result."""
