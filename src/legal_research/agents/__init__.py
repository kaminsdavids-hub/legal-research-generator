"""Multi-agent layer (spec §2).

Nine cooperating agents share one blackboard. Two runtimes — OpenHands (primary
orchestrator) and NemoClaw (second cooperating runtime) — sit behind a single
:class:`AgentRuntime` interface so agents are portable across both.
"""

from __future__ import annotations

from .argument_architect import ArgumentArchitect
from .base import Agent, AgentContext, AgentResult
from .citation_formatter import CitationFormatterAgent
from .editor import EditorAgent
from .finance_analyst import FinanceAnalyst
from .ideator import Ideator
from .interviewer import SocraticInterviewer
from .legal_researcher import LegalResearcher
from .runtime import AgentRuntime, NemoClawRuntime, OpenHandsRuntime, build_runtime
from .verifier import VerifierAgent
from .writer import WriterAgent

ALL_AGENTS = (
    SocraticInterviewer,
    Ideator,
    LegalResearcher,
    FinanceAnalyst,
    ArgumentArchitect,
    WriterAgent,
    EditorAgent,
    CitationFormatterAgent,
    VerifierAgent,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentRuntime",
    "OpenHandsRuntime",
    "NemoClawRuntime",
    "build_runtime",
    "SocraticInterviewer",
    "Ideator",
    "LegalResearcher",
    "FinanceAnalyst",
    "ArgumentArchitect",
    "WriterAgent",
    "EditorAgent",
    "CitationFormatterAgent",
    "VerifierAgent",
    "ALL_AGENTS",
]
