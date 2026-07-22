"""Socratic Interviewer (spec §2.1).

Drives the brainstorm chat, asking the next-best question to sharpen the thesis,
surface counterarguments, test scope, and distinguish the paper's novel
contribution. Plan-first: keeps a living research question set from partial state.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import WRITER
from ..models import BrainstormRole, BrainstormTurn
from .base import Agent, AgentContext, AgentResult

_SYSTEM = (
    "TASK: interview\n"
    "You are a Socratic legal-research interviewer. From the scholar's partial ideas, "
    "ask exactly one sharp next question that tightens the thesis, tests its scope, "
    "surfaces the strongest counterargument, or isolates the novel contribution. "
    "Return only the question."
)


class SocraticInterviewer(Agent):
    name = "Socratic Interviewer"
    expert_role = WRITER

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        scholar_input: str | None = kwargs.get("scholar_input")
        bb = ctx.blackboard

        if scholar_input:
            bb.brainstorm.append(BrainstormTurn(role=BrainstormRole.SCHOLAR, content=scholar_input))
            if not bb.thesis:
                bb.thesis = scholar_input.strip()

        state = self._state_summary(ctx)
        client = ctx.pool.get(self.expert_role)
        question = client.chat(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", state)],
            DecodingPolicy.BALANCED.config,
        ).strip()
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        bb.brainstorm.append(BrainstormTurn(role=BrainstormRole.INTERVIEWER, content=question))
        return AgentResult(
            agent=self.name,
            summary=question,
            payload={"question": question, "turns": len(bb.brainstorm)},
        )

    def _state_summary(self, ctx: AgentContext) -> str:
        bb = ctx.blackboard
        thesis = bb.thesis or "(no thesis yet — the scholar has only a raw idea)"
        recent = bb.brainstorm[-4:]
        transcript = "\n".join(f"{t.role.value}: {t.content}" for t in recent)
        selected = ", ".join(i.text for i in bb.selected_ideas()) or "(none selected yet)"
        return (
            f"Working thesis: {thesis}\n"
            f"Selected ideas: {selected}\n"
            f"Recent conversation:\n{transcript or '(conversation just started)'}"
        )
