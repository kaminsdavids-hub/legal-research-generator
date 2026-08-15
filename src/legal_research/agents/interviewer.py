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
    "Never repeat, rephrase, or narrow a question already listed as asked -- move to "
    "the next unexamined weakness instead. Do not summarize, praise, or propose an "
    "answer. Return only the question."
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
        # Why the failure is reported rather than swallowed: every degradation
        # here (timeout, model-load failure, empty reasoning-only response) used
        # to collapse into the same three canned questions, so a dead model was
        # indistinguishable from a working one that happened to repeat itself.
        degraded: str | None = None
        try:
            question = client.chat(
                [ChatMessage("system", _SYSTEM), ChatMessage("user", state)],
                DecodingPolicy.BALANCED.config,
            ).strip()
        except Exception as exc:  # noqa: BLE001 - the interview must not crash
            question = ""
            degraded = f"interviewer model unavailable ({type(exc).__name__})"
        if not question:
            degraded = degraded or "interviewer model returned no question"
            question = self._fallback_question(bb)
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        bb.brainstorm.append(BrainstormTurn(role=BrainstormRole.INTERVIEWER, content=question))
        # Mirrored onto the blackboard because the brainstorm endpoint returns
        # only the blackboard; the payload alone never reaches the UI.
        bb.brainstorm_degraded = degraded or ""
        payload: dict[str, Any] = {"question": question, "turns": len(bb.brainstorm)}
        if degraded:
            payload["degraded"] = degraded
        return AgentResult(agent=self.name, summary=question, payload=payload)

    def _asked_questions(self, bb) -> list[str]:
        return [
            t.content.strip()
            for t in bb.brainstorm
            if t.role is BrainstormRole.INTERVIEWER and t.content.strip()
        ]

    def _fallback_question(self, bb) -> str:
        """Pick the first canned question that has not been asked yet.

        The fallbacks are ordered from broadest to narrowest. Without the
        already-asked check they repeat verbatim on every degraded turn, which is
        what makes a stalled interview look like a stuck one.
        """

        asked = {self._normalize(q) for q in self._asked_questions(bb)}
        candidates: list[str] = []
        if not bb.thesis:
            candidates.append(
                "What is the single narrow legal issue you want this paper to answer?"
            )
        if not bb.selected_ideas():
            candidates.append(
                "What is the strongest counterargument to your thesis, and where is its "
                "weakest step?"
            )
        candidates.extend(
            [
                "Which assumption in your current thesis is most vulnerable under the best "
                "opposing authority?",
                "Which authority most directly contradicts your thesis, and how do you "
                "distinguish it?",
                "What factual predicate must hold for your rule to apply, and what happens "
                "to the argument if it fails?",
                "What is the narrowest holding that would still make this paper worth "
                "publishing?",
            ]
        )
        for candidate in candidates:
            if self._normalize(candidate) not in asked:
                return candidate
        return candidates[-1]

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().rstrip("?").split())

    def _state_summary(self, ctx: AgentContext) -> str:
        bb = ctx.blackboard
        thesis = bb.thesis or "(no thesis yet — the scholar has only a raw idea)"
        # A 4-turn window hides the questions asked earlier in the interview, so
        # the model re-asks them. Give it a wider window plus an explicit list of
        # every question already put to the scholar.
        recent = bb.brainstorm[-10:]
        transcript = "\n".join(f"{t.role.value}: {t.content}" for t in recent)
        selected = ", ".join(i.text for i in bb.selected_ideas()) or "(none selected yet)"
        asked = self._asked_questions(bb)
        asked_block = "\n".join(f"- {q}" for q in asked) or "(none yet)"
        return (
            f"Working thesis: {thesis}\n"
            f"Selected ideas: {selected}\n"
            f"Questions already asked (do not repeat or rephrase these):\n{asked_block}\n"
            f"Recent conversation:\n{transcript or '(conversation just started)'}"
        )
