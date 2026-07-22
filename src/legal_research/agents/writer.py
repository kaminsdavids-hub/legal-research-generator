"""Writer / Voice (spec §2.6, §4, §5).

Produces human-like, distinctive prose per section. It is the enforcement point for
the anti-hallucination rule: every legal proposition it asserts must be grounded
through the retriever via the :class:`CitationGuard`. A proposition that cannot be
grounded is dropped (never invented) and recorded as a blocked assertion.

Grounded sentences carry a ``{{cite:ID}}`` token that the Citation Formatter later
turns into a numbered footnote.
"""

from __future__ import annotations

from typing import Any

from ..citations.verifier import GroundingError
from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import WRITER
from ..models import SectionStatus
from .base import Agent, AgentContext, AgentResult

_SYSTEM = (
    "TASK: write\n"
    "You are a legal scholar writing in a distinctive, human voice: varied sentence "
    "length, no formulaic transitions, no empty summarizing filler. Write a tight "
    "paragraph developing the given point. Do not invent citations."
)


class WriterAgent(Agent):
    name = "Writer / Voice"
    expert_role = WRITER

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        section_ids: list[str] = kwargs.get("section_ids") or [s.id for s in bb.outline]
        client = ctx.pool.get(self.expert_role)

        drafted = 0
        blocked: list[str] = []
        for section_id in section_ids:
            section = bb.get_section(section_id)
            propositions = self._section_propositions(ctx, section)

            prose = client.chat(
                [
                    ChatMessage("system", _SYSTEM),
                    ChatMessage("user", f"Section: {section.title}\nPoint: {propositions[0] if propositions else section.title}"),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()

            markers = ""
            section.citation_ids = []
            for proposition in propositions:
                try:
                    citation = ctx.guard.ground(proposition)
                    ctx.guard.assert_grounded(citation)
                except GroundingError:
                    blocked.append(proposition)
                    continue
                bb.add_citation(citation)
                section.citation_ids.append(citation.id)
                # Marker sits inline right after the prose; a removed cite's marker is
                # later stripped cleanly, leaving no dangling text.
                markers += f"{{{{cite:{citation.id}}}}}"

            section.content = (prose + markers).strip()
            section.status = SectionStatus.DRAFTED if section.content else SectionStatus.IDEA
            if section.content:
                drafted += 1

        return AgentResult(
            agent=self.name,
            summary=f"drafted {drafted} sections; blocked {len(blocked)} ungrounded assertions",
            payload={"blocked_assertions": blocked},
        )

    def _section_propositions(self, ctx: AgentContext, section: Any) -> list[str]:
        bb = ctx.blackboard
        props: list[str] = []
        # Anchor the introduction on the thesis so the paper's central claim is cited.
        is_intro = bool(bb.outline) and bb.outline[0].id == section.id
        if is_intro and bb.thesis:
            props.append(bb.thesis)
        for idea_id in section.idea_ids:
            try:
                text = bb.get_idea(idea_id).text
                if text not in props:
                    props.append(text)
            except KeyError:
                continue
        return props
