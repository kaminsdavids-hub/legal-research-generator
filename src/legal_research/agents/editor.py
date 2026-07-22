"""Editor / Reviser (spec §2.7).

Line/substantive editing: tightens prose, enforces the anti-"AI voice" lint, applies
the scholar's revision requests on a selection, and tracks changes as EditRecords.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import WRITER
from ..voice.anti_ai_voice import lint_ai_voice, rewrite_for_voice
from .base import Agent, AgentContext, AgentResult

_CITE_TOKEN = re.compile(r"\{\{cite:[^}]+\}\}")

_SYSTEM = (
    "TASK: edit\n"
    "You are a meticulous legal editor. Apply the requested revision to the selection "
    "while preserving citations and meaning. Return only the revised text."
)


def _strip_tokens(text: str) -> str:
    return _CITE_TOKEN.sub("", text).strip()


class EditorAgent(Agent):
    name = "Editor / Reviser"
    expert_role = WRITER

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        section_id: str | None = kwargs.get("section_id")
        instruction: str | None = kwargs.get("instruction")
        bb = ctx.blackboard

        if section_id and instruction:
            return self._revise_selection(ctx, section_id, instruction)

        # Default pass: enforce anti-AI-voice across all drafted sections.
        rewrites = 0
        for section in bb.outline:
            if not section.content:
                continue
            prose = _strip_tokens(section.content)
            report = lint_ai_voice(prose)
            if report.passed:
                continue
            revised = rewrite_for_voice(prose)
            tokens = "".join(_CITE_TOKEN.findall(section.content))
            new_content = (revised + tokens).strip()
            bb.add_edit(
                section_id=section.id,
                before=section.content,
                after=new_content,
                note=f"anti-AI-voice rewrite: {[f.kind for f in report.failures()]}",
                author=self.name,
            )
            section.content = new_content
            rewrites += 1

        return AgentResult(
            agent=self.name,
            summary=f"voice pass complete; rewrote {rewrites} sections",
            payload={"rewrites": rewrites},
        )

    def _revise_selection(
        self, ctx: AgentContext, section_id: str, instruction: str
    ) -> AgentResult:
        bb = ctx.blackboard
        section = bb.get_section(section_id)
        client = ctx.pool.get(self.expert_role)
        tokens = "".join(_CITE_TOKEN.findall(section.content))
        revised = client.chat(
            [
                ChatMessage("system", _SYSTEM),
                ChatMessage("user", f"Instruction: {instruction}\n\nSelection:\n{_strip_tokens(section.content)}"),
            ],
            DecodingPolicy.BALANCED.config,
        ).strip()
        new_content = (revised + tokens).strip()
        record = bb.add_edit(
            section_id=section_id,
            before=section.content,
            after=new_content,
            note=instruction,
            author=self.name,
        )
        section.content = new_content
        return AgentResult(
            agent=self.name,
            summary=f"revised {section_id} per instruction",
            payload={"edit_id": record.id},
        )
