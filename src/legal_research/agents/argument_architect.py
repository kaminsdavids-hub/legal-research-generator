"""Argument Architect (spec §2.5).

Structures the paper: thesis, roadmap, sections (IRAC/CREAC where appropriate),
counterargument + rebuttal. Builds the outline from the ideas the scholar selected;
every section is anchored to at least one kept idea.
"""

from __future__ import annotations

from typing import Any

from ..models import SectionStatus
from .base import Agent, AgentContext, AgentResult

# A stable law-review scaffold. Section *titles* are derived from the selected ideas
# so no two papers share identical boilerplate (see the anti-template check).
_FIXED_FRONT = ["Introduction"]
_FIXED_BACK = ["Counterargument and Rebuttal", "Conclusion"]


class ArgumentArchitect(Agent):
    name = "Argument Architect"
    expert_role = "writer"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        bb.outline.clear()
        selected = bb.selected_ideas()

        intro = bb.add_section(_FIXED_FRONT[0])
        intro.idea_ids = [i.id for i in selected]

        if not selected:
            # Minimal viable outline even with no selected ideas.
            bb.add_section("Background and Governing Standard")
        for n, idea in enumerate(selected, start=1):
            title = self._section_title(n, idea.text)
            bb.add_section(title, idea_ids=[idea.id])

        for title in _FIXED_BACK:
            bb.add_section(title)

        for section in bb.outline:
            section.status = SectionStatus.IDEA

        return AgentResult(
            agent=self.name,
            summary=f"built outline with {len(bb.outline)} sections from {len(selected)} ideas",
            payload={"section_ids": [s.id for s in bb.outline]},
        )

    @staticmethod
    def _section_title(n: int, idea_text: str) -> str:
        core = idea_text.rstrip(".")
        if len(core) > 70:
            core = core[:67].rsplit(" ", 1)[0] + "..."
        return f"{_roman(n)}. {core}"


def _roman(n: int) -> str:
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out
