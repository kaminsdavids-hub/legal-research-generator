"""Ideator (spec §2.2).

HOT decoding: generates candidate theses, framings and novel angles, and proposes
what would make the paper an original contribution. Feeds the idea board. Runs on
the Hermes role, backed by NVIDIA Nemotron 3 Nano, reasoning engine.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import HERMES
from ..models import Idea
from .base import Agent, AgentContext, AgentResult

_SYSTEM = (
    "TASK: ideate\n"
    "You are an idea generator for a legal-research paper. Produce several distinct, "
    "non-obvious candidate angles as a bullet list, one per line beginning with '- '. "
    "Each should suggest a genuinely original contribution."
)


class Ideator(Agent):
    name = "Ideator"
    expert_role = HERMES

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        seed: str = kwargs.get("seed") or ctx.blackboard.thesis or "the doctrine at issue"
        client = ctx.pool.get(self.expert_role)
        raw = client.chat(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", seed)],
            DecodingPolicy.HOT.config,
        )

        created: list[Idea] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            text = line[2:].strip()
            if not text:
                continue
            idea = ctx.blackboard.add_idea(
                text=text,
                angle="original-contribution",
                novelty_note="Proposed as a departure from the retrieved literature.",
            )
            created.append(idea)

        return AgentResult(
            agent=self.name,
            summary=f"generated {len(created)} candidate ideas",
            payload={"idea_ids": [i.id for i in created]},
        )
