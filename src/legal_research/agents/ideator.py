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
from .legal_researcher import is_assertable


def _split_topic_and_claim(body: str) -> tuple[str, str]:
    """Split a ``TOPIC || CLAIM`` bullet.

    A generator that ignores the format still yields a usable topic, so the
    separator being absent is not an error -- it means no claim was offered.
    """
    if _CLAIM_SEPARATOR in body:
        topic, _, claim = body.partition(_CLAIM_SEPARATOR)
        return topic.strip(), claim.strip()
    return body.strip(), ""

_SYSTEM = (
    "TASK: ideate\n"
    "You are an idea generator for a legal-research paper. Produce several distinct, "
    "non-obvious candidate angles as a bullet list, one per line beginning with '- '. "
    "Each should suggest a genuinely original contribution.\n"
    "Write each line as 'TOPIC || CLAIM', where CLAIM is a single declarative "
    "sentence the paper would assert and a source could support or contradict. "
    "Do not write the claim as a question, a title, or an instruction to analyze "
    "something: 'Analyze how weights are treated' is not a claim, "
    "'Model weights are published information under the EAR' is."
)

#: Separates the topic from the claim on each bullet.
_CLAIM_SEPARATOR = "||"


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
        rejected = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:].strip()
            if not body:
                continue

            text, claim = _split_topic_and_claim(body)
            if not text:
                continue
            # An unassertable "claim" is worse than none: it would travel all
            # the way to the Verifier and be asked of a source, which is how a
            # title came to account for every verification in a live run
            # (REMEDIATION §24). Leaving it empty makes the caller fall back to
            # the thesis, which at least is a claim.
            if claim and not is_assertable(claim):
                rejected += 1
                claim = ""

            idea = ctx.blackboard.add_idea(
                text=text,
                claim=claim,
                angle="original-contribution",
                novelty_note="Proposed as a departure from the retrieved literature.",
            )
            created.append(idea)

        with_claims = sum(1 for i in created if i.claim)
        return AgentResult(
            agent=self.name,
            summary=(
                f"generated {len(created)} candidate ideas; "
                f"{with_claims} carry an assertable claim"
                + (f" ({rejected} rejected as unassertable)" if rejected else "")
            ),
            payload={
                "idea_ids": [i.id for i in created],
                "with_claims": with_claims,
                "rejected_claims": rejected,
            },
        )
