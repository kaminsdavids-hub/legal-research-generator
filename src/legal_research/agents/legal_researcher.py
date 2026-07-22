"""Legal Researcher (spec §2.3).

Finds controlling/persuasive authority via retrieval only and builds the authority
map (per proposition: supporting + contrary), flagging splits and open questions.
Authorities never come from model weights. Grounded citations are created later by
the Writer, which cites via the retriever at draft time.
"""

from __future__ import annotations

from typing import Any

from ..citations.verifier import support_score
from ..models import Authority, Relation, SourceType
from .base import Agent, AgentContext, AgentResult

SUPPORT_CUTOFF = 0.34


class LegalResearcher(Agent):
    name = "Legal Researcher"
    expert_role = "saul"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        propositions: list[str] = kwargs.get("propositions") or self._default_props(ctx)

        bb = ctx.blackboard
        for proposition in propositions:
            hits = ctx.retriever.search(proposition, k=4)
            legal_hits = [
                h
                for h in hits
                if (rec := ctx.corpus.get(h.record_id))
                and rec.type in (SourceType.CASE, SourceType.STATUTE, SourceType.REGULATION)
            ]
            for h in legal_hits:
                record = ctx.corpus.get(h.record_id)
                if record is None:
                    continue
                if not any(p.record_id == h.record_id for p in bb.retrieved):
                    bb.retrieved.append(h)
                relation = (
                    Relation.SUPPORTING
                    if support_score(proposition, h.text) >= SUPPORT_CUTOFF
                    else Relation.CONTRARY
                )
                bb.authorities.append(
                    Authority(
                        record_id=record.id,
                        citation=ctx.formatter.full(record),
                        relation=relation,
                        proposition=proposition,
                        passage=h.text,
                    )
                )

        supporting = sum(1 for a in bb.authorities if a.relation is Relation.SUPPORTING)
        contrary = sum(1 for a in bb.authorities if a.relation is Relation.CONTRARY)
        return AgentResult(
            agent=self.name,
            summary=(
                f"mapped {len(bb.authorities)} authorities "
                f"({supporting} supporting, {contrary} contrary)"
            ),
            payload={
                "authorities": len(bb.authorities),
                "supporting": supporting,
                "contrary": contrary,
            },
        )

    def _default_props(self, ctx: AgentContext) -> list[str]:
        bb = ctx.blackboard
        props = [i.text for i in bb.selected_ideas()]
        if bb.thesis:
            props.insert(0, bb.thesis)
        return props or ["the governing legal standard"]
