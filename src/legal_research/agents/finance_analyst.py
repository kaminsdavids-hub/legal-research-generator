"""Finance Analyst (spec §2.4).

Supplies banking/finance substance and ties financial concepts to the legal
argument. Any finance authority (e.g. a Basel framework document) still enters only
through retrieval; grounded citations are created by the Writer at draft time.
"""

from __future__ import annotations

from typing import Any

from ..citations.verifier import support_score
from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import GEMMA
from ..models import Authority, Relation, SourceType
from .base import Agent, AgentContext, AgentResult
from .legal_researcher import SUPPORT_CUTOFF

_SYSTEM = (
    "TASK: generic\n"
    "You are a finance analyst supporting a legal argument. Explain the relevant "
    "banking/finance mechanics precisely and connect them to the legal question."
)


class FinanceAnalyst(Agent):
    name = "Finance Analyst"
    expert_role = GEMMA

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        topics: list[str] = kwargs.get("topics") or self._default_topics(ctx)
        bb = ctx.blackboard
        notes: list[str] = []

        client = ctx.pool.get(self.expert_role)
        for topic in topics:
            hits = ctx.retriever.search(topic, k=3)
            finance_hits = [
                h
                for h in hits
                if (rec := ctx.corpus.get(h.record_id)) and rec.type is SourceType.SECONDARY
            ]
            for h in finance_hits:
                record = ctx.corpus.get(h.record_id)
                if record is None:
                    continue
                if not any(p.record_id == h.record_id for p in bb.retrieved):
                    bb.retrieved.append(h)
                relation = (
                    Relation.SUPPORTING
                    if support_score(topic, h.text) >= SUPPORT_CUTOFF
                    else Relation.CONTRARY
                )
                bb.authorities.append(
                    Authority(
                        record_id=record.id,
                        citation=ctx.formatter.full(record),
                        relation=relation,
                        proposition=topic,
                        passage=h.text,
                    )
                )

            substance = client.chat(
                [ChatMessage("system", _SYSTEM), ChatMessage("user", topic)],
                DecodingPolicy.BALANCED.config,
            )
            notes.append(substance)

        return AgentResult(
            agent=self.name,
            summary=f"analyzed {len(topics)} finance topics",
            payload={"notes": notes},
        )

    def _default_topics(self, ctx: AgentContext) -> list[str]:
        bb = ctx.blackboard
        finance_terms = ("capital", "leverage", "collateral", "basel", "bank", "liquidity")
        topics = [
            i.text for i in bb.selected_ideas() if any(t in i.text.lower() for t in finance_terms)
        ]
        return topics
