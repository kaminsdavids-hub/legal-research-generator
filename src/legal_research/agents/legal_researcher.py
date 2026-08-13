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

#: Openers that mark a research *topic* rather than an assertable claim. The
#: Ideator emits titles -- "Analyzing the Impact of X: A Case Study" -- which are
#: serviceable retrieval queries and impossible entailment targets: asking an NLI
#: model whether a passage entails a title returns ~0 by construction. A live
#: pipeline run removed 122 of 122 citations that way, every one of them for
#: "source does not support the proposition" (REMEDIATION §22).
#: Gerund *and* imperative forms. A gerund-only list let "Analyze how model
#: weights could be treated..." through as a claim, and it became the only
#: proposition that "verified" in a live run -- a title matching a passage on
#: shared vocabulary, which is precisely what this check exists to stop.
#:
#: Written out rather than generated from stems: generating "review" + suffixes
#: produced "reviewe"/"reviewes" and never the base form, so the word the
#: Ideator actually uses was the one form not covered.
_TOPIC_OPENERS = (
    "analyze ", "analyzes ", "analyzing ", "analyse ", "analysing ",
    "investigate ", "investigates ", "investigating ",
    "explore ", "explores ", "exploring ",
    "examine ", "examines ", "examining ",
    "assess ", "assesses ", "assessing ",
    "evaluate ", "evaluates ", "evaluating ",
    "understand ", "understanding ",
    "compare ", "compares ", "comparing ",
    "revisit ", "revisiting ", "rethink ", "rethinking ",
    "consider ", "considering ", "discuss ", "discussing ",
    "review ", "reviews ", "reviewing ", "survey ", "surveying ",
    "identify ", "identifying ", "describe ", "describing ",
    "outline ", "outlining ",
    "towards", "toward", "a study", "a case study", "an analysis", "an overview",
    "the role of", "the impact of", "the case for", "the future of",
    "how ", "why ", "whether ", "what ",
)

#: Markers of a title even when it does not start with a gerund.
_TOPIC_MARKERS = (": a case study", ": an analysis", ": implications", ": a survey")


def is_assertable(text: str) -> bool:
    """Whether *text* is a claim a source could support, rather than a topic.

    Deliberately conservative: it only rejects the shapes the Ideator actually
    produces. A false negative costs a citation the thesis as its proposition,
    which is still true of the paper; a false positive puts a title back in front
    of the entailment check, which is the failure this exists to stop.
    """
    stripped = " ".join(text.strip().split()).lower()
    if not stripped:
        return False
    if stripped.startswith(_TOPIC_OPENERS):
        return False
    return all(marker not in stripped for marker in _TOPIC_MARKERS)


class LegalResearcher(Agent):
    name = "Legal Researcher"
    expert_role = "saul"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        queries: list[str] = kwargs.get("propositions") or self._default_props(ctx)

        bb = ctx.blackboard
        for query in queries:
            # The query and the proposition are not the same thing. A topic title
            # retrieves usefully but cannot be entailed by anything, and it is the
            # proposition that the Verifier later asks a source to support. When
            # the query is a title, the claim the authority is actually being
            # cited for is the paper's thesis.
            proposition = query if is_assertable(query) else (bb.thesis or query)
            hits = ctx.retriever.search(query, k=4)
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
