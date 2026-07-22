"""Verifier (spec §2.9) — the adversarial anti-hallucination gatekeeper.

For every citation it confirms the source resolves in the corpus and actually
supports the proposition (with exact quotes). Unverifiable authorities are marked
REMOVED and flagged. The paper cannot ship while any citation is unverified.
"""

from __future__ import annotations

from typing import Any

from ..models import CiteStatus
from .base import Agent, AgentContext, AgentResult


class VerifierAgent(Agent):
    name = "Verifier"
    expert_role = "saul"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        results = ctx.verifier.verify_all(bb.citations)
        bb.verifications = results

        verified = sum(1 for r in results if r.status == CiteStatus.VERIFIED)
        removed = sum(1 for r in results if r.status == CiteStatus.REMOVED)
        review = sum(1 for r in results if r.status == CiteStatus.NEEDS_REVIEW)
        bb.refresh_section_statuses()

        return AgentResult(
            agent=self.name,
            summary=(
                f"verified {verified}, removed {removed}, needs-review {review} "
                f"of {len(results)} citations; shippable={bb.is_shippable()}"
            ),
            payload={
                "verified": verified,
                "removed": removed,
                "needs_review": review,
                "shippable": bb.is_shippable(),
            },
        )
