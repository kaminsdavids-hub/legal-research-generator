"""is_shippable: the vacuous-truth case a live pipeline run surfaced."""

from __future__ import annotations

from legal_research.blackboard import Blackboard
from legal_research.models import Citation, CiteStatus


def _bb(*statuses: CiteStatus) -> Blackboard:
    bb = Blackboard(session_id="s", title="t")
    for i, st in enumerate(statuses):
        bb.citations.append(
            Citation(id=f"c{i}", record_id="r", citation="1 U.S. 1",
                     proposition="p", quote="q", status=st)
        )
    return bb


def test_a_paper_whose_every_citation_was_removed_is_not_shippable() -> None:
    """Observed live: 61 proposed, 61 removed, shippable=True over a
    9,653-word draft with an empty table of authorities.
    """
    assert not _bb(CiteStatus.REMOVED, CiteStatus.REMOVED, CiteStatus.REMOVED).is_shippable()


def test_one_surviving_verified_citation_is_enough() -> None:
    assert _bb(CiteStatus.REMOVED, CiteStatus.VERIFIED, CiteStatus.REMOVED).is_shippable()


def test_a_pending_citation_still_blocks() -> None:
    assert not _bb(CiteStatus.VERIFIED, CiteStatus.PENDING).is_shippable()


def test_a_needs_review_citation_still_blocks() -> None:
    assert not _bb(CiteStatus.VERIFIED, CiteStatus.NEEDS_REVIEW).is_shippable()


def test_a_paper_that_has_not_reached_the_citation_stage_is_left_alone() -> None:
    """It has not failed anything yet; this must not become a new gate on
    drafts that simply have not been researched.
    """
    assert _bb().is_shippable()
