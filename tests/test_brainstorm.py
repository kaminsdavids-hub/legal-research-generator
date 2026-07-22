"""Interviewer + idea board (spec §9.1)."""

from __future__ import annotations

from legal_research.blackboard import Blackboard
from legal_research.models import BrainstormRole, IdeaStatus
from legal_research.pipeline import LegalResearchPipeline


def test_interviewer_returns_valid_question_from_partial_state(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    result = pipeline.brainstorm(blackboard, scholar_input="I think Section 10(b) needs scienter.")
    question = result.payload["question"]
    assert question.endswith("?")
    assert len(question) > 10
    # The scholar turn and the interviewer turn are both recorded.
    roles = [t.role for t in blackboard.brainstorm]
    assert BrainstormRole.SCHOLAR in roles
    assert BrainstormRole.INTERVIEWER in roles


def test_interviewer_works_with_empty_state(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    result = pipeline.brainstorm(blackboard, scholar_input=None)
    assert result.payload["question"].endswith("?")


def test_idea_selection_flows_into_outline(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    pipeline.ideate(blackboard, seed="aiding and abetting liability under Section 10(b)")
    assert len(blackboard.ideas) >= 2

    keep = blackboard.ideas[:2]
    pipeline.select_ideas(blackboard, [(idea.id, i) for i, idea in enumerate(keep)])
    assert all(i.status is IdeaStatus.KEEP for i in blackboard.selected_ideas())

    pipeline.build_outline(blackboard)
    titles = [s.title for s in blackboard.outline]
    assert titles[0] == "Introduction"
    assert "Conclusion" in titles[-1]
    # Selected ideas produce their own sections anchored to the idea id.
    anchored = [s for s in blackboard.outline if s.idea_ids]
    assert anchored, "expected at least one section anchored to a selected idea"
