"""Interviewer + idea board (spec §9.1)."""

from __future__ import annotations

from legal_research.blackboard import Blackboard
from legal_research.llm.base import ChatMessage, GenerationConfig
from legal_research.llm.pool import WRITER
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


def test_interviewer_falls_back_when_llm_errors(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    class _FailingLLM:
        name = "failing"

        def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
            raise RuntimeError("upstream unavailable")

        def stream(self, messages: list[ChatMessage], config: GenerationConfig | None = None):
            return iter(())

    pipeline.pool.clients[WRITER] = _FailingLLM()

    result = pipeline.brainstorm(
        blackboard,
        scholar_input="I think Section 10(b) requires scienter.",
    )
    question = result.payload["question"]
    assert question.endswith("?")
    assert "counterargument" in question.lower()


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


# --------------------------------------------------------------------------- #
# Degraded interviews must be visible and must not loop
# --------------------------------------------------------------------------- #
class _DeadLLM:
    """Stands in for a model that is unreachable (timeout, 500, failed load)."""

    name = "dead"

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or TimeoutError("model failed to load")

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
        raise self._exc

    def stream(self, messages, config=None):  # noqa: ANN001, ANN201, ARG002
        return iter(())


class _EmptyLLM:
    """A reasoning model that spent its whole budget thinking: no visible answer."""

    name = "empty"

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
        return ""

    def stream(self, messages, config=None):  # noqa: ANN001, ANN201, ARG002
        return iter(())


def test_interviewer_reports_degradation_when_model_is_unreachable(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    """A dead model used to be indistinguishable from a working one."""

    pipeline.pool.clients[WRITER] = _DeadLLM()

    result = pipeline.brainstorm(blackboard, scholar_input="Section 10(b) needs scienter.")

    assert result.payload["question"].endswith("?")
    assert "degraded" in result.payload
    assert "unavailable" in result.payload["degraded"]


def test_interviewer_reports_degradation_on_empty_response(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    pipeline.pool.clients[WRITER] = _EmptyLLM()

    result = pipeline.brainstorm(blackboard, scholar_input="Section 10(b) needs scienter.")

    assert "degraded" in result.payload
    assert result.payload["question"].endswith("?")


def test_healthy_interview_reports_no_degradation(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    result = pipeline.brainstorm(blackboard, scholar_input="Section 10(b) needs scienter.")

    assert "degraded" not in result.payload


def test_degraded_interview_does_not_repeat_the_same_question(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    """The visible symptom of the old bug: the same canned question forever."""

    pipeline.pool.clients[WRITER] = _DeadLLM()

    questions = [
        pipeline.brainstorm(blackboard, scholar_input=f"point {i}").payload["question"]
        for i in range(4)
    ]

    assert len(set(questions)) == len(questions), f"repeated fallback questions: {questions}"


def test_state_summary_lists_already_asked_questions(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    """The model can only avoid repeats if it is told what was already asked."""

    captured: list[str] = []

    class _Recorder:
        name = "recorder"

        def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
            captured.append(messages[-1].content)
            return "What is the narrowest defensible version of the thesis?"

        def stream(self, messages, config=None):  # noqa: ANN001, ANN201, ARG002
            return iter(())

    pipeline.pool.clients[WRITER] = _Recorder()

    pipeline.brainstorm(blackboard, scholar_input="Section 10(b) needs scienter.")
    pipeline.brainstorm(blackboard, scholar_input="It also needs reliance.")

    assert "Questions already asked" in captured[-1]
    assert "narrowest defensible version" in captured[-1]


def test_degraded_flag_is_mirrored_onto_the_blackboard(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    """The brainstorm endpoint returns only the blackboard, so the flag lives there."""

    pipeline.pool.clients[WRITER] = _DeadLLM()
    pipeline.brainstorm(blackboard, scholar_input="Section 10(b) needs scienter.")

    assert blackboard.brainstorm_degraded


def test_degraded_flag_clears_once_the_model_recovers(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    pipeline.pool.clients[WRITER] = _DeadLLM()
    pipeline.brainstorm(blackboard, scholar_input="first")
    assert blackboard.brainstorm_degraded

    class _Healthy:
        name = "healthy"

        def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> str:
            return "Which element of the claim is most contested?"

        def stream(self, messages, config=None):  # noqa: ANN001, ANN201, ARG002
            return iter(())

    pipeline.pool.clients[WRITER] = _Healthy()
    pipeline.brainstorm(blackboard, scholar_input="second")

    assert blackboard.brainstorm_degraded == ""
