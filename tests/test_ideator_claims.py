"""The Ideator emits claims, not just topics. REMEDIATION §25."""

from __future__ import annotations

from legal_research.agents.ideator import _split_topic_and_claim
from legal_research.agents.legal_researcher import is_assertable
from legal_research.config import Settings
from legal_research.models import Idea
from legal_research.pipeline import LegalResearchPipeline

RAW_IDEA = (
    "Private plaintiffs may not maintain aiding-and-abetting suits under Section 10(b), "
    "and primary liability requires a showing of scienter."
)


def test_a_bullet_carries_a_topic_and_a_claim() -> None:
    topic, claim = _split_topic_and_claim(
        "Analyze weights || Model weights are published information under the EAR."
    )
    assert topic == "Analyze weights"
    assert claim.startswith("Model weights are published")


def test_a_bullet_without_the_separator_still_yields_a_topic() -> None:
    """A generator that ignores the format must not lose the idea entirely."""
    topic, claim = _split_topic_and_claim("A topic with no claim offered")
    assert topic == "A topic with no claim offered"
    assert claim == ""


def test_an_idea_defaults_to_no_claim() -> None:
    """Absent is the honest state; the caller falls back to the thesis."""
    assert Idea(id="i", text="t").claim == ""


def test_every_proposition_put_to_a_source_is_a_claim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The point of the whole §21-§25 sequence. A source can only support or
    contradict an assertion; asked to entail a title it returns ~0, which is how
    a live run removed 122 of 122 citations.
    """
    settings = Settings(llm_mode="mock", corpus_path="data/corpus/sample_corpus.jsonl")
    bb = LegalResearchPipeline(settings).run_all(RAW_IDEA, title="T").blackboard

    propositions = {c.proposition for c in bb.citations}
    assert propositions, "the pipeline produced no citations to check"
    unassertable = [p for p in propositions if not is_assertable(p)]
    assert not unassertable, f"topics reached the verifier: {unassertable[:2]}"


def test_the_ideator_gives_every_idea_a_claim_when_the_generator_cooperates() -> None:
    settings = Settings(llm_mode="mock", corpus_path="data/corpus/sample_corpus.jsonl")
    bb = LegalResearchPipeline(settings).run_all(RAW_IDEA, title="T").blackboard
    assert bb.ideas
    assert all(i.claim for i in bb.ideas), "every mock idea should carry a claim"
    assert all(is_assertable(i.claim) for i in bb.ideas)
