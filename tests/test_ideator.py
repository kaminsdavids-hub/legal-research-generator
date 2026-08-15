"""Reading the Ideator's reply, whatever list style it arrived in.

Every reply shape here was produced by the live ideation role (apertus:latest,
HOT decoding) on one thesis across eight calls. Three of those eight returned
every idea under a prefix other than ``- `` and were read as zero ideas; the
pipeline then substituted its deterministic fallback ideas, and the run reported
"generated 0 candidate ideas" — which reads like a model with nothing to say
rather than 4,206 characters of well-formed output going in the bin.
"""

from __future__ import annotations

import pytest

from legal_research.agents.ideator import Ideator, parse_ideas

# --- reply shapes observed live -------------------------------------------- #
HYPHEN = (
    "- Legal challenges to restricting access to models trained abroad || "
    "Restrictions on AI research infringe information rights.\n"
    "- The role of cryptography in protecting model weights || Model owners must "
    "encrypt weights to prevent misuse.\n"
)
NUMBERED = (
    "1. TOPIC || The legal classification of publication under the EAR is unsettled.\n"
    "2. TOPIC || The exemption of model weight publication follows from precedent.\n"
)
HEADED = (
    "### TOPIC || The publication of open-source AI models is protected expression.\n"
    "This paper contends that the dissemination of model weights is speech.\n"
)
BARE = (
    "TOPIC || The publishing of open-source models, such as weights, is lawful.\n"
    "Note: Legal frameworks vary internationally and are subject to change.\n"
)
CLAIM_ONLY = (
    "|| CLAIM: An act of publishing trained machine learning weights is publication.\n"
    "|| CLAIM: The publication of open-source software is protected expression.\n"
)
MIXED_WITH_PROSE = (
    "Here are several candidate angles for your paper:\n\n"
    "- Weights as published information under the EAR || Model weights published "
    "openly are published information and fall outside the deemed-export rule.\n"
    "**Reference:** Recent updates to EAR and scholarly commentary.\n"
)


@pytest.mark.parametrize(
    ("raw", "count"),
    [(HYPHEN, 2), (NUMBERED, 2), (HEADED, 1), (BARE, 1), (CLAIM_ONLY, 2), (MIXED_WITH_PROSE, 1)],
)
def test_every_observed_list_style_is_read(raw: str, count: int) -> None:
    assert len(parse_ideas(raw)) == count


def test_prose_around_the_list_is_not_mistaken_for_an_idea() -> None:
    topics = [text for text, _ in parse_ideas(MIXED_WITH_PROSE)]
    assert topics == ["Weights as published information under the EAR"]
    assert not any("Reference" in t for t in topics)
    assert not any("Here are several" in t for t in topics)


def test_a_note_line_is_not_an_idea() -> None:
    topics = [text for text, _ in parse_ideas(BARE)]
    assert not any(t.startswith("Legal frameworks vary") for t in topics)


def test_an_echoed_placeholder_yields_the_claim_as_the_topic() -> None:
    """"TOPIC || <claim>" means the model reused the format's own word."""

    (text, claim), = parse_ideas(BARE)
    assert text == claim
    assert text.startswith("The publishing of open-source models")


def test_a_claim_only_line_is_kept() -> None:
    ideas = parse_ideas(CLAIM_ONLY)
    assert all(text and text == claim for text, claim in ideas)
    assert "CLAIM:" not in ideas[0][0]


def test_the_separator_is_still_split_when_present() -> None:
    (text, claim), _ = parse_ideas(HYPHEN)
    assert text == "Legal challenges to restricting access to models trained abroad"
    assert claim == "Restrictions on AI research infringe information rights."


def test_bold_markup_is_stripped() -> None:
    (text, claim), = parse_ideas("- **Weights as speech** || **Weights are expression.**")
    assert text == "Weights as speech"
    assert claim == "Weights are expression."


# --------------------------------------------------------------------------- #
# The agent must say which failure happened
# --------------------------------------------------------------------------- #
class _Reply:
    name = "hermes"

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages, config=None) -> str:  # noqa: ANN001 - test double
        return self._text


def _run(pipeline, blackboard, reply: str):
    ctx = pipeline.context(blackboard)
    ctx.pool.clients["hermes"] = _Reply(reply)
    return Ideator().act(ctx)


def test_an_unreadable_reply_is_reported_as_a_parse_failure(pipeline, blackboard) -> None:
    """Not as "generated 0 candidate ideas", which blames the model and sends
    the reader looking in the wrong place."""

    result = _run(pipeline, blackboard, "I think there are several interesting angles here.")

    assert "PARSE FAILURE" in result.summary
    assert result.payload["unparsed_reply_chars"] > 0
    assert blackboard.ideas == []


def test_an_empty_reply_is_reported_as_an_empty_reply(pipeline, blackboard) -> None:
    result = _run(pipeline, blackboard, "   ")

    assert "empty reply" in result.summary
    assert result.payload["unparsed_reply_chars"] == 0


def test_a_readable_reply_creates_ideas(pipeline, blackboard) -> None:
    result = _run(pipeline, blackboard, NUMBERED)

    assert len(blackboard.ideas) == 2
    assert "PARSE FAILURE" not in result.summary
    assert result.payload["idea_ids"]


def test_the_format_echoed_back_is_not_an_idea() -> None:
    """A model that repeats "TOPIC || CLAIM" has produced no idea; the old
    parser filed the word "TOPIC" as a candidate."""

    assert parse_ideas("- TOPIC || CLAIM") == []
    assert parse_ideas("- TOPIC ||") == []
    assert parse_ideas("1. **TOPIC** || **CLAIM**") == []


# --------------------------------------------------------------------------- #
# Claim recovery: a topic must not reach the retriever as a proposition
# --------------------------------------------------------------------------- #
TOPICS_ONLY = (
    "- Legal Challenges of Publishing Open-Source AI Models Under the EAR\n"
    "- Economic Implications of Forcing Developers to Publish Source Code\n"
)
RECOVERED = (
    "1. Open-weight publication is published information and falls outside the "
    "deemed-export rule.\n"
    "2. Compelled source disclosure raises the marginal cost of compliance above "
    "the licensing threshold.\n"
)


class _TwoReplies:
    """First call is the idea list, second is the claim-recovery pass."""

    name = "hermes"

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages, config=None) -> str:  # noqa: ANN001 - test double
        self.calls += 1
        return self._replies.pop(0) if self._replies else ""


def test_missing_claims_are_recovered_on_a_second_pass(pipeline, blackboard) -> None:
    """A live run produced 10 ideas, 0 claims, and 0 of 13 citations survived:
    every proposition put to the retriever was a title."""

    ctx = pipeline.context(blackboard)
    ctx.pool.clients["hermes"] = _TwoReplies(TOPICS_ONLY, RECOVERED)
    result = Ideator().act(ctx)

    assert result.payload["recovered_claims"] == 2
    assert "recovered on a second pass" in result.summary
    assert all(idea.claim for idea in blackboard.ideas)
    assert blackboard.ideas[0].claim.startswith("Open-weight publication is published")


def test_recovery_is_skipped_when_every_idea_has_a_claim(pipeline, blackboard) -> None:
    ctx = pipeline.context(blackboard)
    client = _TwoReplies(HYPHEN, RECOVERED)
    ctx.pool.clients["hermes"] = client
    Ideator().act(ctx)

    assert client.calls == 1  # no second call was made


def test_an_unassertable_recovered_claim_is_discarded(pipeline, blackboard) -> None:
    """A bad claim is worse than none: the empty case has a documented fallback,
    a title silently becomes the thing a source is asked to support."""

    ctx = pipeline.context(blackboard)
    ctx.pool.clients["hermes"] = _TwoReplies(
        TOPICS_ONLY,
        "1. An analysis of how weights are treated under the EAR\n"
        "2. The economic implications of open weights\n",
    )
    result = Ideator().act(ctx)

    assert result.payload["recovered_claims"] == 0
    assert all(not idea.claim for idea in blackboard.ideas)


def test_a_failed_recovery_call_does_not_fail_ideation(pipeline, blackboard) -> None:
    class _Broken:
        name = "hermes"
        calls = 0

        def chat(self, messages, config=None):  # noqa: ANN001 - test double
            self.calls += 1
            if self.calls == 1:
                return TOPICS_ONLY
            raise RuntimeError("endpoint down")

    ctx = pipeline.context(blackboard)
    ctx.pool.clients["hermes"] = _Broken()
    result = Ideator().act(ctx)

    assert len(blackboard.ideas) == 2
    assert result.payload["recovered_claims"] == 0


def test_recovery_can_be_switched_off(pipeline, blackboard) -> None:
    ctx = pipeline.context(blackboard)
    ctx.settings.ideator_claim_recovery_enabled = False
    client = _TwoReplies(TOPICS_ONLY, RECOVERED)
    ctx.pool.clients["hermes"] = client
    Ideator().act(ctx)

    assert client.calls == 1


# --------------------------------------------------------------------------- #
# Bare headings are not claims
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "heading",
    [
        "Conclusion",
        "Introduction",
        "Counterargument and Rebuttal",
        "Background and Governing Standard",
        "Analysis",
        "conclusion.",
    ],
)
def test_a_bare_section_heading_is_not_assertable(heading: str) -> None:
    """Run 7 filed three of thirteen citations against the literal string
    "Conclusion": a section with no ideas took its own title as the point its
    paragraphs were cited for."""

    from legal_research.agents.legal_researcher import is_assertable

    assert is_assertable(heading) is False


@pytest.mark.parametrize(
    "claim",
    [
        "Model weights are published information",
        "Open-weight publication is protected expression under the First Amendment.",
        "Restrictions on AI research infringe information rights.",
    ],
)
def test_a_short_real_claim_still_passes(claim: str) -> None:
    from legal_research.agents.legal_researcher import is_assertable

    assert is_assertable(claim) is True


def test_a_fragment_is_not_a_claim() -> None:
    from legal_research.agents.legal_researcher import is_assertable

    assert is_assertable("Open weights") is False
    assert is_assertable("The EAR") is False
