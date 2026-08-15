"""Section headings: an idea is a sentence, a heading is a noun phrase.

Every case here is a title a live run actually produced. Truncating the idea at
70 characters gave headings that were sentences, that carried the model's
scaffolding ("**TOPIC:**"), and — the one that matters — that cited a provision
which does not exist. Nothing downstream verifies a heading: the CitationGuard
grounds body prose, the Verifier re-checks it, and a title is touched by
neither, so it reaches the PDF and the table of contents unchallenged.
"""

from __future__ import annotations

import types

import pytest

from legal_research.agents.argument_architect import (
    FALLBACK_HEADING,
    MAX_HEADING_WORDS,
    ArgumentArchitect,
    _corpus_knows,
    heading_from,
)
from legal_research.citations.corpus import load_corpus

CORPUS_PATH = "data/corpus/sample_corpus.jsonl"


@pytest.fixture
def known():
    corpus = load_corpus(CORPUS_PATH)
    return _corpus_knows(types.SimpleNamespace(corpus=corpus))


# --------------------------------------------------------------------------- #
# Sentences become noun phrases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("idea", "expected"),
    [
        # Live run 4: the model's scaffolding shipped as the heading.
        ("**TOPIC: AI Algorithmic Protection**", "AI Algorithmic Protection"),
        ("III. Topic: Transparency in Legal Systems", "Transparency in Legal Systems"),
        # Live run 4: a whole sentence, cut mid-word at 70 characters.
        (
            "Courts have consistently ruled that publishing source codes and underlying "
            "intellectual property does not qualify.",
            "Courts have consistently ruled",
        ),
        # "and" must not break a noun phrase: this used to yield "The strongest textual".
        (
            "The strongest textual and precedential objection to open-weight publication, "
            "which the agency raises.",
            "The strongest textual and precedential objection to open-weight publication",
        ),
    ],
)
def test_a_sentence_becomes_a_heading(idea: str, expected: str) -> None:
    assert heading_from(idea) == expected


def test_a_heading_is_never_longer_than_the_cap() -> None:
    idea = " ".join(f"word{i}" for i in range(40))
    assert len(heading_from(idea).split()) <= MAX_HEADING_WORDS


def test_nothing_usable_falls_back_rather_than_inventing() -> None:
    assert heading_from("   ") == FALLBACK_HEADING
    assert heading_from("") == FALLBACK_HEADING


# --------------------------------------------------------------------------- #
# Citations: the place with no guard
# --------------------------------------------------------------------------- #
def test_a_section_citation_never_reaches_a_heading() -> None:
    """Live run 4 produced "II. Publishing AI model weights may be subject to
    EAR § 3599.7(b)(4)..." — a provision that does not exist, in a heading no
    stage verifies."""

    heading = heading_from(
        "Publishing AI model weights may be subject to EAR § 3599.7(b)(4) restrictions."
    )
    assert "§" not in heading
    assert "3599" not in heading
    assert heading == "Publishing AI model weights"


@pytest.mark.parametrize(
    "idea",
    [
        "Weights are protected under 17 U.S.C. § 106 as expression.",
        "The rule in 550 U.S. 544 governs the pleading standard here.",
        "See 17 C.F.R. § 240.10b-5 for the operative prohibition.",
    ],
)
def test_no_reporter_or_code_citation_survives(idea: str) -> None:
    heading = heading_from(idea)
    assert not any(token in heading for token in ("§", "U.S.C.", "C.F.R.", "550 U.S."))


def test_a_case_the_corpus_holds_may_stay(known) -> None:
    """Naming an authority the paper actually has is ordinary practice."""

    corpus = load_corpus(CORPUS_PATH)
    case = next(r for r in corpus.records if " v. " in r.title)
    left = case.title.split(" v. ")[0].split()[-1]
    right = case.title.split(" v. ")[1].split()[0].strip(".,;:")

    heading = heading_from(f"{left} v. {right} controls because the text is expressive.", known)
    assert f"{left} v. {right}" in heading


def test_a_case_the_corpus_does_not_hold_is_dropped(known) -> None:
    heading = heading_from(
        "The First Amendment protects code, as Imaginary v. Defendant held.", known
    )
    assert "Imaginary" not in heading
    assert heading == "The First Amendment protects code"


def test_dropping_an_unknown_case_does_not_leave_debris(known) -> None:
    """Excising it mid-sentence produced "In re Corp. controls the analysis
    here", which reads like a heading and is not one."""

    heading = heading_from(
        "In re Seagate Tech. v. Fabricated Corp. controls the analysis here.", known
    )
    assert heading == FALLBACK_HEADING


def test_without_a_corpus_predicate_every_case_name_is_dropped() -> None:
    """The safe default: unverifiable unless something says otherwise."""

    assert heading_from("Junger v. Daley controls here, plainly.") == FALLBACK_HEADING


# --------------------------------------------------------------------------- #
# Through the agent
# --------------------------------------------------------------------------- #
def test_the_architect_numbers_and_cleans_titles(pipeline, blackboard) -> None:
    bb = blackboard
    bb.add_idea(text="**TOPIC: AI Algorithmic Protection**")
    bb.add_idea(
        text="Publishing AI model weights may be subject to EAR § 3599.7(b)(4) restrictions."
    )
    for idea in bb.ideas:
        bb.set_idea_status(idea.id, type(bb.ideas[0]).model_fields["status"].default)

    ArgumentArchitect().act(pipeline.context(bb))
    titles = [s.title for s in bb.outline]

    assert titles[0] == "Introduction"
    assert titles[-2:] == ["Counterargument and Rebuttal", "Conclusion"]
    assert not any("§" in t for t in titles)
    assert not any("**" in t for t in titles)


def test_a_fallback_idea_takes_a_noun_phrase_not_a_thesis(pipeline, blackboard) -> None:
    """The Ideator returned 0 ideas on a live run, so these templates ran. They
    interpolated the scholar's whole thesis, producing an idea that was not
    grammatical, a heading built from it, and a retrieval proposition that
    asserted two things at once."""

    thesis = (
        "Publishing open model weights is protected expression, and the Export "
        "Administration Regulations may not treat that publication as a deemed export."
    )
    pipeline._seed_fallback_ideas(blackboard, thesis, 3)

    assert blackboard.ideas
    for idea in blackboard.ideas:
        assert "Administration Regulations" not in idea.text
        assert "," not in idea.text  # one claim, not two joined by a comma
        assert len(idea.text.split()) <= 20
    assert blackboard.ideas[0].text.endswith("Publishing open model weights")
