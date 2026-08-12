"""Learning which questions are worth asking. Offline: no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

from modules.maieutic.cli import main
from modules.maieutic.learn import (
    MAX_SHIFT,
    MIN_EVIDENCE,
    Episode,
    Journal,
    LearnedPolicy,
    Outcome,
    episodes_by_section,
    gap_episode,
)
from modules.maieutic.loop import Session
from modules.maieutic.socratic import PRIORITY, Gap, GapKind

THESIS = "Publishing open model weights is not a deemed export."
LONG = "The published-information exclusion removes public information from the regulation."


def _episodes(kind: GapKind, outcome: Outcome, n: int, words: int = 20) -> list[Episode]:
    return [Episode(kind, outcome, answer_words=words) for _ in range(n)]


def _journal(*groups: list[Episode]) -> Journal:
    journal = Journal()
    for group in groups:
        for episode in group:
            journal.record(episode)
    return journal


# --------------------------------------------------------------------------- #
# The gates are not the teacher
# --------------------------------------------------------------------------- #
def test_a_refused_answer_still_counts_as_a_question_worth_asking() -> None:
    """The author wrote a paragraph and the machinery rejected it. That is a
    fact about the gates, not about the question.
    """
    session = Session()
    session.begin(THESIS)
    session.ask()
    result = session.answer("It may perhaps arguably possibly seem so.")
    assert not result.merged, "the banality gate should have refused this"
    assert [e.outcome for e in session.journal.episodes] == [Outcome.ANSWERED]


def test_the_journal_records_nothing_about_gate_verdicts() -> None:
    """If a verdict ever reached the journal, the machinery would be grading its
    own curriculum (REMEDIATION §18, one layer up).
    """
    fields = set(Episode.__dataclass_fields__)
    assert fields == {"gap_kind", "outcome", "section", "answer_words"}


def test_declining_is_the_only_negative_signal() -> None:
    session = Session()
    session.begin(THESIS)
    session.ask()
    session.decline()
    assert [e.outcome for e in session.journal.episodes] == [Outcome.DECLINED]


def test_declining_with_nothing_pending_is_not_an_error() -> None:
    assert Session().decline() is None


def test_a_declined_question_is_not_asked_again() -> None:
    session = Session()
    session.begin(THESIS)
    first = session.ask()
    session.decline()
    assert session.ask() != first


# --------------------------------------------------------------------------- #
# Engagement
# --------------------------------------------------------------------------- #
def test_engagement_is_unmeasured_below_the_evidence_floor() -> None:
    """One answer is a mood. Unmeasured is not the same as zero — §11.8."""
    journal = _journal(_episodes(GapKind.ORPHANED_NODE, Outcome.ANSWERED, MIN_EVIDENCE - 1))
    assert journal.engagement(GapKind.ORPHANED_NODE) is None


def test_engagement_counts_substantive_answers_only() -> None:
    """A two-word answer is the author typing something to move on."""
    journal = _journal(
        _episodes(GapKind.ORPHANED_NODE, Outcome.ANSWERED, 2, words=40),
        _episodes(GapKind.ORPHANED_NODE, Outcome.ANSWERED, 2, words=2),
    )
    assert journal.engagement(GapKind.ORPHANED_NODE) == 0.5


def test_a_kind_never_asked_has_no_engagement_rather_than_zero() -> None:
    assert Journal().engagement(GapKind.SELF_GROUNDING) is None


# --------------------------------------------------------------------------- #
# Evidence nudges, it does not overrule
# --------------------------------------------------------------------------- #
def test_a_kind_the_author_engages_with_rises() -> None:
    journal = _journal(_episodes(GapKind.ORPHANED_NODE, Outcome.ANSWERED, 5, words=40))
    policy = LearnedPolicy(journal)
    assert policy.rank(GapKind.ORPHANED_NODE) < PRIORITY.index(GapKind.ORPHANED_NODE)


def test_a_kind_the_author_declines_falls() -> None:
    journal = _journal(_episodes(GapKind.UNANSWERED_ATTACK, Outcome.DECLINED, 5))
    policy = LearnedPolicy(journal)
    assert policy.rank(GapKind.UNANSWERED_ATTACK) > PRIORITY.index(GapKind.UNANSWERED_ATTACK)


def test_a_structural_defect_cannot_be_learned_away() -> None:
    """A DEPENDS_ON cycle is a defect whether or not the author enjoys being
    asked about it. The adjustment is bounded so evidence cannot bury it.
    """
    journal = _journal(
        _episodes(GapKind.SELF_GROUNDING, Outcome.DECLINED, 10),
        _episodes(GapKind.UNVERIFIED_AUTHORITY, Outcome.ANSWERED, 10, words=60),
    )
    policy = LearnedPolicy(journal)
    assert policy.rank(GapKind.SELF_GROUNDING) < policy.rank(GapKind.UNVERIFIED_AUTHORITY)


def test_adjacent_kinds_can_swap_on_strong_evidence() -> None:
    """Intended, not tolerated. Neighbouring priorities were close to a
    judgement call, and the author's behaviour is better evidence than the guess.
    """
    journal = _journal(_episodes(GapKind.UNANSWERED_ATTACK, Outcome.ANSWERED, 5, words=40))
    policy = LearnedPolicy(journal)
    assert policy.rank(GapKind.UNANSWERED_ATTACK) < policy.rank(GapKind.SELF_GROUNDING)


def test_evidence_cannot_travel_further_than_max_shift() -> None:
    """The declared order dominates over any distance greater than the bound:
    a cycle the author skips still outranks a citation gap they enjoy.
    """
    journal = _journal(
        _episodes(GapKind.SELF_GROUNDING, Outcome.DECLINED, 8),
        _episodes(GapKind.UNVERIFIED_AUTHORITY, Outcome.ANSWERED, 8, words=60),
    )
    policy = LearnedPolicy(journal)
    gap = PRIORITY.index(GapKind.UNVERIFIED_AUTHORITY) - PRIORITY.index(GapKind.SELF_GROUNDING)
    assert gap > 2 * MAX_SHIFT
    assert policy.rank(GapKind.SELF_GROUNDING) < policy.rank(GapKind.UNVERIFIED_AUTHORITY)


def test_the_adjustment_is_bounded_by_max_shift() -> None:
    for outcome in (Outcome.ANSWERED, Outcome.DECLINED):
        journal = _journal(_episodes(GapKind.UNCONTESTED_CLAIM, outcome, 20, words=50))
        shift = abs(
            LearnedPolicy(journal).rank(GapKind.UNCONTESTED_CLAIM)
            - PRIORITY.index(GapKind.UNCONTESTED_CLAIM)
        )
        assert shift <= MAX_SHIFT + 1e-9


def test_no_kind_is_ever_suppressed_outright() -> None:
    """A kind that stopped being asked could never earn its way back, and the
    loop would narrow to whatever the author answered first.
    """
    journal = _journal(_episodes(GapKind.ORPHANED_NODE, Outcome.DECLINED, 50))
    order = LearnedPolicy(journal).order(list(GapKind))
    assert GapKind.ORPHANED_NODE in order
    assert len(order) == len(list(GapKind))


def test_with_no_evidence_the_order_is_the_declared_priority() -> None:
    assert LearnedPolicy().order(list(PRIORITY)) == list(PRIORITY)


def test_the_policy_can_say_why_it_moved_something() -> None:
    """A policy that cannot explain itself is one nobody can argue with."""
    journal = _journal(_episodes(GapKind.ORPHANED_NODE, Outcome.ANSWERED, 4, words=40))
    lines = "\n".join(LearnedPolicy(journal).explain())
    assert "orphaned_node" in lines
    assert "100% substantive over 4 asked" in lines
    assert "not enough evidence yet" in lines


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_the_engine_orders_by_the_learned_policy() -> None:
    session = Session()
    session.begin(THESIS)
    for _ in range(MIN_EVIDENCE + 2):
        session.journal.record(
            Episode(GapKind.UNCONTESTED_CLAIM, Outcome.DECLINED)
        )
    # The only gap on a fresh thesis is UNCONTESTED_CLAIM, so it is still asked:
    # a declined kind is deprioritised, never suppressed.
    assert session.ask().gap.kind is GapKind.UNCONTESTED_CLAIM


def test_gap_episode_carries_section_and_answer_length() -> None:
    gap = Gap(GapKind.UNCONTESTED_CLAIM, ("n1",), section="Doctrine")
    episode = gap_episode(gap, Outcome.ANSWERED, "three words here")
    assert episode.section == "Doctrine"
    assert episode.answer_words == 3


def test_sections_where_the_author_actually_works_are_countable() -> None:
    journal = _journal(
        [Episode(GapKind.UNCONTESTED_CLAIM, Outcome.ANSWERED, "A", 40)],
        [Episode(GapKind.UNCONTESTED_CLAIM, Outcome.ANSWERED, "A", 40)],
        [Episode(GapKind.UNCONTESTED_CLAIM, Outcome.DECLINED, "B", 0)],
    )
    assert episodes_by_section(journal) == {"A": 2}


def test_the_journal_persists_separately_from_the_manuscript(tmp_path: Path) -> None:
    """What is learned is a fact about the author, not about one paper."""
    session = Session()
    session.begin(THESIS)
    session.ask()
    session.answer(LONG)

    path = tmp_path / "journal.json"
    session.journal.save(path)
    assert [e.to_dict() for e in Journal.load(path).episodes] == [
        e.to_dict() for e in session.journal.episodes
    ]

    # The session file carries the manuscript and not the history.
    state = tmp_path / "s.json"
    session.save(state)
    assert "journal" not in json.loads(state.read_text())


def test_evidence_accumulates_across_manuscripts(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A single offline manuscript runs dry after two or three questions, below
    the evidence floor. A per-manuscript journal would never once activate.
    """
    journal = tmp_path / "journal.json"
    for index in range(3):
        state = ["--state", str(tmp_path / f"m{index}.json"), "--journal", str(journal)]
        main([*state, "begin", f"Thesis number {index} about export control law."])
        main([*state, "skip"])
    capsys.readouterr()

    accumulated = Journal.load(journal)
    assert len(accumulated.episodes) >= 3
    assert accumulated.engagement(GapKind.UNCONTESTED_CLAIM) is not None


def test_a_missing_journal_file_starts_empty(tmp_path: Path) -> None:
    assert Journal.load(tmp_path / "absent.json").episodes == []


def test_a_journal_save_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    Journal().save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["journal.json"]


# --------------------------------------------------------------------------- #
# The CLI, invoked as a CLI
# --------------------------------------------------------------------------- #
def test_skip_records_a_decline_and_moves_on(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json"), "--journal", str(tmp_path / "j.json")]
    main([*state, "begin", THESIS])
    capsys.readouterr()

    assert main([*state, "skip"]) == 0
    assert "Skipped" in capsys.readouterr().out
    assert Journal.load(tmp_path / "j.json").episodes[0].outcome is Outcome.DECLINED


def test_skip_with_nothing_pending_says_so(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json"), "--journal", str(tmp_path / "j.json")]
    assert main([*state, "skip"]) == 0
    assert "No question is pending" in capsys.readouterr().out


def test_policy_explains_itself_and_names_its_signal(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json"), "--journal", str(tmp_path / "j.json")]
    main([*state, "begin", THESIS])
    main([*state, "answer", LONG])
    capsys.readouterr()

    assert main([*state, "policy"]) == 0
    out = capsys.readouterr().out
    assert "uncontested_claim" in out
    assert "never from whether the gates accepted it" in out


def test_policy_on_a_fresh_session_says_nothing_is_learned(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json"), "--journal", str(tmp_path / "j.json")]
    main([*state, "begin", THESIS])
    capsys.readouterr()
    main([*state, "policy"])
    assert "nothing learned" in capsys.readouterr().out
