"""The cycle and the CLI. Offline: no models, no network.

Every CLI test invokes `main()` with real argv. A tool whose parser is never
exercised is a tool that can be broken for every invocation while the suite
stays green — which is exactly what happened in REMEDIATION §11.11c.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.maieutic.cli import build_parser, main
from modules.maieutic.graph import Edge, EdgeType, GraphPatch, Node, NodeType
from modules.maieutic.loop import Gates, Session
from modules.maieutic.socratic import GapKind

THESIS = "Publishing open model weights is not a deemed export."
ANSWER = "The published exclusion removes public information from the regulation."


def _started() -> Session:
    session = Session()
    session.begin(THESIS)
    return session


# --------------------------------------------------------------------------- #
# The cycle
# --------------------------------------------------------------------------- #
def test_a_manuscript_opens_with_the_authors_own_thesis() -> None:
    session = _started()
    node = next(iter(session.graph.nodes.values()))
    assert node.text == THESIS
    assert node.provenance.value == "human"


def test_a_manuscript_cannot_be_started_twice() -> None:
    with pytest.raises(ValueError):
        _started().begin("A second thesis.")


def test_the_first_question_is_about_the_untested_thesis() -> None:
    assert _started().ask().gap.kind is GapKind.UNCONTESTED_CLAIM


def test_an_answer_that_passes_every_gate_merges() -> None:
    session = _started()
    session.ask()
    result = session.answer(ANSWER)
    assert result.merged
    assert len(result.added) == 1
    assert result.report and result.report.passed


def test_answering_without_a_pending_question_is_an_error() -> None:
    with pytest.raises(ValueError):
        _started().answer(ANSWER)


def test_a_refused_patch_leaves_the_question_pending() -> None:
    """A refusal means the author has not yet answered, not that the gap is dealt
    with.
    """
    session = _started()
    question = session.ask()
    result = session.answer("It may perhaps arguably possibly seem so.")
    assert not result.merged
    assert session.pending == question
    assert not session.asked.has_asked(question.gap)


def test_every_refusal_is_reported_not_just_the_first() -> None:
    """Fix one, resubmit, discover the next is how a loop wastes the author's
    attention — the one thing this system treats as scarce.
    """
    session = _started()
    session.ask()
    result = session.answer("It could perhaps arguably seem possible.")
    assert result.refusals
    assert any("banality" in r for r in result.refusals)


def test_a_merged_question_is_not_asked_again() -> None:
    session = _started()
    first = session.ask()
    session.answer(ANSWER)
    assert session.ask() != first


#: Genuinely distinct answers. Near-paraphrases are correctly refused by the
#: novelty gate, so a loop test fed on them stalls rather than running dry —
#: which is the gate working, not the loop failing.
ANSWERS = [
    "The published exclusion removes public information from the regulation.",
    "Weights encode human choices, and choosing among expressions is expressive.",
    "Functionality has never defeated protection for written technical material.",
    "Congress legislated against a backdrop assuming physical shipment of goods.",
    "No deemed-export rule has ever been applied to a public internet posting.",
    "The agency itself disclaimed authority over published research in guidance.",
]


def test_the_loop_runs_dry_rather_than_inventing_questions() -> None:
    session = _started()
    for answer in ANSWERS:
        if session.ask() is None:
            break
        session.answer(answer)
    assert session.ask() is None


def test_an_unresolvable_gap_says_the_answer_did_not_close_it() -> None:
    session = Session()
    a = Node.from_human(NodeType.ORIGINAL, "Claim A stands on B.")
    b = Node.from_human(NodeType.ORIGINAL, "Claim B stands on A.")
    session.graph.apply(
        GraphPatch(
            nodes=[a, b],
            edges=[
                Edge(a.id, b.id, EdgeType.DEPENDS_ON),
                Edge(b.id, a.id, EdgeType.DEPENDS_ON),
            ],
        )
    )
    assert session.ask().gap.kind is GapKind.SELF_GROUNDING
    assert session.answer("Claim B is the prior one, on reflection.").unresolved


# --------------------------------------------------------------------------- #
# Gate ordering and reporting
# --------------------------------------------------------------------------- #
def test_offline_grounding_fails_closed_on_an_authority() -> None:
    """Nothing offline can confirm a citation, and merging one on trust is the
    failure the gate exists to prevent.
    """
    session = _started()
    thesis = next(iter(session.graph.nodes))
    authority = Node.propose(NodeType.AUTHORITY, "A holding.", citation="1 U.S. 1")
    patch = GraphPatch(
        nodes=[authority], edges=[Edge(authority.id, thesis, EdgeType.SUPPORTS)]
    )
    report = Gates.offline().check(patch, session.graph)
    assert not report.passed
    assert any("grounding" in r for r in report.refusals)


def test_advisories_are_reported_without_blocking() -> None:
    session = _started()
    session.ask()
    result = session.answer("This may perhaps arguably be the generally better view.")
    assert result.merged
    assert result.report and result.report.advisories


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_a_session_round_trips_through_disk(tmp_path: Path) -> None:
    session = _started()
    question = session.ask()
    session.answer(ANSWER)
    session.ask()
    path = tmp_path / "s.json"
    session.save(path)

    loaded = Session.load(path)
    assert loaded.graph.to_dict() == session.graph.to_dict()
    assert loaded.asked.keys == session.asked.keys
    assert loaded.pending is not None
    assert loaded.pending.text == session.pending.text
    assert loaded.asked.has_asked(question.gap)


def test_loading_a_missing_file_starts_a_fresh_session(tmp_path: Path) -> None:
    assert Session.load(tmp_path / "absent.json").graph.nodes == {}


def test_a_save_does_not_leave_a_temp_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _started().save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


# --------------------------------------------------------------------------- #
# The CLI, invoked as a CLI
# --------------------------------------------------------------------------- #
def test_the_parser_accepts_every_advertised_subcommand() -> None:
    parser = build_parser()
    for argv in (
        ["begin", "A thesis."],
        ["ask"],
        ["answer", "An answer."],
        ["cycle"],
        ["cycle", "--rounds", "3"],
        ["status"],
        ["render"],
        ["render", "--review"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_a_missing_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_begin_then_answer_then_render(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The whole loop through argv, which is the only way this stays honest."""
    state = ["--state", str(tmp_path / "s.json")]
    assert main([*state, "begin", THESIS]) == 0
    assert main([*state, "answer", ANSWER]) == 0
    capsys.readouterr()

    assert main([*state, "render"]) == 0
    assert THESIS in capsys.readouterr().out


def test_beginning_twice_fails_rather_than_overwriting(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json")]
    main([*state, "begin", THESIS])
    assert main([*state, "begin", "A different thesis."]) == 1
    capsys.readouterr()

    main([*state, "render"])
    assert "A different thesis." not in capsys.readouterr().out


def test_the_cli_reports_a_refusal_rather_than_merging(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json")]
    main([*state, "begin", THESIS])
    capsys.readouterr()

    main([*state, "answer", "It may perhaps arguably possibly seem so."])
    out = capsys.readouterr().out
    assert "Not merged" in out
    assert "banality" in out


def test_status_names_the_open_gaps(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json")]
    main([*state, "begin", THESIS])
    capsys.readouterr()

    main([*state, "status"])
    assert GapKind.UNCONTESTED_CLAIM.value in capsys.readouterr().out


def test_render_review_annotates_provenance(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json")]
    main([*state, "begin", THESIS])
    capsys.readouterr()

    main([*state, "render", "--review"])
    assert "`[human]`" in capsys.readouterr().out


def test_cycle_reads_answers_from_stdin_and_stops_on_a_blank_line(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    state = ["--state", str(tmp_path / "s.json")]
    main([*state, "begin", THESIS])

    answers = iter([ANSWER, ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert main([*state, "cycle"]) == 0
    assert "Merged" in capsys.readouterr().out

    assert len(Session.load(tmp_path / "s.json").graph.nodes) == 2


def test_cycle_saves_every_round_not_only_at_the_end(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """A crash mid-session must not cost the author the answers they gave."""
    path = tmp_path / "s.json"
    state = ["--state", str(path)]
    main([*state, "begin", THESIS])

    def _answer_then_crash(_: str = "") -> str:
        if Session.load(path).graph.nodes.__len__() > 1:
            raise KeyboardInterrupt
        return ANSWER

    monkeypatch.setattr("builtins.input", _answer_then_crash)
    with pytest.raises(KeyboardInterrupt):
        main([*state, "cycle"])
    capsys.readouterr()

    assert len(Session.load(path).graph.nodes) == 2
