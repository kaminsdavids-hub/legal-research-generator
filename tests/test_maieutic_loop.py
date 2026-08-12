"""The cycle and the CLI. Offline: no models, no network.

Every CLI test invokes `main()` with real argv. A tool whose parser is never
exercised is a tool that can be broken for every invocation while the suite
stays green — which is exactly what happened in REMEDIATION §11.11c.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.dialectic.models import CitationSlot, DialecticTurn, Position, SlotStatus
from modules.maieutic.cli import build_parser, main
from modules.maieutic.graph import Edge, EdgeType, GraphPatch, Node, NodeType
from modules.maieutic.loop import Gates, Session
from modules.maieutic.service import debate_prompt
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
# The live dialectic exchange
# --------------------------------------------------------------------------- #
def _turn(thesis: str, antithesis: str) -> DialecticTurn:
    return DialecticTurn(
        question="q",
        thesis=Position(
            side="thesis",
            model="m1",
            family="f1",
            propositions=[CitationSlot(proposition=thesis)],
        ),
        antithesis=Position(
            side="antithesis",
            model="m2",
            family="f2",
            propositions=[CitationSlot(proposition=antithesis)],
        ),
        synthesis="The machine's reconciliation of the two.",
    )


def test_an_exchange_adds_the_machines_pressure_around_the_answer() -> None:
    session = _started()
    session.ask()
    result = session.answer(
        ANSWER,
        turns=lambda q, a: _turn(
            "Export control has always excluded published technical data.",
            "Regulators have narrowed that exclusion for dual-use software.",
        ),
    )
    assert result.merged
    types = {session.graph.nodes[i].type for i in result.added}
    assert NodeType.OBJECTION in types, "the antithesis must arrive as an objection"
    assert len(result.added) == 3


def test_the_synthesis_does_not_reach_the_manuscript_through_the_loop() -> None:
    session = _started()
    session.ask()
    session.answer(ANSWER, turns=lambda q, a: _turn("Support.", "Objection."))
    assert "reconciliation" not in session.manuscript().text


def test_a_failed_exchange_costs_the_pressure_never_the_answer() -> None:
    """The author's work is not hostage to a model being down."""

    def _down(question: str, answer: str) -> DialecticTurn:
        raise RuntimeError("connection refused")

    session = _started()
    session.ask()
    result = session.answer(ANSWER, turns=_down)
    assert result.merged
    assert result.added, "the author's own answer still merged"
    assert "connection refused" in result.turn_error


def test_a_successful_exchange_reports_no_error() -> None:
    session = _started()
    session.ask()
    result = session.answer(ANSWER, turns=lambda q, a: _turn("Support.", "Objection."))
    assert result.turn_error == ""


def test_the_machine_cannot_merge_the_authors_answer_back_at_them() -> None:
    """A thesis that parrots the answer is compared against it and refused."""
    session = _started()
    session.ask()
    result = session.answer(ANSWER, turns=lambda q, a: _turn(a, "A real objection."))
    assert result.report is not None
    echoed = [
        v for v in result.report.novelty.verdicts if not v.accepted
    ]
    assert echoed, "restating the author's answer must not pass novelty"


def test_the_debate_prompt_puts_the_answer_under_test_not_the_question() -> None:
    """Sending the question would have the models debate the topic in general."""
    prompt = debate_prompt("What is the strongest objection?", ANSWER)
    assert ANSWER in prompt
    assert prompt.index(ANSWER) < prompt.index("strongest objection")


def _cited_turn() -> DialecticTurn:
    """A turn whose thesis cites an authority nothing can confirm."""
    return DialecticTurn(
        question="q",
        thesis=Position(
            side="thesis",
            model="m1",
            family="f1",
            propositions=[
                CitationSlot(
                    proposition="Published technical data has always been excluded.",
                    status=SlotStatus.VERIFIED,
                    normalized_cite="445 U.S. 222",
                )
            ],
        ),
        antithesis=Position(side="antithesis", model="m2", family="f2"),
    )


def test_the_machines_bad_citation_does_not_cost_the_author_their_answer() -> None:
    """Grounding is all-or-nothing across a patch, and in the assembled loop
    that meant the party who cites badly is the machine and the party who loses
    their work is the author. Every live exchange failed this way
    (REMEDIATION §12.3).
    """
    session = _started()
    session.ask()
    result = session.answer(ANSWER, turns=lambda q, a: _cited_turn())
    assert result.merged, "the author's answer must survive the machine's bad cite"
    assert result.dropped, "and the ungrounded node must be named, not silently lost"
    assert ANSWER in session.manuscript().text


def test_nothing_ungrounded_reaches_the_manuscript() -> None:
    """The invariant that actually matters is preserved by dropping, not by
    refusing the whole patch.
    """
    session = _started()
    session.ask()
    session.answer(ANSWER, turns=lambda q, a: _cited_turn())
    assert "445 U.S. 222" not in session.manuscript().text
    assert not [n for n in session.graph.nodes.values() if n.type is NodeType.AUTHORITY]


def test_atomicity_gives_way_only_across_parties() -> None:
    """If a node the author wrote fails grounding, the patch still fails whole."""
    session = Session()
    floating = Node.from_human(NodeType.ORIGINAL, "An unsupported position.")
    session.graph.apply(GraphPatch(nodes=[Node.from_human(NodeType.THESIS, "Seed.")]))

    patch = GraphPatch(nodes=[floating])
    report = Gates.offline().check(patch, session.graph)
    assert not report.passed
    assert any("unargued_original" in r for r in report.refusals)


def test_a_dropped_node_is_reported_by_the_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    session = _started()
    session.ask()
    result = session.answer(ANSWER, turns=lambda q, a: _cited_turn())
    from modules.maieutic.cli import _report

    _report(result)
    assert "could not be grounded" in capsys.readouterr().out


def test_live_gates_can_confirm_what_a_live_exchange_produces() -> None:
    """A live run retrieves and verifies citations, so it produces AUTHORITY
    nodes. Offline gates have no verifier and refuse every one of them at the
    fabrication wall — for want of a verifier, not for want of an authority.
    The two halves must be configured together (REMEDIATION §12.2).
    """
    from legal_research.config import get_settings

    session = _started()
    thesis = next(iter(session.graph.nodes))
    authority = Node.propose(
        NodeType.AUTHORITY, "Source code is protected expression.", citation="1 U.S. 1"
    )
    patch = GraphPatch(
        nodes=[authority], edges=[Edge(authority.id, thesis, EdgeType.SUPPORTS)]
    )

    offline = Gates.offline().check(patch, session.graph)
    assert any("no verifier configured" in r for r in offline.refusals)

    live = Gates.live(get_settings())
    assert live.grounding.verifier is not None, "a live run must be able to confirm"


def test_the_premise_framing_forbids_citing_the_claim_itself() -> None:
    """An authority attached to the paper's novel claim cannot be grounded: a
    claim the corpus supports would be COMMONPLACE and the paper would have
    nothing to argue (REMEDIATION §14).
    """
    prompt = debate_prompt("Q?", "Weights are expressive.", "premise")
    assert "Weights are expressive." in prompt
    assert "do not attach authority to it" in prompt.lower()
    assert "established" in prompt.lower()


def test_the_two_framings_differ_in_what_they_ask_for() -> None:
    claim = debate_prompt("Q?", "A claim.", "claim")
    premise = debate_prompt("Q?", "A claim.", "premise")
    assert claim != premise
    assert "Is the following claim correct?" in claim
    assert "Is the following claim correct?" not in premise


def test_an_unknown_framing_is_refused_rather_than_defaulted() -> None:
    """Silently falling back to `claim` would make an A/B run measure the same
    arm twice — the shape that nearly wrecked D7.
    """
    with pytest.raises(ValueError):
        debate_prompt("Q?", "A.", "premises")


def test_the_runner_carries_the_framing_into_the_report_mode() -> None:
    """A run that cannot say which arm produced it cannot be compared."""
    from evals.run_loop_eval import build_parser as eval_parser

    assert eval_parser().parse_args([]).cite_for == "claim"
    assert eval_parser().parse_args(["--cite-for", "premise"]).cite_for == "premise"


def test_live_is_off_unless_asked_for() -> None:
    parser = build_parser()
    assert parser.parse_args(["answer", "x"]).live is False
    assert parser.parse_args(["answer", "--live", "x"]).live is True
    assert parser.parse_args(["cycle", "--live"]).live is True


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
