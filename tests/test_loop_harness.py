"""The loop eval harness. Offline: no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

from evals.loop_harness import (
    ExchangeRecord,
    SessionReport,
    SessionScript,
    load_scripts,
    run_all,
    run_script,
)
from evals.run_loop_eval import DEFAULT_SESSIONS, build_parser, main, report_lines
from modules.dialectic.models import CitationSlot, DialecticTurn, Position, SlotStatus
from modules.maieutic.loop import Gates

SCRIPT = SessionScript(
    id="T1",
    thesis="Publishing open model weights is not a deemed export.",
    answers=[
        "Someone would say weights are functional artifacts, not published information.",
        "Functionality has never defeated the exclusion for other open technical material.",
    ],
)


def _turn(objection: str, cite: str = "") -> DialecticTurn:
    slot = CitationSlot(proposition=objection)
    if cite:
        slot = CitationSlot(
            proposition=objection, status=SlotStatus.VERIFIED, normalized_cite=cite
        )
    return DialecticTurn(
        question="q",
        thesis=Position(side="thesis", model="a", family="fa", propositions=[slot]),
        antithesis=Position(
            side="antithesis",
            model="b",
            family="fb",
            propositions=[CitationSlot(proposition=f"Counter to: {objection}")],
        ),
    )


# --------------------------------------------------------------------------- #
# Missing data is not zero
# --------------------------------------------------------------------------- #
def test_authority_survival_is_unmeasured_when_none_was_proposed() -> None:
    """The wall was never approached, which is not the same as it holding.

    Averaging this in as 0.0 would report every offline run as total failure.
    """
    assert ExchangeRecord(index=0, authorities_proposed=0).authority_survival is None


def test_authority_survival_is_a_real_zero_when_all_were_refused() -> None:
    record = ExchangeRecord(index=0, authorities_proposed=3, authorities_grounded=0)
    assert record.authority_survival == 0.0


def test_a_session_that_measured_nothing_reports_no_survival_rate() -> None:
    assert SessionReport("x", "fixture", "offline").answer_survival is None


def test_an_errored_exchange_is_excluded_rather_than_counted_as_a_refusal() -> None:
    report = SessionReport("x", "fixture", "offline")
    report.exchanges = [
        ExchangeRecord(index=0, merged=True),
        ExchangeRecord(index=1, error="RuntimeError: boom"),
    ]
    assert report.answer_survival == 1.0, "a crash is missing data, not a refusal"


def test_the_report_prints_missing_data_as_missing() -> None:
    report = run_all([SCRIPT], Gates.offline())
    lines = "\n".join(report_lines(report))
    assert "authority survival: n/a (nothing measured)" in lines


# --------------------------------------------------------------------------- #
# What it measures
# --------------------------------------------------------------------------- #
def test_a_scripted_session_replays_through_the_loop() -> None:
    report = run_script(SCRIPT, Gates.offline())
    assert len(report.exchanges) == 2
    assert all(e.merged for e in report.exchanges)
    assert report.answer_survival == 1.0
    assert report.nodes == 3


def test_the_machines_objections_are_counted_separately_from_the_authors() -> None:
    report = run_script(SCRIPT, Gates.offline(), turns=lambda q, a: _turn("An objection."))
    first = report.exchanges[0]
    assert first.author_nodes == 1
    assert first.machine_nodes == 2
    assert first.objections == 1, "only the machine's objections count as pressure"


def test_a_dropped_authority_is_counted_as_proposed_and_not_grounded() -> None:
    report = run_script(
        SCRIPT, Gates.offline(), turns=lambda q, a: _turn("A cited point.", "1 U.S. 1")
    )
    first = report.exchanges[0]
    assert first.authorities_proposed == 1
    assert first.authorities_grounded == 0
    assert first.authority_survival == 0.0
    assert first.dropped == 1
    assert first.merged, "the author's answer still merged"


def test_a_thin_session_is_refused_and_the_refusals_are_recorded() -> None:
    thin = SessionScript(
        id="thin",
        thesis="The treatment of open weights is unsettled.",
        answers=["It may perhaps arguably possibly seem so."],
    )
    report = run_script(thin, Gates.offline())
    assert report.answer_survival == 0.0
    assert any("banality" in r for r in report.exchanges[0].refusals)


def test_a_session_that_runs_dry_says_how_many_answers_went_unused() -> None:
    script = SessionScript(
        id="dry",
        thesis="A thesis.",
        answers=[
            "The strongest objection is that the premise is unsupported.",
            "Supporting material addressing that objection directly.",
            "A third answer with nowhere left to attach.",
            "A fourth answer likewise.",
        ],
    )
    report = run_script(script, Gates.offline())
    if report.ran_dry:
        assert report.answers_unused > 0
        assert len(report.exchanges) + report.answers_unused == len(script.answers)


def test_an_exchange_that_raises_does_not_discard_the_ones_before_it() -> None:
    calls = {"n": 0}

    def _explode(question: str, answer: str) -> DialecticTurn:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return _turn("An objection.")

    report = run_script(SCRIPT, Gates.offline(), turns=_explode)
    assert len(report.exchanges) == 2
    assert report.exchanges[0].merged
    # The loop catches provider failures itself, so this surfaces as turn_error
    # rather than as a harness error -- the author's answer still merged.
    assert report.exchanges[1].turn_error or report.exchanges[1].error


def test_a_broken_script_is_data_not_a_crash() -> None:
    report = run_script(SessionScript(id="bad", thesis="   ", answers=["x"]))
    assert report.error
    assert report.exchanges == []


# --------------------------------------------------------------------------- #
# The fixture, and the honesty it carries
# --------------------------------------------------------------------------- #
def test_the_shipped_fixture_loads_and_runs() -> None:
    scripts = load_scripts(DEFAULT_SESSIONS)
    assert scripts
    report = run_all(scripts, Gates.offline())
    assert len(report.sessions) == len(scripts)
    assert report.summary()["exchanges"] > 0


def test_every_script_records_where_its_answers_came_from() -> None:
    """A run over stand-ins and a run over a real transcript are not comparable,
    so the provenance travels with every report.
    """
    for script in load_scripts(DEFAULT_SESSIONS):
        assert script.source
    report = run_all(load_scripts(DEFAULT_SESSIONS), Gates.offline())
    assert all(s.source for s in report.sessions)


def test_the_fixture_says_its_answers_are_stand_ins() -> None:
    """If that note ever goes missing, the numbers start looking like evidence
    about real authors, which they are not.
    """
    raw = json.loads(DEFAULT_SESSIONS.read_text(encoding="utf-8"))
    note = " ".join(raw["note"]).lower()
    assert "stand-in" in note
    assert "not a real author" in note or "not an author" in note


# --------------------------------------------------------------------------- #
# The runner, invoked as a runner
# --------------------------------------------------------------------------- #
def test_the_parser_accepts_its_advertised_flags() -> None:
    parser = build_parser()
    assert parser.parse_args([]).live is False
    assert parser.parse_args(["--live"]).live is True
    assert parser.parse_args(["--only", "S1-export-control"]).only == "S1-export-control"


def test_the_runner_runs_offline_and_prints_a_summary(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--only", "S1-export-control"]) == 0
    out = capsys.readouterr().out
    assert "mode: offline" in out
    assert "answer survival" in out


def test_an_unknown_session_id_is_an_error_not_an_empty_run(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--only", "does-not-exist"]) == 1
    assert "no session with id" in capsys.readouterr().err


def test_the_runner_writes_a_full_report(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "report.json"
    main(["--only", "S3-thin", "--out", str(out)])
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["mode"] == "offline"
    assert payload["sessions"][0]["script_id"] == "S3-thin"
    assert payload["sessions"][0]["exchanges"], "per-exchange detail must survive"
