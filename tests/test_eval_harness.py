"""Eval harness tests. Offline: no network, no GPU, no models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evals.harness import (
    CRITERIA,
    EvalSet,
    LoggingRetriever,
    QuestionResult,
    RunReport,
    gate_citation_integrity,
    gate_temporal_validity,
    has_plateaued,
    judge_response,
    parse_verdict,
)
from modules.dialectic.models import (
    CitationSlot,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)

EVAL_SET_PATH = Path(__file__).resolve().parents[1] / "evals/openweights_first_amendment.json"


def _turn(*slots: CitationSlot) -> DialecticTurn:
    return DialecticTurn(
        question="Q?",
        thesis=Position(
            side="thesis", model="hermes3", family="hermes", propositions=list(slots)
        ),
        antithesis=Position(side="antithesis", model="llama3.1", family="llama"),
    )


# --------------------------------------------------------------------------- #
# Eval set
# --------------------------------------------------------------------------- #
def test_eval_set_loads_with_expected_shape() -> None:
    es = EvalSet.load(EVAL_SET_PATH)
    assert len(es.questions) == 40
    assert len(es.clusters) == 6
    counts: dict[str, int] = {}
    for q in es.questions:
        counts[q.cluster] = counts.get(q.cluster, 0) + 1
    assert counts == {"A": 8, "B": 7, "C": 7, "D": 6, "E": 6, "F": 6}
    assert sum(q.holdout for q in es.questions) == 8


def test_holdouts_are_excluded_by_default() -> None:
    es = EvalSet.load(EVAL_SET_PATH)
    default = es.select()
    assert len(default) == 32
    assert not any(q.holdout for q in default)
    assert {q.id for q in es.select(include_holdout=True)} - {q.id for q in default} == {
        "A4", "A8", "B4", "C3", "D3", "E3", "F3", "F6"
    }


def test_holdout_flag_list_mismatch_is_an_error(tmp_path: Path) -> None:
    """A silent mismatch would leak a holdout into an optimization loop."""
    data = json.loads(EVAL_SET_PATH.read_text())
    data["holdout_ids"] = ["A4"]  # disagrees with the per-question flags
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="holdout mismatch"):
        EvalSet.load(bad)


def test_question_prompt_includes_the_forum_frame() -> None:
    es = EvalSet.load(EVAL_SET_PATH)
    a1 = next(q for q in es.questions if q.id == "A1")
    assert "Frame: declaratory judgment action, N.D. Cal. / Ninth Circuit" in a1.prompt()
    a4 = next(q for q in es.questions if q.id == "A4")
    assert "Frame:" not in a4.prompt(), "no frame means no dangling label"


# --------------------------------------------------------------------------- #
# Retrieval log
# --------------------------------------------------------------------------- #
def test_logging_retriever_records_every_proposal() -> None:
    class _Inner:
        def propose(self, court_hint: str, proposition: str) -> list[str]:
            return ["392 U.S. 1"] if "frisk" in court_hint else []

    r = LoggingRetriever(_Inner())
    assert r.propose("stop and frisk", "P.") == ["392 U.S. 1"]
    assert r.propose("unrelated", "P.") == []
    assert len(r.log) == 2
    assert r.retrieved_cites == {"392 U.S. 1"}
    r.reset()
    assert not r.log and not r.retrieved_cites


# --------------------------------------------------------------------------- #
# Gate: citation integrity
# --------------------------------------------------------------------------- #
def test_citation_gate_passes_when_every_cite_was_retrieved_and_verified() -> None:
    turn = _turn(
        CitationSlot(
            proposition="P.",
            normalized_cite="392 U.S. 1",
            status=SlotStatus.VERIFIED,
            weight=Weight.CONTROLLING,
        )
    )
    assert gate_citation_integrity(turn, {"392 U.S. 1"}).passed


def test_citation_gate_fails_a_cite_absent_from_the_retrieval_log() -> None:
    """The fabrication case the gate exists to catch."""
    turn = _turn(
        CitationSlot(
            proposition="P.", normalized_cite="999 U.S. 999", status=SlotStatus.VERIFIED
        )
    )
    result = gate_citation_integrity(turn, {"392 U.S. 1"})
    assert not result.passed
    assert "not in the retrieval log" in result.detail
    assert "999 U.S. 999" in result.detail


def test_citation_gate_distinguishes_unverified_from_fabricated() -> None:
    turn = _turn(
        CitationSlot(
            proposition="P.", normalized_cite="392 U.S. 1", status=SlotStatus.PROPOSED
        )
    )
    result = gate_citation_integrity(turn, {"392 U.S. 1"})
    assert not result.passed
    assert "retrieved but not verified" in result.detail


def test_citation_gate_passes_a_response_that_cites_nothing() -> None:
    """Citing nothing is not a citation-integrity failure; the judge handles it."""
    turn = _turn(CitationSlot(proposition="P.", weight=Weight.SUPPORTING))
    assert gate_citation_integrity(turn, set()).passed


# --------------------------------------------------------------------------- #
# Gate: temporal validity
# --------------------------------------------------------------------------- #
_STATUS = {
    "90 Fed. Reg. 4544": ("rescinded", "rescinded May 2025"),
    "15 C.F.R. 734.7": ("in_force", ""),
}


def test_temporal_gate_fails_a_rescinded_authority_presented_as_operative() -> None:
    turn = _turn(
        CitationSlot(
            proposition="The rule controls these weights.",
            normalized_cite="90 Fed. Reg. 4544",
            status=SlotStatus.VERIFIED,
            note="resolved by CourtListener v4 citation-lookup",
        )
    )
    result = gate_temporal_validity(turn, _STATUS)
    assert not result.passed
    assert "rescinded" in result.detail


def test_temporal_gate_ignores_the_machinery_authored_note() -> None:
    """The retrieval stage writes the status into `note`.

    A gate that read it would be checking that our own code wrote a string, not
    that the response understood the rule was repealed -- it would pass every
    time and measure nothing.
    """
    turn = _turn(
        CitationSlot(
            proposition="The rule controls these weights.",
            normalized_cite="90 Fed. Reg. 4544",
            status=SlotStatus.VERIFIED,
            note="candidate proposed by retrieval; NOT CURRENTLY OPERATIVE (rescinded)",
        )
    )
    assert not gate_temporal_validity(turn, _STATUS).passed, (
        "the auto-written note must not satisfy the gate"
    )


def test_temporal_gate_passes_when_the_synthesis_acknowledges_the_rescission() -> None:
    """The synthesis is the only role that runs after retrieval."""
    turn = _turn(
        CitationSlot(
            proposition="The framework shows the control design.",
            normalized_cite="90 Fed. Reg. 4544",
            status=SlotStatus.VERIFIED,
            note="candidate proposed by retrieval",
        )
    )
    turn.synthesis = (
        "That framework was rescinded in 2025 and is no longer operative law; it "
        "carries only precedential weight on what the agency believed it could do."
    )
    assert gate_temporal_validity(turn, _STATUS).passed


def test_temporal_gate_accepts_acknowledgement_in_a_proposition() -> None:
    turn = _turn(
        CitationSlot(
            proposition="The now-repealed framework shows the intended design.",
            normalized_cite="90 Fed. Reg. 4544",
            status=SlotStatus.VERIFIED,
        )
    )
    assert gate_temporal_validity(turn, _STATUS).passed


def test_temporal_gate_ignores_in_force_authorities() -> None:
    turn = _turn(
        CitationSlot(
            proposition="Published information is not subject to the EAR.",
            normalized_cite="15 C.F.R. 734.7",
            status=SlotStatus.VERIFIED,
        )
    )
    assert gate_temporal_validity(turn, _STATUS).passed


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
def _judge_json(**scores: int) -> str:
    base = dict.fromkeys(CRITERIA, 5)
    base.update(scores)
    return json.dumps({"scores": base, "rationale": "because"})


def test_parse_verdict_accepts_well_formed_output() -> None:
    v = parse_verdict(_judge_json(steelmanning=9))
    assert v.parsed
    assert v.scores["steelmanning"] == 9
    assert v.rationale == "because"


def test_parse_verdict_strips_markdown_fences() -> None:
    assert parse_verdict(f"```json\n{_judge_json()}\n```").parsed


@pytest.mark.parametrize(
    "bad",
    [
        "8/10, pretty good",
        '{"rationale": "no scores"}',
        '{"scores": {"steelmanning": 5}}',
        '{"scores": {"steelmanning": "high"}}',
        json.dumps({"scores": dict.fromkeys(CRITERIA, 99), "rationale": "x"}),
        json.dumps({"scores": dict.fromkeys(CRITERIA, 0), "rationale": "x"}),
        "not json",
    ],
)
def test_parse_verdict_rejects_malformed_or_out_of_range(bad: str) -> None:
    v = parse_verdict(bad)
    assert not v.parsed
    assert v.mean == 0.0


def test_judge_failure_is_a_result_not_a_crash() -> None:
    class _Exploding:
        name = "saul"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            raise RuntimeError("judge offline")

    es = EvalSet.load(EVAL_SET_PATH)
    v = judge_response(_Exploding(), es.questions[0], _turn(CitationSlot(proposition="P.")))
    assert not v.parsed
    assert "judge offline" in v.rationale


def test_judge_sees_the_anchors_and_the_rendered_exchange() -> None:
    seen: list[str] = []

    class _Recording:
        name = "hermes3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            seen.append(messages[1]["content"])
            return _judge_json()

    es = EvalSet.load(EVAL_SET_PATH)
    a1 = next(q for q in es.questions if q.id == "A1")
    judge_response(_Recording(), a1, _turn(CitationSlot(proposition="Weights are speech.")))
    assert "Bernstein v. DOJ" in seen[0]
    assert "Weights are speech." in seen[0]
    assert "CRUXES:" in seen[0]
    # [UNSUPPORTED] markers must survive into the judge's view: an unverified
    # authority is exactly what authority_hierarchy is meant to penalise.
    assert "[UNSUPPORTED" in seen[0]


# --------------------------------------------------------------------------- #
# Scoring and aggregation
# --------------------------------------------------------------------------- #
def test_a_failed_hard_gate_scores_zero_however_good_the_judge_thought_it_was() -> None:
    from evals.harness import GateResult

    r = QuestionResult(
        id="A1",
        cluster="A",
        holdout=False,
        gates=[GateResult("citation_integrity", False, "fabricated")],
        verdict=parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 10))),
    )
    assert r.verdict.mean == 10.0
    assert r.score == 0.0
    assert r.to_dict()["gates"]["pass"] is False


def test_report_aggregates_per_cluster_and_lists_gate_failures() -> None:
    from evals.harness import GateResult

    ok = GateResult("citation_integrity", True, "")
    bad = GateResult("citation_integrity", False, "fabricated")
    report = RunReport(eval_set="x")
    report.results = [
        QuestionResult("A1", "A", False, [ok], parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 8)))),
        QuestionResult("A2", "A", False, [ok], parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 6)))),
        QuestionResult("B1", "B", False, [bad], parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 9)))),
    ]
    assert report.cluster_means() == {"A": 7.0, "B": 0.0}
    assert report.gate_failures() == ["B1"]
    payload = report.to_dict()
    assert payload["overall_mean"] == pytest.approx(14 / 3, abs=1e-3)
    assert len(payload["questions"][0]["scores"]) == 5


# --------------------------------------------------------------------------- #
# Plateau rule
# --------------------------------------------------------------------------- #
def test_plateau_needs_three_consecutive_small_changes() -> None:
    assert has_plateaued([5.0, 5.05, 5.1, 5.12])
    assert not has_plateaued([5.0, 5.05, 5.1])  # only 2 changes observed
    assert not has_plateaued([5.0, 5.05, 5.9, 5.95])  # one big jump
    assert not has_plateaued([])


def test_the_threshold_sits_above_the_measured_noise_floor() -> None:
    """0.2 was the specified threshold; measurement showed it is not usable.

    Three runs of unchanged code moved by 0.087 in one set and 0.219 in another,
    the difference being whether a citation gate happened to flip. A flip costs
    about 0.24 on a 32-question mean, so a threshold below that reports
    convergence on a coin toss.
    """
    from evals.harness import PLATEAU_DELTA

    assert PLATEAU_DELTA >= 0.5
    # The worst observed floor must not register as a plateau.
    assert not has_plateaued([6.769, 6.550, 6.700, 6.950], delta=0.2)
    # A genuinely flat series still does, at the new threshold.
    assert has_plateaued([6.70, 6.72, 6.69, 6.71])
    # And a single gate flip, worth ~0.24, does not break a plateau at 0.5.
    assert has_plateaued([6.70, 6.94, 6.70, 6.94])


def test_plateau_is_insensitive_to_direction() -> None:
    """A steady decline is a plateau too; the rule is about magnitude."""
    assert has_plateaued([6.0, 5.9, 5.8, 5.7])


def test_citation_gate_accepts_corpus_verified_non_case_authority() -> None:
    """CourtListener cannot adjudicate a C.F.R. section; the corpus verifies it.

    Requiring a cluster for `15 C.F.R. 734.7` would fail every response that
    correctly relies on the EAR published exclusion.
    """
    turn = _turn(
        CitationSlot(
            proposition="Published information is outside the EAR.",
            normalized_cite="15 C.F.R. 734.7",
            status=SlotStatus.NOT_FOUND,
            note="CourtListener returned no result",
        )
    )
    assert not gate_citation_integrity(turn, {"15 C.F.R. 734.7"}).passed
    assert gate_citation_integrity(
        turn, {"15 C.F.R. 734.7"}, corpus_verified={"15 C.F.R. 734.7"}
    ).passed


def test_corpus_verification_does_not_excuse_a_fabricated_cite() -> None:
    """Corpus standing is not a licence to invent; it must still be retrieved."""
    turn = _turn(
        CitationSlot(
            proposition="P.", normalized_cite="15 C.F.R. 999.9", status=SlotStatus.NOT_FOUND
        )
    )
    result = gate_citation_integrity(turn, set(), corpus_verified={"15 C.F.R. 734.7"})
    assert not result.passed
    assert "not in the retrieval log" in result.detail


def test_logging_retriever_forwards_the_annotator_capability() -> None:
    """A wrapper that drops an optional capability silently turns it off.

    The engine probes with getattr(retriever, "annotate", None). When the log
    wrapper did not forward it, every non-operative authority reached the
    synthesis unflagged and the temporal gate failed runs the module was
    getting right.
    """
    class _Annotating:
        def propose(self, court_hint: str, proposition: str) -> list[str]:
            return ["90 Fed. Reg. 4544"]

        def annotate(self, cite: str) -> str:
            return "NOT CURRENTLY OPERATIVE (rescinded)" if "Fed. Reg." in cite else ""

    r = LoggingRetriever(_Annotating())
    assert r.annotate("90 Fed. Reg. 4544") == "NOT CURRENTLY OPERATIVE (rescinded)"
    assert r.annotate("392 U.S. 1") == ""
    # And the engine's own probe must find it through the wrapper.
    assert callable(getattr(r, "annotate", None))


def test_logging_retriever_tolerates_a_retriever_without_annotate() -> None:
    class _Plain:
        def propose(self, court_hint: str, proposition: str) -> list[str]:
            return []

    assert LoggingRetriever(_Plain()).annotate("392 U.S. 1") == ""


def _result(gid_pass: bool, judge_raw: str, cluster: str = "A", qid: str = "Q1"):
    from evals.harness import GateResult
    return QuestionResult(
        id=qid, cluster=cluster, holdout=False,
        gates=[GateResult("citation_integrity", gid_pass, "")],
        verdict=parse_verdict(judge_raw),
    )


def test_a_broken_judge_is_missing_data_not_a_zero() -> None:
    """Collapsing an unparseable judge to 0.0 reports it as a terrible response.

    On the first full run that single conflation moved cluster D from 7.30 to
    5.84 and turned the best-performing cluster into the worst.
    """
    broken = _result(True, "the judge rambled instead of scoring")
    assert broken.score is None
    assert broken.scored is False
    assert broken.to_dict()["mean"] is None

    # A failed gate is a real zero and must stay one.
    gate_failed = _result(False, _judge_json(**dict.fromkeys(CRITERIA, 9)))
    assert gate_failed.score == 0.0
    assert gate_failed.scored is True


def test_unscored_questions_are_excluded_from_the_means() -> None:
    from evals.harness import GateResult

    ok = GateResult("citation_integrity", True, "")
    report = RunReport(eval_set="x")
    report.results = [
        QuestionResult("D1", "D", False, [ok], parse_verdict("garbage")),
        QuestionResult("D2", "D", False, [ok], parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 7)))),
        QuestionResult("D3", "D", False, [ok], parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 8)))),
    ]
    # 7 and 8 average to 7.5; a 0 for D1 would drag it to 5.0.
    assert report.overall_mean == pytest.approx(7.5)
    assert report.cluster_means() == {"D": 7.5}
    assert report.unscored() == ["D1"]
    payload = report.to_dict()
    assert payload["scored_count"] == 2
    assert payload["total_count"] == 3
    assert payload["unscored_judge_failed"] == ["D1"]


def test_gate_failure_still_counts_as_zero_in_the_mean() -> None:
    from evals.harness import GateResult

    report = RunReport(eval_set="x")
    report.results = [
        QuestionResult("A1", "A", False, [GateResult("g", False, "")],
                       parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 10)))),
        QuestionResult("A2", "A", False, [GateResult("g", True, "")],
                       parse_verdict(_judge_json(**dict.fromkeys(CRITERIA, 8)))),
    ]
    assert report.overall_mean == pytest.approx(4.0)
    assert report.unscored() == []


def test_a_crashed_question_is_unmeasured_not_a_zero() -> None:
    """One slow call must not be scored as a worthless response.

    A timeout previously killed the whole run and discarded 4.4 hours of
    completed work; questions are now isolated, and a dead one is missing data.
    """
    from evals.harness import JudgeVerdict

    crashed = QuestionResult(
        id="D1", cluster="D", holdout=False, gates=[],
        verdict=JudgeVerdict({}, "", parsed=False),
        error="ReadTimeout: timed out",
    )
    assert crashed.score is None
    assert crashed.scored is False
    assert crashed.to_dict()["error"] == "ReadTimeout: timed out"

    report = RunReport(eval_set="x")
    report.results = [crashed, _result(True, _judge_json(**dict.fromkeys(CRITERIA, 8)), "D", "D2")]
    # The crashed question must not drag 8.0 down to 4.0.
    assert report.overall_mean == pytest.approx(8.0)
    assert report.errors() == {"D1": "ReadTimeout: timed out"}
    assert report.to_dict()["run_errors"] == {"D1": "ReadTimeout: timed out"}


def test_empty_gate_list_is_not_treated_as_all_gates_passing() -> None:
    """all([]) is True, which would have made a crashed question 'pass'."""
    from evals.harness import JudgeVerdict

    crashed = QuestionResult(
        id="X", cluster="A", holdout=False, gates=[],
        verdict=JudgeVerdict({}, "", parsed=False), error="boom",
    )
    assert crashed.gates_passed is False


def _crux_turn(n: int) -> DialecticTurn:
    from modules.dialectic.models import Crux
    turn = _turn(CitationSlot(proposition="Thesis point.", weight=Weight.CONTROLLING))
    turn.cruxes = [
        Crux(
            thesis_prop=CitationSlot(proposition=f"T{i}", weight=Weight.CONTROLLING),
            antithesis_prop=CitationSlot(proposition=f"A{i}", weight=Weight.CONTROLLING),
            negates=True, partition="open", winner="none",
            outcome_bearing=False, nli_source="model",
        )
        for i in range(n)
    ]
    return turn


def test_judge_input_does_not_hand_the_judge_a_json_shaped_crux_table() -> None:
    """D1's seven cruxes made the judge transcribe the table instead of scoring.

    Its reply echoed exactly the field names the human-facing renderer emits.
    """
    es = EvalSet.load(EVAL_SET_PATH)
    from evals.harness import render_for_judge

    text = render_for_judge(es.questions[0], _crux_turn(7))
    assert "7 contradiction(s) were identified" in text
    # None of the key-value shapes the judge previously copied back.
    assert "nli:" not in text
    assert "(winner:" not in text
    assert "outcome-bearing;" not in text
    assert "Verification calls spent" not in text
    # The crux content is present exactly once, not twice.
    assert text.count("T0") == 1


def test_judge_retries_with_a_correction_when_the_reply_will_not_parse() -> None:
    seen: list[str] = []

    class _BadThenGood:
        name = "hermes3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            seen.append(messages[1]["content"])
            if len(seen) == 1:
                return '{"7":{"outcome-bearing":"no","winner":"antithesis"}}'
            return _judge_json(**dict.fromkeys(CRITERIA, 7))

    es = EvalSet.load(EVAL_SET_PATH)
    v = judge_response(_BadThenGood(), es.questions[0], _crux_turn(7))
    assert v.parsed
    assert v.mean == 7.0
    assert len(seen) == 2
    assert "REJECTED" in seen[1], "the retry must tell the judge what went wrong"


def test_judge_gives_up_cleanly_after_its_retries() -> None:
    class _AlwaysBad:
        name = "hermes3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            return "not json"

    es = EvalSet.load(EVAL_SET_PATH)
    v = judge_response(_AlwaysBad(), es.questions[0], _crux_turn(2))
    assert not v.parsed
    assert v.mean == 0.0


def test_round_means_group_by_round_id() -> None:
    from evals.harness import round_means

    runs = [("r1", 6.9), ("r1", 6.8), ("r1", 6.7), ("r2", 6.5), ("r2", 6.6), ("r2", 6.4)]
    assert round_means(runs) == pytest.approx([6.8, 6.5])


def test_untagged_runs_are_chunked_in_order() -> None:
    """Sets produced before round ids existed still group sensibly."""
    from evals.harness import round_means

    runs = [("", 6.9), ("", 6.8), ("", 6.7), ("", 6.5), ("", 6.6), ("", 6.4)]
    assert round_means(runs, per_round=3) == pytest.approx([6.8, 6.5])


def test_averaging_shrinks_a_gate_flip_below_the_threshold() -> None:
    """The reason for rounds at all.

    A flip moves a single run's mean by ~0.24, over the 0.2 threshold. Averaged
    across three runs it moves the round by ~0.08, inside it.
    """
    from evals.harness import ROUND_PLATEAU_DELTA, round_means

    clean = [("r1", 6.70), ("r1", 6.70), ("r1", 6.70)]
    # Same round, but one run lost a question to a gate flip.
    flipped = [("r2", 6.70), ("r2", 6.46), ("r2", 6.70)]
    a, b = round_means(clean + flipped)
    assert abs(b - a) == pytest.approx(0.08, abs=0.005)
    assert abs(b - a) < ROUND_PLATEAU_DELTA

    # The same flip judged run-by-run would breach it.
    assert abs(6.70 - 6.46) > ROUND_PLATEAU_DELTA


def test_a_real_regression_still_registers_across_rounds() -> None:
    """Averaging must not blunt the signal the rule exists to catch."""
    from evals.harness import ROUND_PLATEAU_DELTA, has_plateaued, round_means

    runs = [("r1", 6.9), ("r1", 6.9), ("r1", 6.9), ("r2", 6.2), ("r2", 6.2), ("r2", 6.2)]
    a, b = round_means(runs)
    assert abs(b - a) > ROUND_PLATEAU_DELTA
    assert not has_plateaued([a, b, 6.2, 6.2], delta=ROUND_PLATEAU_DELTA)


def test_the_runner_cli_accepts_every_option_main_reads() -> None:
    """The suite never invoked the CLI, so a broken runner passed all 98 tests.

    A --round-id flag was added by a string replacement that silently matched
    nothing, while main() went on reading args.round_id. Every invocation would
    have raised AttributeError, and it was pushed.
    """
    import argparse
    import inspect

    from evals import run_eval

    parser = run_eval.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Defaults alone must produce a usable namespace.
    args = parser.parse_args([])
    source = inspect.getsource(run_eval.main)
    for attr in sorted(set(re.findall(r"args\.([a-z_]+)", source))):
        assert hasattr(args, attr), f"main() reads args.{attr}, which the parser never defines"


def test_round_id_reaches_the_report() -> None:
    from evals.run_eval import build_parser

    assert build_parser().parse_args(["--round-id", "floor-A"]).round_id == "floor-A"
    assert build_parser().parse_args([]).round_id == ""
    report = RunReport(eval_set="x", round_id="floor-A")
    assert report.to_dict()["round_id"] == "floor-A"
