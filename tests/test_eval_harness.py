"""Eval harness tests. Offline: no network, no GPU, no models."""

from __future__ import annotations

import json
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
        name = "saul"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            seen.append(messages[1]["content"])
            return _judge_json()

    es = EvalSet.load(EVAL_SET_PATH)
    a1 = next(q for q in es.questions if q.id == "A1")
    judge_response(_Recording(), a1, _turn(CitationSlot(proposition="Weights are speech.")))
    assert "Bernstein v. DOJ" in seen[0]
    assert "Weights are speech." in seen[0]
    assert "CRUX TABLE" in seen[0]


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
