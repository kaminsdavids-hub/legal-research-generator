"""The retrieval ceiling. Offline: no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

from evals.compare_support_scorers import Pair
from evals.retrieval_ceiling import build_parser, main, measure


class _Record:
    def __init__(self, title: str, passages: list[str]) -> None:
        self.title = title
        self.passages = passages


ON_POINT = "Source code is expression protected by the First Amendment in this circuit."
OFF_POINT = "Maryland recognises civil aiding and abetting as a cause of action."


def _index() -> dict[str, _Record]:
    return {
        "1 U.S. 1": _Record("On Point", [ON_POINT]),
        "2 U.S. 2": _Record("Off Point", [OFF_POINT]),
    }


def test_a_claim_retrieval_got_right_is_not_counted_as_recoverable() -> None:
    report = measure([Pair("1 U.S. 1", ON_POINT, True)], _index(), "lexical")
    assert report.actual == 1
    assert report.ceiling == 1
    assert report.missed == []


def test_a_better_record_retrieval_passed_over_is_recoverable() -> None:
    """The finding that matters: the corpus contained a right answer and
    retrieval chose a different record.
    """
    report = measure([Pair("2 U.S. 2", ON_POINT, False)], _index(), "lexical")
    assert report.actual == 0
    assert report.ceiling == 1
    assert [c.best_cite for c in report.missed] == ["1 U.S. 1"]


def test_a_claim_no_record_supports_is_not_blamed_on_retrieval() -> None:
    """Ceiling equal to actual is the answer 'varying retrieval cannot help'."""
    report = measure(
        [Pair("1 U.S. 1", "An entirely unrelated proposition about tax deadlines.", False)],
        _index(),
        "lexical",
    )
    assert report.ceiling == 0
    assert report.missed == []


def test_the_chosen_record_is_scored_with_the_same_scorer_as_the_ceiling() -> None:
    """Comparing a live run's verdict against a ceiling computed by a different
    scorer would blame retrieval for a scorer disagreement.
    """
    # `chosen_grounded=True` is what the live run said; under this scorer the
    # chosen record does not support the claim, and `actual` must follow the
    # scorer rather than the stale verdict.
    report = measure([Pair("2 U.S. 2", ON_POINT, True)], _index(), "lexical")
    assert report.live_grounded == 1
    assert report.actual == 0
    assert report.claims[0].chosen_score < report.threshold


def test_a_citation_absent_from_the_corpus_scores_zero_not_crashes() -> None:
    report = measure([Pair("9 U.S. 9", ON_POINT, False)], _index(), "lexical")
    assert report.claims[0].chosen_score == 0.0
    assert report.ceiling == 1


def test_the_best_record_is_searched_across_the_whole_corpus() -> None:
    """A ceiling that only looked at the chosen record would measure nothing."""
    report = measure([Pair("2 U.S. 2", ON_POINT, False)], _index(), "lexical")
    assert report.claims[0].best_cite == "1 U.S. 1"
    assert report.claims[0].best_score > report.claims[0].chosen_score


def test_an_empty_report_is_an_error_not_a_ceiling_of_zero(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    assert main([str(path)]) == 1
    assert "nothing to measure" in capsys.readouterr().err


def test_the_parser_defaults_to_the_two_fast_scorers() -> None:
    """NLI over every record for every claim is quadratic and slow; it is
    available but not the default.
    """
    assert build_parser().parse_args(["r.json"]).modes == "lexical,embedding"
