"""The D7 scorer comparison. Offline: no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.compare_support_scorers import (
    EXPECTED,
    FellBackToLexical,
    Pair,
    build,
    build_parser,
    load_pairs,
    main,
    score_all,
)


class _Record:
    def __init__(self, passages: list[str]) -> None:
        self.passages = passages


def _report(tmp_path: Path) -> Path:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "exchanges": [
                            {
                                "authorities": [
                                    {
                                        "citation": "1 U.S. 1",
                                        "claim": "Source code is protected expression.",
                                        "grounded": False,
                                    },
                                    {
                                        "citation": "2 U.S. 2",
                                        "claim": "A wholly unrelated proposition.",
                                        "grounded": True,
                                    },
                                ]
                            },
                            {"authorities": []},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pairs_are_read_from_a_live_report(tmp_path: Path) -> None:
    pairs = load_pairs(_report(tmp_path))
    assert [p.citation for p in pairs] == ["1 U.S. 1", "2 U.S. 2"]
    assert [p.grounded for p in pairs] == [False, True]


def test_a_report_with_no_authorities_is_an_error_not_an_empty_table(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    assert main([str(path)]) == 1
    assert "nothing to compare" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The silent fallback is the hazard this experiment exists to avoid
# --------------------------------------------------------------------------- #
def test_lexical_builds_as_itself() -> None:
    assert type(build("lexical")).__name__ == EXPECTED["lexical"]


def test_a_scorer_that_fell_back_is_a_hard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`build_support_scorer` substitutes lexical when the extra is missing, so
    an experiment could measure it three times and report "no difference".
    """
    from legal_research.citations import support

    monkeypatch.setattr(
        support, "EmbeddingSupportScorer", lambda *a, **k: (_ for _ in ()).throw(ImportError())
    )
    with pytest.raises(FellBackToLexical) as caught:
        build("embedding")
    assert "under another name" in str(caught.value)


def test_every_mode_declares_the_class_it_must_produce() -> None:
    assert set(EXPECTED) == {"lexical", "embedding", "nli"}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_a_citation_with_no_corpus_text_is_unjudgeable_not_unsupported() -> None:
    """The scorer never got the chance to disagree with anything."""
    result = score_all([Pair("9 U.S. 9", "A claim.", False)], "lexical", {})
    assert result["unjudgeable"] == 1
    assert result["judged"] == 0
    assert result["rows"][0]["supports"] is None


def test_a_supporting_passage_scores_above_the_threshold() -> None:
    claim = "Source code is protected expression under the First Amendment."
    index = {"1 U.S. 1": _Record([claim])}
    result = score_all([Pair("1 U.S. 1", claim, False)], "lexical", index)
    assert result["supported"] == 1
    assert result["rows"][0]["best"] is not None


def test_an_unrelated_passage_does_not_support_the_claim() -> None:
    index = {"1 U.S. 1": _Record(["Preemption governs conflicting state statutes."])}
    result = score_all(
        [Pair("1 U.S. 1", "Weights are expressive material.", False)], "lexical", index
    )
    assert result["supported"] == 0


def test_the_best_passage_decides() -> None:
    claim = "Source code is protected expression."
    index = {"1 U.S. 1": _Record(["Something unrelated entirely.", claim])}
    assert score_all([Pair("1 U.S. 1", claim, False)], "lexical", index)["supported"] == 1


def test_the_result_records_which_scorer_actually_ran() -> None:
    """A run that cannot say what produced it cannot be compared to another."""
    result = score_all([Pair("1 U.S. 1", "A claim.", False)], "lexical", {})
    assert result["scorer"] == "LexicalSupportScorer"
    assert result["threshold"] > 0


def test_the_parser_accepts_its_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args([str(tmp_path / "r.json"), "--modes", "lexical"])
    assert args.modes == "lexical"


def test_the_runner_compares_and_prints(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([str(_report(tmp_path)), "--modes", "lexical"]) == 0
    out = capsys.readouterr().out
    assert "authority claim(s)" in out
    assert "LexicalSupportScorer" in out


# --------------------------------------------------------------------------- #
# The length confound in the default scorer
# --------------------------------------------------------------------------- #
def test_lexical_support_cannot_fall_as_a_passage_grows() -> None:
    """`lexical_support` is recall of the CLAIM's tokens in the passage, with no
    penalty for what else the passage says. It is therefore monotonically
    non-decreasing in passage length, which makes scores incomparable across
    corpora whose passages differ in length (REMEDIATION §16).
    """
    from legal_research.citations.support import lexical_support

    claim = "Model weights are a form of expression protected by the First Amendment."
    short = "The court considered whether the statute was severable."
    padded = short + " Model weights and expression and protection and amendments."
    assert lexical_support(claim, padded) >= lexical_support(claim, short)


def test_unrelated_text_can_clear_the_lexical_threshold_on_length_alone() -> None:
    """A passage that says nothing about the claim passes if it happens to
    contain the claim's vocabulary. This is why the corpus rebuild raised the
    lexical column without any authority becoming better supported.
    """
    from legal_research.citations.support import LEXICAL_THRESHOLD, lexical_support

    claim = "Model weights are a form of expression protected by the First Amendment."
    unrelated = (
        "The court considered whether the statute was severable. Model weights and "
        "expression and protection and amendments arise in many unrelated contexts "
        "throughout a long judicial opinion concerned entirely with procedure."
    )
    assert lexical_support(claim, unrelated) > LEXICAL_THRESHOLD
