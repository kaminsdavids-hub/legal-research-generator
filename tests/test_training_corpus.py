"""Capturing exchanges for a fine-tune: what is kept, what is not, and consent."""

from __future__ import annotations

import json
from pathlib import Path

from modules.maieutic.cli import build_parser, main
from modules.maieutic.corpus import MIN_SAMPLE_WORDS, TrainingCorpus, Triple, from_env
from modules.maieutic.loop import Gates, Session

ANSWER = (
    "The disclosure exception turns on whether the recipient was under a duty of "
    "confidence at the moment of receipt, not on what they later did with it."
)


def _triple(**overrides) -> Triple:
    base = {
        "question": "What makes the duty attach at receipt rather than at use?",
        "answer": ANSWER,
        "gap_kind": "unsupported_claim",
        "section": "II.A",
        "synthesis": "A synthesis the machine produced and never merged.",
        "merged": True,
        "manuscript": "paper-1",
    }
    return Triple(**{**base, **overrides})


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #
def test_capture_is_off_unless_both_variables_are_set() -> None:
    assert from_env({}) is None
    assert from_env({"LRG_MAIEUTIC_TRAINING_CAPTURE": "1"}) is None
    assert from_env({"LRG_MAIEUTIC_TRAINING_PATH": "/tmp/x.jsonl"}) is None


def test_capture_is_on_only_when_asked_for_by_name(tmp_path: Path) -> None:
    corpus = from_env(
        {
            "LRG_MAIEUTIC_TRAINING_CAPTURE": "1",
            "LRG_MAIEUTIC_TRAINING_PATH": str(tmp_path / "training.jsonl"),
        }
    )
    assert isinstance(corpus, TrainingCorpus)
    assert corpus.path == tmp_path / "training.jsonl"


# --------------------------------------------------------------------------- #
# What is kept
# --------------------------------------------------------------------------- #
def test_a_recorded_exchange_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    corpus = TrainingCorpus.load(path)
    assert corpus.record(_triple()) is True

    reloaded = TrainingCorpus.load(path)
    assert len(reloaded.triples) == 1
    assert reloaded.triples[0].answer == ANSWER
    assert reloaded.triples[0].merged is True


def test_an_exact_duplicate_is_not_counted_twice(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    assert corpus.record(_triple()) is True
    assert corpus.record(_triple()) is False
    assert len(corpus.triples) == 1


def test_a_second_attempt_at_the_same_question_is_kept(tmp_path: Path) -> None:
    """The second answer is evidence about the first; it does not replace it."""

    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    corpus.record(_triple())
    corpus.record(_triple(answer=ANSWER + " The duty is contractual, not equitable."))
    assert len(corpus.triples) == 2


def test_a_keystroke_is_not_a_sample(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    assert corpus.record(_triple(answer="yes")) is False
    assert corpus.record(_triple(answer="")) is False
    assert corpus.triples == []
    # And the threshold is the one the docstring names.
    assert MIN_SAMPLE_WORDS == 8


def test_a_refused_patch_is_still_captured(tmp_path: Path) -> None:
    """Training on merged answers only would narrow the sample to what the
    machinery liked — the collapse learn.py refuses one layer up."""

    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    assert corpus.record(_triple(merged=False)) is True
    assert corpus.triples[0].merged is False


def test_a_corrupt_line_does_not_cost_the_history(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    corpus = TrainingCorpus.load(path)
    corpus.record(_triple())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    corpus.record(_triple(answer=ANSWER + " Second."))

    assert len(TrainingCorpus.load(path).triples) == 2


# --------------------------------------------------------------------------- #
# The loop, which is what has to actually call it
# --------------------------------------------------------------------------- #
def _answer_in_the_loop(training: TrainingCorpus | None) -> Session:
    session = Session()
    session.training = training
    session.manuscript_id = "paper-1"
    session.begin(
        "Open-weight model publication is protected speech under the First Amendment.",
        section="I",
    )
    assert session.ask() is not None
    session.answer(
        "Publication is protected because the weights are expressive: they encode the "
        "authors' choices about architecture and data, and a reader can study them, "
        "which is the reason source code was protected in the first place.",
        Gates.offline(),
    )
    return session


def test_the_loop_captures_an_answered_question(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    session = _answer_in_the_loop(corpus)

    assert len(corpus.triples) == 1
    triple = corpus.triples[0]
    assert triple.manuscript == "paper-1"
    assert triple.gap_kind
    assert triple.question.strip()
    assert "expressive" in triple.answer
    # Written through, not just held in memory.
    assert len(TrainingCorpus.load(corpus.path).triples) == 1
    assert session.journal.episodes  # the journal still counts the question


def test_the_loop_captures_nothing_when_the_author_did_not_opt_in(tmp_path: Path) -> None:
    session = _answer_in_the_loop(None)
    assert session.training is None
    assert not list(tmp_path.iterdir())


def test_a_declined_question_leaves_no_sample(tmp_path: Path) -> None:
    """No answer, no target. The journal still counts it; this must not."""

    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    session = Session()
    session.training = corpus
    session.begin("A thesis worth arguing about, at some length.", section="I")
    session.ask()
    session.decline()

    assert corpus.triples == []
    assert len(session.journal.episodes) == 1


def test_the_manuscript_label_survives_a_reload(tmp_path: Path) -> None:
    session = Session()
    session.manuscript_id = "paper-7"
    session.begin("A thesis worth arguing about, at some length.", section="I")
    state = tmp_path / "session.json"
    session.save(state)

    assert Session.load(state).manuscript_id == "paper-7"


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_targets_the_authors_words(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    corpus.record(_triple())
    out = tmp_path / "sft.jsonl"

    assert corpus.export_sft(out) == 1
    sample = json.loads(out.read_text(encoding="utf-8").strip())
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]
    assert sample["messages"][1]["content"] == ANSWER
    assert sample["metadata"]["gap_kind"] == "unsupported_claim"


def test_synthesis_is_excluded_unless_asked_for(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    corpus.record(_triple())
    plain, with_synthesis = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    corpus.export_sft(plain)
    corpus.export_sft(with_synthesis, include_synthesis=True)

    assert len(json.loads(plain.read_text().strip())["messages"]) == 2
    assert len(json.loads(with_synthesis.read_text().strip())["messages"]) == 3


def test_merged_only_filters_at_export_not_at_capture(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    corpus.record(_triple(merged=True))
    corpus.record(_triple(answer=ANSWER + " Refused.", merged=False))

    assert corpus.export_sft(tmp_path / "all.jsonl") == 2
    assert corpus.export_sft(tmp_path / "merged.jsonl", merged_only=True) == 1


def test_summary_reports_more_than_a_count(tmp_path: Path) -> None:
    corpus = TrainingCorpus.load(tmp_path / "training.jsonl")
    corpus.record(_triple())
    corpus.record(_triple(answer=ANSWER + " Refused.", merged=False, manuscript="paper-2"))

    summary = corpus.summary()
    assert summary["samples"] == 2
    assert summary["merged"] == 1
    assert summary["manuscripts"] == 2
    assert summary["authored_words"] > 40


# --------------------------------------------------------------------------- #
# The CLI, which is how anyone actually reaches this
# --------------------------------------------------------------------------- #
def test_the_training_command_exists_and_parses() -> None:
    args = build_parser().parse_args(["training", "--export", "out.jsonl", "--merged-only"])
    assert args.command == "training"
    assert args.merged_only is True


def test_training_command_names_the_switch_when_capture_is_off(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("LRG_MAIEUTIC_TRAINING_CAPTURE", raising=False)
    code = main(["--state", str(tmp_path / "s.json"), "--journal", str(tmp_path / "j.json"), "training"])
    out = capsys.readouterr().out

    assert code == 1
    assert "LRG_MAIEUTIC_TRAINING_CAPTURE=1" in out
    assert "LRG_MAIEUTIC_TRAINING_PATH" in out


def test_training_command_reports_and_exports(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "training.jsonl"
    TrainingCorpus.load(path).record(_triple())
    monkeypatch.setenv("LRG_MAIEUTIC_TRAINING_CAPTURE", "1")
    monkeypatch.setenv("LRG_MAIEUTIC_TRAINING_PATH", str(path))

    code = main(
        [
            "--state",
            str(tmp_path / "s.json"),
            "--journal",
            str(tmp_path / "j.json"),
            "training",
            "--export",
            str(tmp_path / "sft.jsonl"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "1 exchange(s)" in out
    assert "wrote 1 sample(s)" in out


def test_an_empty_export_is_reported_as_a_problem(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "training.jsonl"
    TrainingCorpus.load(path).record(_triple(merged=False))
    monkeypatch.setenv("LRG_MAIEUTIC_TRAINING_CAPTURE", "1")
    monkeypatch.setenv("LRG_MAIEUTIC_TRAINING_PATH", str(path))

    code = main(
        [
            "--state",
            str(tmp_path / "s.json"),
            "--journal",
            str(tmp_path / "j.json"),
            "training",
            "--export",
            str(tmp_path / "sft.jsonl"),
            "--merged-only",
        ]
    )
    err = capsys.readouterr().err

    assert code == 1
    assert "empty file" in err
