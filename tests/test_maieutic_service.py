"""The corpus-backed citation verifier. Offline: no models, no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.maieutic.service import CorpusCitationVerifier


@dataclass
class _Status:
    value: str


@dataclass
class _Record:
    title: str = "Bernstein v. DOJ"
    volume: int | None = 176
    reporter: str = "F.3d"
    page: int | None = 1132
    code: str = ""
    section: str = ""
    passages: list[str] = field(default_factory=list)
    status: Any = field(default_factory=lambda: _Status("in_force"))
    status_note: str = ""


@dataclass
class _Corpus:
    records: list[_Record]


class _Scorer:
    """Mirrors the real ``SupportScorer``: one passage at a time, plus a threshold."""

    def __init__(self, scores: dict[str, float], threshold: float = 0.5) -> None:
        self._scores = scores
        self.threshold = threshold
        self.seen: list[Any] = []

    def score(self, proposition: str, passage: str) -> float:
        self.seen.append(passage)
        return self._scores.get(passage, 0.0)


def _verifier(scorer: Any = None, **kwargs: Any) -> CorpusCitationVerifier:
    return CorpusCitationVerifier(_Corpus([_Record(**kwargs)]), scorer)


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def test_a_case_in_the_corpus_resolves() -> None:
    ok, detail = _verifier().resolve("176 F.3d 1132")
    assert ok
    assert "Bernstein" in detail


def test_a_citation_absent_from_the_corpus_does_not_resolve() -> None:
    ok, _ = _verifier().resolve("999 U.S. 999")
    assert not ok


def test_non_case_authority_resolves_without_courtlistener() -> None:
    """CourtListener indexes case law and cannot adjudicate a C.F.R. section.

    Sending one there poisoned the cite cache and cost three eval runs
    (REMEDIATION §11.9a); the corpus alone must be able to resolve it.
    """
    v = _verifier(volume=None, page=None, reporter="", code="15 C.F.R.", section="734.7(c)")
    ok, _ = v.resolve("15 C.F.R. 734.7(c)")
    assert ok


def test_resolving_is_not_the_same_as_being_good_law() -> None:
    v = _verifier(status=_Status("superseded"))
    ok, detail = v.resolve("176 F.3d 1132")
    assert ok, "a superseded rule still exists; its status is a separate question"
    assert "NOT CURRENTLY OPERATIVE" in detail


# --------------------------------------------------------------------------- #
# supports
# --------------------------------------------------------------------------- #
def test_the_scorer_is_called_with_one_passage_at_a_time() -> None:
    """Regression: ``score`` takes ``(str, str)``. Handing it the whole list
    silently scored a stringified list and would have passed or crashed for
    reasons unrelated to support.
    """
    scorer = _Scorer({"a": 0.9, "b": 0.1})
    v = _verifier(scorer, passages=["a", "b"])
    v.supports("176 F.3d 1132", "the claim")
    assert scorer.seen == ["a", "b"]
    assert all(isinstance(p, str) for p in scorer.seen)


def test_the_best_passage_decides() -> None:
    """A record supports a claim if any of its text does."""
    scorer = _Scorer({"off point": 0.0, "on point": 0.8})
    ok, detail = _verifier(scorer, passages=["off point", "on point"]).supports(
        "176 F.3d 1132", "the claim"
    )
    assert ok
    assert "0.80" in detail


def test_a_score_below_the_scorers_own_threshold_fails() -> None:
    scorer = _Scorer({"weak": 0.4}, threshold=0.5)
    ok, _ = _verifier(scorer, passages=["weak"]).supports("176 F.3d 1132", "the claim")
    assert not ok


def test_the_threshold_comes_from_the_scorer_not_from_us() -> None:
    """Lexical overlap and NLI entailment are not on comparable scales, so a
    cut chosen here would be wrong for at least one of them.
    """
    passages = ["some text"]
    lenient = _Scorer({"some text": 0.3}, threshold=0.2)
    strict = _Scorer({"some text": 0.3}, threshold=0.9)
    assert _verifier(lenient, passages=passages).supports("176 F.3d 1132", "c")[0]
    assert not _verifier(strict, passages=passages).supports("176 F.3d 1132", "c")[0]


def test_without_a_scorer_support_fails_closed() -> None:
    ok, detail = _verifier(None, passages=["text"]).supports("176 F.3d 1132", "the claim")
    assert not ok
    assert "on trust" in detail


def test_a_record_with_no_text_cannot_support_anything() -> None:
    ok, _ = _verifier(_Scorer({}), passages=[]).supports("176 F.3d 1132", "the claim")
    assert not ok


def test_a_broken_scorer_is_a_failure_not_a_pass() -> None:
    class _Broken:
        threshold = 0.5

        def score(self, proposition: str, passage: str) -> float:
            raise RuntimeError("model unavailable")

    ok, detail = _verifier(_Broken(), passages=["text"]).supports("176 F.3d 1132", "c")
    assert not ok
    assert "support scorer failed" in detail


def test_an_unresolvable_citation_supports_nothing() -> None:
    ok, _ = _verifier(_Scorer({"text": 1.0}), passages=["text"]).supports("999 U.S. 999", "c")
    assert not ok
