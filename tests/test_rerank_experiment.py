"""Independent-signal reranking. Offline: no models, no network."""

from __future__ import annotations

from evals.compare_support_scorers import Pair
from evals.rerank_experiment import RERANK_FIELDS, measure, rerank_score, rerank_text


class _Record:
    def __init__(
        self,
        title: str = "",
        passages: list[str] | None = None,
        headnotes: list[str] | None = None,
        court: str = "",
    ) -> None:
        self.title = title
        self.passages = passages or []
        self.headnotes = headnotes or []
        self.court = court
        self.type = "case"
        self.code = ""
        self.section = ""


SECRET = "zygomatic phlogiston"


# --------------------------------------------------------------------------- #
# Independence is the whole point, so it is enforced rather than intended
# --------------------------------------------------------------------------- #
def test_the_reranker_never_reads_passages() -> None:
    """Reading the gate's input would align retrieval with the check and make
    the gate decorative (REMEDIATION §18).
    """
    assert "passages" not in RERANK_FIELDS
    record = _Record(title="A Case", passages=[SECRET], headnotes=["A headnote."])
    assert SECRET not in rerank_text(record)


def test_a_record_whose_only_text_is_passages_scores_on_metadata_alone() -> None:
    """Non-case records have no headnotes, and their passages are what the gate
    reads. Reaching into them for those would break the disjointness.
    """
    record = _Record(title="", passages=["Everything is in here " + SECRET])
    assert rerank_score("everything", record) == 0.0


def test_headnotes_are_the_signal() -> None:
    on_point = _Record(headnotes=["Encryption source code as protected expression."])
    off_point = _Record(headnotes=["Civil aiding and abetting under state law."])
    claim = "Encryption source code is protected expression."
    assert rerank_score(claim, on_point) > rerank_score(claim, off_point)


def test_title_and_court_contribute_when_headnotes_are_absent() -> None:
    record = _Record(title="Bernstein v. Department of Justice", court="9th Cir.")
    assert rerank_score("Bernstein justice", record) > 0.0


def test_a_claim_with_no_content_words_scores_zero_rather_than_dividing_by_zero() -> None:
    assert rerank_score("the and of", _Record(headnotes=["Anything."])) == 0.0


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def _index() -> dict[str, _Record]:
    return {
        "1 U.S. 1": _Record(
            title="On Point",
            headnotes=["Encryption source code as protected expression."],
            passages=["Source code is expression protected by the First Amendment."],
        ),
        "2 U.S. 2": _Record(
            title="Off Point",
            headnotes=["Civil aiding and abetting."],
            passages=["Maryland recognises civil aiding and abetting."],
        ),
    }


def test_reranking_can_recover_a_claim_retrieval_missed() -> None:
    claim = "Source code is expression protected by the First Amendment."
    report = measure([Pair("2 U.S. 2", claim, False)], _index(), "lexical")
    assert report.actual == 0
    assert report.reranked == 1
    assert report.ceiling == 1


def test_agreement_with_the_gates_favourite_is_reported_as_a_warning_signal() -> None:
    """High agreement would mean the 'independent' signal has collapsed into the
    gate's measure, which is the failure this design is avoiding.
    """
    claim = "Source code is expression protected by the First Amendment."
    report = measure([Pair("1 U.S. 1", claim, True)], _index(), "lexical")
    assert report.agrees_with_ceiling == 1
    assert report.outcomes[0].rerank_cite == report.outcomes[0].ceiling_cite


def test_the_three_numbers_are_measured_against_the_same_scorer() -> None:
    """actual, reranked and ceiling must share a measure or they cannot be
    compared -- the mismatch §18 had to correct.
    """
    claim = "Source code is expression protected by the First Amendment."
    report = measure([Pair("2 U.S. 2", claim, False)], _index(), "lexical")
    assert report.actual <= report.reranked <= report.ceiling


def test_a_record_missing_from_the_corpus_is_unsupported_not_an_error() -> None:
    report = measure([Pair("9 U.S. 9", "A claim about something.", False)], _index(), "lexical")
    assert report.actual == 0


# --------------------------------------------------------------------------- #
# The field that made the measurement possible
# --------------------------------------------------------------------------- #
def test_headnotes_survive_the_corpus_loader() -> None:
    """pydantic drops unknown keys, so the rebuilt corpus's headnotes vanished
    on load and the first run of this experiment scored metadata only.
    """
    from legal_research.citations.corpus import CorpusRecord

    record = CorpusRecord(
        id="x",
        type="case",
        title="A Case",
        passages=["Opinion text."],
        headnotes=["A curated summary."],
    )
    assert record.headnotes == ["A curated summary."]
