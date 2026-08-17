"""Which authority a slot ends up resting on.

Three stages decide that, and each one was observed getting it wrong on the
shipped corpus before these tests existed:

* ``propose`` ranks *passages*, so a long opinion can occupy the whole top-k and
  crowd out a regulation that scored just below it.
* ``_select_candidate`` takes the top candidate, which on an export-control
  question meant a First Amendment case was attached to every proposition while
  the regulation the question was about sat unused at rank two.
* ``confirm`` decides whether the corpus may warrant an authority CourtListener
  cannot adjudicate — and must refuse a rule that is no longer in force, which
  is the one case where a wrong answer puts rescinded law in front of a reader
  wearing a verified badge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from modules.dialectic.engine import DialecticChat
from modules.dialectic.models import CitationSlot
from modules.dialectic.service import _CorpusCiteRetriever

CANDIDATES = ["176 F.3d 1132", "15 C.F.R. 734.13(b)", "15 C.F.R. 734.7", "209 F.3d 481"]


def _slot(hint: str) -> CitationSlot:
    return CitationSlot(proposition="a proposition", court_hint=hint)


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        # Instrument named, no court named: take the regulation over the
        # higher-scoring case. Both hints are verbatim from a live exchange.
        ("Interpretation of statutory language in export control regulations", "15 C.F.R. 734.13(b)"),
        ("National security and foreign policy exceptions in export control regulations", "15 C.F.R. 734.13(b)"),
        # Court named: similarity ranking stands, whatever else the hint says.
        ("Supreme Court precedent on the First Amendment and copyright law", "176 F.3d 1132"),
        ("Circuit split on the applicability of the First Amendment to copyright law", "176 F.3d 1132"),
        # The trap, also verbatim from a live exchange: contains "regulation"
        # and wants a case. "regulation of speech" is a subject, not an
        # instrument, which is why bare "regulation" is not a signal.
        ("First Amendment analysis of government regulation of speech", "176 F.3d 1132"),
        # Nothing to go on: never override the ranking on a guess.
        ("", "176 F.3d 1132"),
        ("Authority on the question", "176 F.3d 1132"),
    ],
)
def test_the_hint_decides_only_when_it_is_unambiguous(hint: str, expected: str) -> None:
    assert DialecticChat._select_candidate(_slot(hint), CANDIDATES) == expected


def test_an_instrument_hint_with_no_instrument_retrieved_keeps_the_ranking() -> None:
    """Wanting a regulation is not a reason to invent one.

    A hint can ask for a statute while retrieval offers only cases. Returning
    nothing would strip the slot of authority it did have; returning a case is
    the honest outcome and leaves the marker to say it was never confirmed.
    """
    only_cases = ["176 F.3d 1132", "209 F.3d 481"]
    hint = "Interpretation of statutory language in export control regulations"
    assert DialecticChat._select_candidate(_slot(hint), only_cases) == "176 F.3d 1132"


def test_no_candidates_is_not_an_error() -> None:
    assert DialecticChat._select_candidate(_slot("anything"), []) == ""


# --- retrieval: one record must not spend the whole budget --------------------


@dataclass
class _Rec:
    id: str
    type: str = "case"
    status: str = "in_force"
    unverified: bool = False
    volume: int | None = None
    reporter: str = ""
    page: int | None = None
    code: str = ""
    section: str = ""
    status_note: str = ""


@dataclass
class _Hit:
    record_id: str


@dataclass
class _Corpus:
    records: list[_Rec] = field(default_factory=list)

    def get(self, record_id: str) -> _Rec | None:
        return next((r for r in self.records if r.id == record_id), None)


class _PassageRetriever:
    """Returns hits per passage, so one record can repeat — like the real one."""

    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.last_k: int | None = None

    def search(self, query: str, k: int = 5) -> list[_Hit]:  # noqa: ARG002
        self.last_k = k
        return [_Hit(record_id=r) for r in self._order[:k]]


def _fixture() -> tuple[_PassageRetriever, _Corpus]:
    corpus = _Corpus(
        [
            _Rec("case-a", volume=176, reporter="F.3d", page=1132),
            _Rec("reg-a", type="regulation", code="15 C.F.R.", section="734.13(b)"),
            _Rec("reg-b", type="regulation", code="15 C.F.R.", section="734.7"),
            _Rec("case-b", volume=209, reporter="F.3d", page=481),
        ]
    )
    # The measured shape: one opinion takes eight of the first ten passages.
    order = ["case-a"] * 8 + ["reg-a", "reg-b", "case-b"]
    return _PassageRetriever(order), corpus


def test_one_record_cannot_crowd_out_the_rest() -> None:
    retriever, corpus = _fixture()
    cites = _CorpusCiteRetriever(retriever, corpus, k=5).propose("hint", "proposition")

    assert cites[0] == "176 F.3d 1132", "highest-scoring record still leads"
    assert "15 C.F.R. 734.13(b)" in cites, "a regulation below the flood must survive"
    assert "15 C.F.R. 734.7" in cites
    assert len(cites) == len(set(cites)), "one entry per authority"


def test_the_passage_window_is_wider_than_the_record_budget() -> None:
    """Over-fetching is what gives the record-level cut anything to choose from."""
    retriever, corpus = _fixture()
    _CorpusCiteRetriever(retriever, corpus, k=5).propose("hint", "proposition")
    assert retriever.last_k is not None and retriever.last_k > 5


# --- confirmation: the corpus warrants only what it actually holds ------------


def test_confirm_accepts_in_force_authority() -> None:
    _, corpus = _fixture()
    assert _CorpusCiteRetriever(_PassageRetriever([]), corpus).confirm("15 C.F.R. 734.7")


def test_confirm_refuses_authority_that_is_no_longer_in_force() -> None:
    """The failure this guards against puts rescinded law behind a green badge."""
    corpus = _Corpus(
        [_Rec("fedreg", type="regulation", status="rescinded", code="90 Fed. Reg.", section="4544")]
    )
    r = _CorpusCiteRetriever(_PassageRetriever([]), corpus)
    assert r.confirm("90 Fed. Reg. 4544") is False
    # The two capabilities must agree about the same record, or a cite could be
    # refused confirmation while going unflagged.
    assert "rescinded" in r.annotate("90 Fed. Reg. 4544")


def test_confirm_refuses_a_record_that_admits_it_is_unchecked() -> None:
    corpus = _Corpus(
        [_Rec("x", type="regulation", unverified=True, code="15 C.F.R.", section="999.1")]
    )
    assert _CorpusCiteRetriever(_PassageRetriever([]), corpus).confirm("15 C.F.R. 999.1") is False


def test_confirm_refuses_what_the_corpus_does_not_hold() -> None:
    _, corpus = _fixture()
    r = _CorpusCiteRetriever(_PassageRetriever([]), corpus)
    assert r.confirm("15 C.F.R. 999.1") is False
    assert r.confirm("") is False
