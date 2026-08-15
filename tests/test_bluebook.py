"""Bluebook formatting, short forms and table of authorities (spec §6)."""

from __future__ import annotations

from legal_research.citations.bluebook import (
    CitationFormatter,
    FootnoteBuilder,
    table_of_authorities,
)
from legal_research.citations.corpus import Corpus


def test_full_case_cite(corpus: Corpus, formatter: CitationFormatter) -> None:
    rec = corpus.get("us-425-185")
    assert formatter.full(rec) == "Ernst & Ernst v. Hochfelder, 425 U.S. 185 (1976)."


def test_full_statute_and_regulation(corpus: Corpus, formatter: CitationFormatter) -> None:
    assert formatter.full(corpus.get("usc-15-78j-b")) == "15 U.S.C. § 78j(b) (2018)."
    assert formatter.full(corpus.get("cfr-17-240-10b-5")) == "17 C.F.R. § 240.10b-5 (2023)."


def test_full_secondary_cite(corpus: Corpus, formatter: CitationFormatter) -> None:
    cite = formatter.full(corpus.get("sec-basel-iii-2011"))
    assert cite.startswith("Basel Comm. on Banking Supervision, Basel III")
    assert cite.endswith("(2011).")


def test_short_form_case(corpus: Corpus, formatter: CitationFormatter) -> None:
    assert formatter.short(corpus.get("us-445-222")) == "Chiarella, 445 U.S. at 222."


def test_footnote_builder_id_and_short_forms(corpus: Corpus, formatter: CitationFormatter) -> None:
    chiarella = corpus.get("us-445-222")
    hochfelder = corpus.get("us-425-185")
    fb = FootnoteBuilder(formatter)

    n1, t1 = fb.cite(chiarella)
    n2, t2 = fb.cite(chiarella)  # immediate repeat -> Id.
    n3, t3 = fb.cite(hochfelder)
    n4, t4 = fb.cite(chiarella)  # later repeat -> short form

    assert (n1, n2, n3, n4) == (1, 2, 3, 4)
    assert t1 == "Chiarella v. United States, 445 U.S. 222 (1980)."
    assert t2 == "Id."
    assert t3.startswith("Ernst & Ernst v. Hochfelder")
    assert t4 == "Chiarella, 445 U.S. at 222."


def test_footnote_builder_supra_for_secondary(corpus: Corpus, formatter: CitationFormatter) -> None:
    basel = corpus.get("sec-basel-iii-2011")
    hochfelder = corpus.get("us-425-185")
    fb = FootnoteBuilder(formatter)

    fb.cite(basel)  # note 1, full
    fb.cite(hochfelder)  # note 2, full
    _, t3 = fb.cite(basel)  # note 3, supra note 1
    assert t3 == "Basel Comm. on Banking Supervision, supra note 1."


def test_table_of_authorities_groups(corpus: Corpus, formatter: CitationFormatter) -> None:
    records = [
        corpus.get("us-425-185"),
        corpus.get("usc-15-78j-b"),
        corpus.get("cfr-17-240-10b-5"),
        corpus.get("sec-basel-iii-2011"),
    ]
    toa = table_of_authorities(records, formatter)
    assert set(toa) == {"Cases", "Statutes", "Regulations", "Other Authorities"}
    assert toa["Cases"] == ["Ernst & Ernst v. Hochfelder, 425 U.S. 185 (1976)"]
