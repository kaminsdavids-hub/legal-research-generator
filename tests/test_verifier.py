"""Adversarial verifier (spec §9.1, §4)."""

from __future__ import annotations

from legal_research.citations.corpus import Corpus
from legal_research.citations.report import build_verification_report
from legal_research.citations.verifier import CitationVerifier
from legal_research.models import Citation, CiteStatus


def _cite(**kw) -> Citation:
    base = dict(
        id="c",
        record_id="us-425-185",
        proposition="Section 10(b) and Rule 10b-5 require a showing of scienter.",
        from_retrieval=True,
        status=CiteStatus.PENDING,
    )
    base.update(kw)
    return Citation(**base)


def test_known_good_citation_verifies(verifier: CitationVerifier) -> None:
    result = verifier.verify(_cite(id="c1"))
    assert result.status is CiteStatus.VERIFIED
    assert result.supporting_passage


def test_misattributed_holding_is_removed(verifier: CitationVerifier) -> None:
    # Chiarella is about duty-to-disclose, not scienter. Citing it for scienter is a
    # misattributed holding and must be removed.
    result = verifier.verify(_cite(id="c2", record_id="us-445-222"))
    assert result.status is CiteStatus.REMOVED
    assert "misattributed" in result.reason


def test_fabricated_authority_does_not_resolve(verifier: CitationVerifier) -> None:
    result = verifier.verify(_cite(id="c3", record_id="us-000-000"))
    assert result.status is CiteStatus.REMOVED
    assert "does not resolve" in result.reason


def test_non_retrieval_citation_is_removed(verifier: CitationVerifier) -> None:
    result = verifier.verify(_cite(id="c4", from_retrieval=False))
    assert result.status is CiteStatus.REMOVED
    assert "retriever" in result.reason


def test_inexact_quote_is_removed(verifier: CitationVerifier) -> None:
    result = verifier.verify(_cite(id="c5", quote="a quote that never appears in the opinion"))
    assert result.status is CiteStatus.REMOVED
    assert "verbatim" in result.reason


def test_exact_quote_verifies(verifier: CitationVerifier) -> None:
    result = verifier.verify(
        _cite(id="c6", quote="require a showing of scienter")
    )
    assert result.status is CiteStatus.VERIFIED


def test_verify_all_updates_statuses_and_report(verifier: CitationVerifier, corpus: Corpus) -> None:
    good = _cite(id="good")
    bad = _cite(id="bad", record_id="us-445-222")
    citations = [good, bad]
    results = verifier.verify_all(citations)

    assert good.status is CiteStatus.VERIFIED
    assert bad.status is CiteStatus.REMOVED

    report = build_verification_report(citations, results, corpus)
    assert "VERIFIED" in report
    assert "REMOVED" in report
    assert "good" in report and "bad" in report
