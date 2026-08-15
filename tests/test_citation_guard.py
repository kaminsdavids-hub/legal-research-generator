"""Citation guard: the hard anti-hallucination rule (spec §9.1, §4)."""

from __future__ import annotations

import pytest

from legal_research.citations.verifier import CitationGuard, GroundingError
from legal_research.models import Citation, CiteStatus


def test_guard_grounds_a_supported_proposition(guard: CitationGuard) -> None:
    citation = guard.ground("Section 10(b) and Rule 10b-5 require a showing of scienter.")
    assert citation.from_retrieval is True
    assert citation.record_id  # resolved to a corpus record
    assert citation.supporting_passage


def test_guard_rejects_authority_absent_from_retrieval(guard: CitationGuard) -> None:
    # Nothing in the corpus supports this; the guard must refuse to invent a cite.
    with pytest.raises(GroundingError):
        guard.ground("Quantum chromodynamics governs medieval tapestry taxation.")


def test_assert_grounded_blocks_non_retrieval_citation() -> None:
    fabricated = Citation(
        id="cite-x",
        record_id="us-123-456",
        proposition="A fabricated proposition asserted from model weights.",
        from_retrieval=False,
        status=CiteStatus.PENDING,
    )
    with pytest.raises(GroundingError):
        CitationGuard.assert_grounded(fabricated)
