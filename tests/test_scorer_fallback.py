"""A degraded support scorer must announce itself. REMEDIATION §21."""

from __future__ import annotations

import logging

import pytest

from legal_research.citations import support
from legal_research.citations.support import (
    LEXICAL_THRESHOLD,
    LexicalSupportScorer,
    build_support_scorer,
)
from legal_research.config import Settings


def _settings(**kw: object) -> Settings:
    return Settings(**kw)  # type: ignore[arg-type]


def test_a_silent_fallback_is_what_removed_61_of_61_citations(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The API ran lexical while configured for NLI for its whole lifetime, and
    nothing said so. The fallback stays; the silence does not.
    """
    monkeypatch.setattr(
        support,
        "NliSupportScorer",
        lambda *a, **k: (_ for _ in ()).throw(OSError("model not cached")),
    )
    with caplog.at_level(logging.WARNING, logger=support.__name__):
        scorer = build_support_scorer(_settings(support_scorer="nli"))

    assert isinstance(scorer, LexicalSupportScorer)
    assert scorer.threshold == LEXICAL_THRESHOLD
    assert caplog.records, "a fallback must be logged"
    message = caplog.records[0].getMessage()
    assert "nli" in message
    assert "OSError" in message and "model not cached" in message
    assert "not comparable" in message, "the warning must say the scores changed meaning"


def test_an_embedding_fallback_is_announced_too(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        support,
        "EmbeddingSupportScorer",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no sentence_transformers")),
    )
    with caplog.at_level(logging.WARNING, logger=support.__name__):
        build_support_scorer(_settings(support_scorer="embedding"))
    assert any("embedding" in r.getMessage() for r in caplog.records)


def test_asking_for_lexical_is_not_a_fallback_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a *degradation* is worth a warning; a chosen scorer is not."""
    with caplog.at_level(logging.WARNING, logger=support.__name__):
        scorer = build_support_scorer(_settings(support_scorer="lexical"))
    assert isinstance(scorer, LexicalSupportScorer)
    assert not caplog.records


def test_a_scorer_that_builds_cleanly_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _Fake:
        name = "nli"
        threshold = 0.55

        def __init__(self, *a: object, **k: object) -> None: ...

        def score(self, proposition: str, passage: str) -> float:
            return 1.0

    monkeypatch.setattr(support, "NliSupportScorer", _Fake)
    with caplog.at_level(logging.WARNING, logger=support.__name__):
        scorer = build_support_scorer(_settings(support_scorer="nli"))
    assert isinstance(scorer, _Fake)
    assert not caplog.records
