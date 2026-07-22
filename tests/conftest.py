"""Shared fixtures. The whole suite runs on the mock backend: no GPU, no network."""

from __future__ import annotations

import itertools

import pytest

from legal_research.blackboard import Blackboard
from legal_research.citations.bluebook import build_formatter
from legal_research.citations.corpus import Corpus, load_corpus
from legal_research.citations.retriever import MockRetriever, Retriever
from legal_research.citations.verifier import CitationGuard, CitationVerifier
from legal_research.config import Settings
from legal_research.pipeline import LegalResearchPipeline

CORPUS_PATH = "data/corpus/sample_corpus.jsonl"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_mode="mock",
        retriever_mode="mock",
        pdf_renderer="html",
        citation_style="bluebook",
        corpus_path=CORPUS_PATH,
    )


@pytest.fixture
def corpus() -> Corpus:
    return load_corpus(CORPUS_PATH)


@pytest.fixture
def retriever(corpus: Corpus) -> Retriever:
    return MockRetriever(corpus)


@pytest.fixture
def id_gen():
    counter = itertools.count(1)
    return lambda: f"cite-{next(counter):03d}"


@pytest.fixture
def guard(retriever: Retriever, id_gen) -> CitationGuard:
    return CitationGuard(retriever, id_gen)


@pytest.fixture
def verifier(corpus: Corpus) -> CitationVerifier:
    return CitationVerifier(corpus)


@pytest.fixture
def formatter():
    return build_formatter("bluebook")


@pytest.fixture
def pipeline(settings: Settings) -> LegalResearchPipeline:
    return LegalResearchPipeline(settings)


@pytest.fixture
def blackboard() -> Blackboard:
    return Blackboard(session_id="test-session", title="Test Paper")
