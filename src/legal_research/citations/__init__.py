"""Citation grounding and verification (spec §4).

Hard rule enforced here: no legal authority, quote, or holding may originate from
model weights. Authorities only enter the paper through the retriever, and the
verifier confirms each one both *resolves* to a real corpus record and *supports*
the proposition it is cited for.
"""

from __future__ import annotations

from .bluebook import CitationFormatter, build_formatter
from .corpus import Corpus, CorpusRecord, load_corpus
from .report import build_verification_report
from .retriever import MockRetriever, Retriever, build_retriever
from .support import (
    LexicalSupportScorer,
    SupportScorer,
    build_support_scorer,
)
from .verifier import CitationGuard, CitationVerifier, GroundingError, build_verifier

__all__ = [
    "Corpus",
    "CorpusRecord",
    "load_corpus",
    "Retriever",
    "MockRetriever",
    "build_retriever",
    "CitationVerifier",
    "CitationGuard",
    "GroundingError",
    "build_verifier",
    "SupportScorer",
    "LexicalSupportScorer",
    "build_support_scorer",
    "CitationFormatter",
    "build_formatter",
    "build_verification_report",
]
