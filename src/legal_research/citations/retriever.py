"""Retrieval over the legal corpus.

Two backends behind one :class:`Retriever` protocol:

* :class:`MockRetriever` — deterministic lexical scorer with zero dependencies.
  Used by CI and as a graceful fallback when dense retrieval deps are absent.
* :class:`FaissRetriever` — real local FAISS + sentence-transformers embeddings,
  used by the default Spark profile when ``LRG_RETRIEVER_MODE=faiss``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..models import RetrievedPassage
from .corpus import Corpus, load_corpus

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "that", "this",
    "is", "are", "be", "as", "by", "with", "any", "such", "under", "section",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]: ...


class MockRetriever(Retriever):
    """Lexical retriever using cosine similarity over TF vectors with IDF weights."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._passages: list[tuple[str, str, Counter[str]]] = []
        doc_freq: Counter[str] = Counter()
        for record in corpus.records:
            for i, passage in enumerate(record.passages):
                tf = Counter(_tokens(passage))
                locator = f"p.{i + 1}"
                self._passages.append((record.id, f"{passage}||{locator}", tf))
                for term in tf:
                    doc_freq[term] += 1
        n = max(1, len(self._passages))
        self._idf = {term: math.log(1 + n / (1 + df)) for term, df in doc_freq.items()}

    def _weight(self, tf: Counter[str]) -> dict[str, float]:
        return {term: count * self._idf.get(term, 0.0) for term, count in tf.items()}

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        q_tf = Counter(_tokens(query))
        if not q_tf:
            return []
        q_vec = self._weight(q_tf)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[RetrievedPassage] = []
        for record_id, passage_with_loc, tf in self._passages:
            p_vec = self._weight(tf)
            p_norm = math.sqrt(sum(v * v for v in p_vec.values())) or 1.0
            dot = sum(q_vec.get(term, 0.0) * val for term, val in p_vec.items())
            score = dot / (q_norm * p_norm)
            if score <= 0:
                continue
            passage, locator = passage_with_loc.split("||", 1)
            scored.append(
                RetrievedPassage(
                    record_id=record_id, score=round(score, 6), text=passage, locator=locator
                )
            )
        scored.sort(key=lambda p: (-p.score, p.record_id, p.locator))
        return scored[:k]


class FaissRetriever(Retriever):  # pragma: no cover - exercised only on the Spark
    """Local dense retriever. Requires the optional ``gpu`` extra."""

    def __init__(self, corpus: Corpus, embed_model: str) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._corpus = corpus
        self._embed_model = embed_model
        self._nemotron_prompt_mode = "nemotron-3-embed" in embed_model.lower()
        self._model = SentenceTransformer(embed_model)
        self._meta: list[tuple[str, str, str]] = []
        texts: list[str] = []
        for record in corpus.records:
            for i, passage in enumerate(record.passages):
                self._meta.append((record.id, passage, f"p.{i + 1}"))
                texts.append(passage)
        embeddings = self._encode_documents(texts)
        embeddings = np.asarray(embeddings, dtype="float32")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _encode_documents(self, texts: list[str]):
        encode_document = getattr(self._model, "encode_document", None)
        if callable(encode_document):
            return encode_document(texts, normalize_embeddings=True)
        if self._nemotron_prompt_mode:
            texts = [f"document: {text}" for text in texts]
        return self._model.encode(texts, normalize_embeddings=True)

    def _encode_queries(self, texts: list[str]):
        encode_query = getattr(self._model, "encode_query", None)
        if callable(encode_query):
            return encode_query(texts, normalize_embeddings=True)
        if self._nemotron_prompt_mode:
            texts = [f"query: {text}" for text in texts]
        return self._model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        import numpy as np

        q = self._encode_queries([query])
        q = np.asarray(q, dtype="float32")
        scores, idx = self._index.search(q, min(k, len(self._meta)))
        out: list[RetrievedPassage] = []
        for score, i in zip(scores[0], idx[0], strict=False):
            if i < 0:
                continue
            record_id, passage, locator = self._meta[int(i)]
            out.append(
                RetrievedPassage(
                    record_id=record_id, score=float(score), text=passage, locator=locator
                )
            )
        return out


def build_retriever(
    settings: Settings | None = None, corpus: Corpus | None = None
) -> Retriever:
    s = settings or get_settings()
    c = corpus or load_corpus(s.corpus_path)
    if s.retriever_mode == "faiss":
        try:
            return FaissRetriever(c, s.embed_model)
        except Exception:  # noqa: BLE001 - fall back to mock if deps missing
            return MockRetriever(c)
    return MockRetriever(c)
