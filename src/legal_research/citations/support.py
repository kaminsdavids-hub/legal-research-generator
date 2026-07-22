"""Does a passage actually *support* a proposition?

This is the Verifier's most important question — a citation that resolves but is
misattributed is exactly the failure mode we exist to catch. Three scorers sit
behind one :class:`SupportScorer` protocol:

* :class:`LexicalSupportScorer` — content-token recall. Zero deps, deterministic;
  the default and what CI uses. A blunt instrument: it can be fooled by shared
  vocabulary in either direction.
* :class:`EmbeddingSupportScorer` — cosine similarity of sentence embeddings.
  Captures paraphrase, but similarity is not entailment.
* :class:`NliSupportScorer` — a cross-encoder NLI model scoring P(passage ⊨
  proposition). This is true semantic entailment and the recommended setting on
  the Spark.

The two semantic scorers require the optional ``gpu`` extra; if the deps are
missing, :func:`build_support_scorer` falls back to lexical, mirroring the
retriever's graceful degradation so the pipeline always runs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from .retriever import _tokens

LEXICAL_THRESHOLD = 0.34
SEMANTIC_THRESHOLD = 0.55
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def lexical_support(proposition: str, passage: str) -> float:
    """Recall of the proposition's content tokens present in the passage (0..1)."""

    prop = set(_tokens(proposition))
    if not prop:
        return 0.0
    passage_tokens = set(_tokens(passage))
    return len(prop & passage_tokens) / len(prop)


@runtime_checkable
class SupportScorer(Protocol):
    """Scores how strongly ``passage`` supports ``proposition`` in ``[0, 1]``."""

    threshold: float

    def score(self, proposition: str, passage: str) -> float: ...


class LexicalSupportScorer:
    name = "lexical"

    def __init__(self, threshold: float = LEXICAL_THRESHOLD) -> None:
        self.threshold = threshold

    def score(self, proposition: str, passage: str) -> float:
        return lexical_support(proposition, passage)


class EmbeddingSupportScorer:  # pragma: no cover - requires the gpu extra / Spark
    name = "embedding"

    def __init__(self, model: str, threshold: float = SEMANTIC_THRESHOLD) -> None:
        from sentence_transformers import SentenceTransformer

        self.threshold = threshold
        self._model = SentenceTransformer(model)

    def score(self, proposition: str, passage: str) -> float:
        import numpy as np

        emb = self._model.encode([passage, proposition], normalize_embeddings=True)
        arr = np.asarray(emb, dtype="float32")
        cos = float(arr[0] @ arr[1])
        return max(0.0, min(1.0, cos))


class NliSupportScorer:  # pragma: no cover - requires the gpu extra / Spark
    """Cross-encoder NLI: returns P(entailment) for premise=passage, hypothesis=prop."""

    name = "nli"

    def __init__(self, model: str = DEFAULT_NLI_MODEL, threshold: float = SEMANTIC_THRESHOLD) -> None:
        from sentence_transformers import CrossEncoder

        self.threshold = threshold
        self._model = CrossEncoder(model)

    def score(self, proposition: str, passage: str) -> float:
        import numpy as np

        logits = self._model.predict([(passage, proposition)])
        row = np.asarray(logits, dtype="float32").reshape(-1)
        # cross-encoder/nli-* label order is [contradiction, entailment, neutral].
        if row.shape[0] < 3:
            return max(0.0, min(1.0, float(row.reshape(-1)[0])))
        exp = np.exp(row - row.max())
        probs = exp / exp.sum()
        return float(probs[1])


def build_support_scorer(settings: Settings | None = None) -> SupportScorer:
    """Select a scorer from settings, falling back to lexical if deps are absent."""

    s = settings or get_settings()
    mode = (s.support_scorer or "lexical").lower()
    override = s.support_threshold

    if mode == "embedding":
        try:
            return EmbeddingSupportScorer(s.embed_model, override or SEMANTIC_THRESHOLD)
        except Exception:  # noqa: BLE001 - fall back to lexical if the extra is missing
            return LexicalSupportScorer(override or LEXICAL_THRESHOLD)
    if mode == "nli":
        try:
            return NliSupportScorer(s.nli_model, override or SEMANTIC_THRESHOLD)
        except Exception:  # noqa: BLE001 - fall back to lexical if the extra is missing
            return LexicalSupportScorer(override or LEXICAL_THRESHOLD)
    return LexicalSupportScorer(override or LEXICAL_THRESHOLD)
