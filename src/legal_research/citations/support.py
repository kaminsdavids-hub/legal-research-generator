"""Does a passage actually *support* a proposition?

This is the Verifier's most important question — a citation that resolves but is
misattributed is exactly the failure mode we exist to catch. Three scorers sit
behind one :class:`SupportScorer` protocol:

* :class:`LexicalSupportScorer` — content-token recall. Zero deps, deterministic;
  used by CI and as a graceful fallback when semantic deps are unavailable.
  A blunt instrument: it can be fooled by shared vocabulary in either direction.
* :class:`EmbeddingSupportScorer` — cosine similarity of sentence embeddings.
  Captures paraphrase, but similarity is not entailment.
* :class:`NliSupportScorer` — a cross-encoder NLI model scoring P(passage ⊨
  proposition). This is true semantic entailment and the default setting in the
  live Spark profile.

The two semantic scorers require the optional ``gpu`` extra; if the deps are
missing, :func:`build_support_scorer` falls back to lexical, mirroring the
retriever's graceful degradation so the pipeline always runs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from .retriever import _tokens

LEXICAL_THRESHOLD = 0.34
SEMANTIC_THRESHOLD = 0.55
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

#: Clause boundaries used to decompose a proposition into separately citable parts.
CLAUSE_SPLIT_RE = re.compile(
    r";|,\s+(?:which|where|while|although|whereas)\s+|\s+(?:and|but|yet|whereas)\s+"
)

#: A clause with fewer content tokens than this asserts too little to test.
MIN_CLAUSE_TOKENS = 4

#: Cap on candidates scored per citation. Semantic scorers are the slow path, so
#: the proposition plus its three longest clauses is a deliberate bound on cost.
MAX_SUPPORT_CANDIDATES = 4


def candidate_propositions(proposition: str) -> list[str]:
    """Decompose a proposition into the parts a single citation might support.

    Legal writing routinely joins a supported holding to a second claim that lives
    in a *different* authority::

        Private plaintiffs may not maintain aiding-and-abetting suits under
        Section 10(b), and primary liability requires a showing of scienter.

    The first clause is near-verbatim in *Central Bank*; the second belongs to
    *Hochfelder*. Entailment against the conjunction therefore scores ~0.00 even
    though the citation is perfectly correct for the clause it follows. Scoring the
    clauses separately is how citation actually works: a cite supports the
    proposition it is attached to, not every claim in the sentence.

    The trade-off is deliberate: a conjunction is accepted when *any* clause is
    supported. Per-clause citation is the writer's job; the verifier's job is to
    reject authority that supports *nothing* in the sentence.
    """

    whole = " ".join(proposition.split())
    candidates = [whole]
    clauses = [
        " ".join(clause.split())
        for clause in CLAUSE_SPLIT_RE.split(whole)
        if len(_tokens(clause)) >= MIN_CLAUSE_TOKENS
    ]
    for clause in sorted(clauses, key=len, reverse=True):
        if clause not in candidates:
            candidates.append(clause)
    return candidates[:MAX_SUPPORT_CANDIDATES]


def best_support(
    scorer: SupportScorer,
    proposition: str,
    passages: Sequence[str],
    *,
    threshold: float | None = None,
) -> tuple[float, str]:
    """Best support score across ``passages`` and the proposition's clauses.

    Returns ``(score, passage)``. Scoring stops as soon as ``threshold`` is met,
    since further work cannot change the verdict.
    """

    cutoff = scorer.threshold if threshold is None else threshold
    best = 0.0
    best_passage = ""
    for candidate in candidate_propositions(proposition):
        for passage in passages:
            score = scorer.score(candidate, passage)
            if score > best:
                best, best_passage = score, passage
            if best >= cutoff:
                return best, best_passage
    return best, best_passage


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
        self._nemotron_prompt_mode = "nemotron-3-embed" in model.lower()
        self._model = SentenceTransformer(model)

    def _encode_passage(self, passage: str):
        encode_document = getattr(self._model, "encode_document", None)
        if callable(encode_document):
            return encode_document([passage], normalize_embeddings=True)
        if self._nemotron_prompt_mode:
            passage = f"document: {passage}"
        return self._model.encode([passage], normalize_embeddings=True)

    def _encode_proposition(self, proposition: str):
        encode_query = getattr(self._model, "encode_query", None)
        if callable(encode_query):
            return encode_query([proposition], normalize_embeddings=True)
        if self._nemotron_prompt_mode:
            proposition = f"query: {proposition}"
        return self._model.encode([proposition], normalize_embeddings=True)

    def score(self, proposition: str, passage: str) -> float:
        import numpy as np

        p_emb = self._encode_passage(passage)
        q_emb = self._encode_proposition(proposition)
        arr = np.asarray([p_emb[0], q_emb[0]], dtype="float32")
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
