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

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from .retriever import _tokens

logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------- #
# The second relation
# --------------------------------------------------------------------------- #
class SupportRelation(str, Enum):
    """How a passage supports a proposition, if it does.

    Entailment was the only relation, and a probe of a live run showed what that
    costs. The paper's central claim -- "publishing open model weights is
    protected expression, and the EAR may not treat that publication as a deemed
    export" -- retrieved, first, the EAR's published-information exclusion:

        Information is published, and so is not subject to the EAR, when it has
        been made available to the public without restrictions upon its further
        dissemination.

    That is the authority the claim rests on. NLI scored it **0.043** against a
    0.55 threshold, and the lexical scorer 0.250 against 0.34 -- so this was not
    a choice of scorer, it was the relation. The passage states a *rule*; the
    proposition *applies* it to a new object, and entailment holds only with the
    premise that weights are "information" -- which is the contested question the
    paper exists to argue. Entailment asks "does this passage make the claim
    true"; citation asks "is this the authority the claim rests on". They come
    apart exactly where the analysis is interesting.

    So there are two relations, and they are never merged. RULE_SUPPORT is the
    weaker one, it is labelled everywhere it appears, and a reader is told which
    inference they are being asked to accept.
    """

    #: The passage makes the proposition true. The original, strict test.
    ENTAILED = "entailed"
    #: The passage states the rule the proposition applies, does not contradict
    #: it, and is about the same subject. The application step is the author's,
    #: and the report says so.
    RULE_SUPPORT = "rule_support"
    NONE = "none"


@dataclass(frozen=True)
class SupportAssessment:
    relation: SupportRelation
    score: float
    passage: str = ""
    #: P(contradiction) from the NLI model, when one is available. A passage that
    #: contradicts the proposition is never rule support however topical it is --
    #: this is the check that stops "the EAR's deemed-export rule applies" being
    #: read as support for "the EAR may not treat this as a deemed export".
    contradiction: float = 0.0
    note: str = ""

    @property
    def supported(self) -> bool:
        return self.relation is not SupportRelation.NONE


#: Normative language: the passage states a rule rather than reciting facts.
#: Without this, any topical passage would qualify, and "the parties stipulated
#: to the following facts" is not authority for anything.
RULE_LANGUAGE = re.compile(
    r"\b(?:is|are)\s+not\s+subject\s+to\b|\bshall\b|\bmust\b|\bmay\s+not\b|"
    r"\bis\s+required\b|\brequires\b|\bapplies\s+to\b|\bdoes\s+not\s+apply\b|"
    r"\bis\s+defined\s+as\b|\bmeans\s+(?:any|a|the)\b|\bno\s+\w+\s+may\b|"
    r"\bauthorize[sd]?\b|\bprohibit(?:s|ed)?\b|\bexempt(?:s|ed|ion)?\b|"
    r"\bis\s+published\b|\bwhen\s+it\s+has\s+been\b",
    re.IGNORECASE,
)

#: Above this, the passage argues *against* the proposition and cannot support
#: it. Deliberately strict: the whole risk of a second, looser relation is that
#: it admits a contrary authority as if it agreed.
MAX_CONTRADICTION = 0.10

#: Topicality is **the retriever's** judgement, not a lexical one, so there is no
#: overlap threshold here. Measuring it lexically was tried and was wrong on the
#: exact pair this relation exists for: the thesis and the EAR's
#: published-information exclusion share almost no vocabulary (0.125 overlap),
#: because the connection is semantic -- "EAR" against "Export Administration
#: Regulations", model weights against "information". A word-overlap gate set
#: high enough to exclude noise excluded the right answer.
#:
#: So rule support is offered only for a passage the retriever returned *for
#: this proposition*. Retrieval is a dense semantic index and is the component
#: whose job this is; asking it again in a cruder way would be both redundant
#: and worse.


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


@runtime_checkable
class ContradictionAware(Protocol):
    """A scorer that can also report P(contradiction), not only P(entailment).

    Rule support needs it: the difference between "states the rule this claim
    applies" and "states the rule this claim denies" is the whole safety margin,
    and an entailment score near zero does not distinguish them. A scorer
    without this cannot offer the second relation, and :func:`assess_support`
    says so rather than guessing.
    """

    def contradiction(self, proposition: str, passage: str) -> float: ...


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
        return self._probabilities(proposition, passage)[1]

    def contradiction(self, proposition: str, passage: str) -> float:
        return self._probabilities(proposition, passage)[0]

    def _probabilities(self, proposition: str, passage: str) -> tuple[float, float, float]:
        """``(contradiction, entailment, neutral)`` for premise=passage."""

        import numpy as np

        logits = self._model.predict([(passage, proposition)])
        row = np.asarray(logits, dtype="float32").reshape(-1)
        # cross-encoder/nli-* label order is [contradiction, entailment, neutral].
        if row.shape[0] < 3:
            entail = max(0.0, min(1.0, float(row.reshape(-1)[0])))
            return 0.0, entail, 1.0 - entail
        exp = np.exp(row - row.max())
        probs = exp / exp.sum()
        return float(probs[0]), float(probs[1]), float(probs[2])


def assess_support(
    scorer: SupportScorer,
    proposition: str,
    passages: Sequence[str],
    *,
    threshold: float | None = None,
    rule_relation: bool = True,
    retrieved_passage: str = "",
) -> SupportAssessment:
    """Entailment first; rule support only if entailment fails.

    The order matters and is not an optimisation. A passage that entails the
    proposition is the stronger finding and must be reported as such, so the
    weaker relation is never reached when the stronger one holds.

    Rule support requires all four:

    1. **Entailment failed.** Otherwise this is the wrong label.
    2. **The retriever returned this passage for this proposition**
       (``retrieved_passage``) -- that is the topicality evidence, and without
       it the relation is unavailable rather than guessed at.
    3. **The passage states a rule.** Recited facts are not authority.
    4. **It does not contradict the proposition** (``MAX_CONTRADICTION``). This
       is the one that keeps a contrary authority out, and it needs a
       :class:`ContradictionAware` scorer -- a lexical fallback cannot offer
       this relation at all, and returns NONE rather than guessing.

    Each condition is checked by the component that is good at it: the retriever
    for topicality, the NLI model for contradiction, a regex for whether the
    text is written as a rule at all.
    """

    cutoff = scorer.threshold if threshold is None else threshold
    best, best_passage = best_support(scorer, proposition, passages, threshold=cutoff)
    if best >= cutoff:
        return SupportAssessment(
            relation=SupportRelation.ENTAILED, score=best, passage=best_passage
        )

    if not rule_relation:
        return SupportAssessment(SupportRelation.NONE, best, best_passage)
    if not isinstance(scorer, ContradictionAware):
        return SupportAssessment(
            SupportRelation.NONE,
            best,
            best_passage,
            note=(
                f"{type(scorer).__name__} cannot report contradiction, so the "
                "rule-support relation is unavailable and only entailment was tested"
            ),
        )

    candidate = _matching_passage(retrieved_passage, passages)
    if candidate is None:
        return SupportAssessment(
            SupportRelation.NONE,
            best,
            best_passage,
            note=(
                "no retrieved passage was supplied, so there is no evidence this "
                "source is about this proposition and rule support was not tested"
            ),
        )
    if not RULE_LANGUAGE.search(candidate):
        return SupportAssessment(
            SupportRelation.NONE,
            best,
            best_passage,
            note="the retrieved passage recites facts rather than stating a rule",
        )

    clauses = candidate_propositions(proposition)
    contra = max(scorer.contradiction(clause, candidate) for clause in clauses)
    if contra > MAX_CONTRADICTION:
        return SupportAssessment(
            SupportRelation.NONE,
            best,
            candidate,
            contradiction=contra,
            note=(
                f"the retrieved rule contradicts the proposition "
                f"(contradiction={contra:.2f}); it is contrary authority, not support"
            ),
        )

    return SupportAssessment(
        relation=SupportRelation.RULE_SUPPORT,
        score=max(scorer.score(clause, candidate) for clause in clauses),
        passage=candidate,
        contradiction=contra,
        note=(
            "states the rule the proposition applies; the application to these "
            "facts is the author's and is not verified here"
        ),
    )


def _matching_passage(retrieved: str, passages: Sequence[str]) -> str | None:
    """The record passage the retriever returned, matched on normalised text."""

    if not retrieved.strip():
        return None
    needle = " ".join(retrieved.split()).casefold()
    for passage in passages:
        haystack = " ".join(passage.split()).casefold()
        if needle == haystack or needle in haystack or haystack in needle:
            return passage
    return None


def build_support_scorer(settings: Settings | None = None) -> SupportScorer:
    """Select a scorer from settings, falling back to lexical if deps are absent.

    **A fallback is logged at WARNING, never silent.** This function returned a
    lexical scorer while the configuration said ``nli``: the API process had been
    started at a moment when the cross-encoder could not be constructed, kept the
    fallback for its whole lifetime, and nothing anywhere said so. A full
    pipeline run then removed 61 of 61 citations against a 0.34 threshold nobody
    had configured, and reported the paper shippable. The same silent-degradation
    shape has now cost this repository three separate investigations
    (REMEDIATION §11.6, §14, §21).

    Falling back is right -- a missing extra should degrade rather than crash.
    Doing it quietly is what is wrong.
    """

    s = settings or get_settings()
    mode = (s.support_scorer or "lexical").lower()
    override = s.support_threshold

    def _fallback(reason: BaseException) -> SupportScorer:
        logger.warning(
            "support scorer %r was requested but could not be built (%s: %s); "
            "falling back to lexical at threshold %s. Support is now token "
            "recall, not entailment, and scores are not comparable to a "
            "semantic run.",
            mode,
            type(reason).__name__,
            str(reason)[:200],
            override or LEXICAL_THRESHOLD,
        )
        return LexicalSupportScorer(override or LEXICAL_THRESHOLD)

    if mode == "embedding":
        try:
            return EmbeddingSupportScorer(s.embed_model, override or SEMANTIC_THRESHOLD)
        except Exception as exc:  # noqa: BLE001 - degrade rather than crash, but say so
            return _fallback(exc)
    if mode == "nli":
        try:
            return NliSupportScorer(s.nli_model, override or SEMANTIC_THRESHOLD)
        except Exception as exc:  # noqa: BLE001 - degrade rather than crash, but say so
            return _fallback(exc)
    return LexicalSupportScorer(override or LEXICAL_THRESHOLD)
