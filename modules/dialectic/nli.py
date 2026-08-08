"""Natural-language inference pass for deriving contradiction relations.

The NLI model is a fourth model, distinct from the three debate roles. When one
is configured, :meth:`NLIEvaluator.relate` asks it for a constrained
classification and parses the answer strictly. When none is configured — or the
model's answer does not parse — a deterministic heuristic runs instead so tests
work offline with no GPU and no network.

Which path produced a label is reported alongside it, because a silent downgrade
from the model to the heuristic changes the meaning of the output and must be
visible to a reader rather than invisible.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

Label = Literal["entailment", "contradiction", "neutral"]
Source = Literal["model", "heuristic"]

_LABELS: frozenset[str] = frozenset({"entailment", "contradiction", "neutral"})

_NLI_PROMPT = (
    "You are a natural-language inference classifier for legal propositions.\n"
    "Given a PREMISE and a HYPOTHESIS, decide their relation.\n"
    "Respond ONLY with a JSON object: {\"label\": \"...\"}.\n"
    "label must be exactly one of: entailment, contradiction, neutral.\n"
    "  entailment    — the premise makes the hypothesis true.\n"
    "  contradiction — the premise and the hypothesis cannot both be true.\n"
    "  neutral       — neither; they are about different things, or either "
    "could hold.\n"
    "Emit no prose, no explanation, and no other key. A response that is not "
    "this exact JSON object is discarded."
)


class NLIEvaluator:
    """Fourth-model NLI: classify the relationship between two propositions."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    _ANTONYMS = {
        "applies": "does not apply",
        "applicable": "not applicable",
        "valid": "invalid",
        "constitutional": "unconstitutional",
        "required": "not required",
        "sufficient": "insufficient",
        "establishes": "does not establish",
        "satisfies": "does not satisfy",
        "meets": "does not meet",
        "supports": "does not support",
        "outcome": "opposite outcome",
    }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def relation(self, premise: str, hypothesis: str) -> Label:
        """Return the label alone (see :meth:`relate` for the label's source)."""
        label, _ = self.relate(premise, hypothesis)
        return label

    def relate(self, premise: str, hypothesis: str) -> tuple[Label, Source]:
        """Return ``(label, source)`` where source is ``model`` or ``heuristic``."""
        if self.client is not None:
            label = self._ask_model(premise, hypothesis)
            if label is not None:
                return label, "model"
            # Parse failure: fall back, but say so rather than pretending the
            # model produced this answer.
            return self._heuristic(premise, hypothesis), "heuristic"
        return self._heuristic(premise, hypothesis), "heuristic"

    # --------------------------------------------------------------------- #
    # Model path
    # --------------------------------------------------------------------- #
    def _ask_model(self, premise: str, hypothesis: str) -> Label | None:
        """Ask the NLI model for a constrained label. ``None`` on any failure."""
        client = self.client
        if client is None:
            return None
        messages = [
            {"role": "system", "content": _NLI_PROMPT},
            {
                "role": "user",
                "content": f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}",
            },
        ]
        try:
            raw = client.chat(messages, config={"temperature": 0.0, "seed": 7})
        except Exception:  # noqa: BLE001 - a broken NLI model must not void the turn
            return None
        return self._parse_label(raw)

    @staticmethod
    def _parse_label(raw: Any) -> Label | None:
        """Strictly parse ``{"label": "..."}``; anything else is a failure."""
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        # Some models wrap JSON in markdown fences; strip them, as the debaters' parser does.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or "label" not in data:
            return None
        label = str(data["label"]).strip().lower()
        if label not in _LABELS:
            return None
        return label  # type: ignore[return-value]

    # --------------------------------------------------------------------- #
    # Heuristic path
    # --------------------------------------------------------------------- #
    # Tokens that carry no topical content. Negation words are excluded from
    # content separately, so a bare "not" never counts as shared subject matter.
    _STOP = {
        "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "that", "this",
        "is", "are", "be", "as", "by", "with", "any", "such", "under", "section",
        "it", "its", "from", "at", "into", "over", "after", "before", "when", "where",
        "was", "were", "been", "has", "have", "had", "does", "did", "do", "will",
        "there", "here", "but", "than", "then", "so", "if", "not",
    }

    # Negation markers. Applied to raw (lower-cased) text, never to normalized
    # text: `_normalize` strips apostrophes, which made every contraction
    # alternative unreachable and made "doesn't" read as non-negated.
    _NEGATION_RE = re.compile(
        r"\b(?:not|no|never|cannot|nor|neither|without|lacks?|fails?|failed|"
        r"absent|denies|denied)\b|\b\w+n't\b"
    )

    #: How many tokens after a negation marker fall within its scope.
    _NEGATION_SCOPE = 5

    #: Minimum share of the smaller proposition's content words that must be
    #: shared before two propositions are treated as being about the same thing.
    _MIN_OVERLAP = 0.5

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _raw_tokens(text: str) -> list[str]:
        """Lower-case tokens with apostrophes preserved, so contractions survive."""
        return re.findall(r"[a-z0-9']+", text.lower())

    def _has_negation(self, text: str) -> bool:
        """True when *text* contains a negation marker. Pass RAW text, not normalized."""
        return bool(self._NEGATION_RE.search(text.lower()))

    def _content_words(self, text: str) -> set[str]:
        return {
            w
            for w in self._normalize(text).split()
            if len(w) > 2 and w not in self._STOP
        }

    def _overlap_ratio(self, p_words: set[str], h_words: set[str]) -> float:
        if not p_words or not h_words:
            return 0.0
        return len(p_words & h_words) / min(len(p_words), len(h_words))

    def _negation_scope_terms(self, text: str) -> set[str]:
        """Content words falling within the scope of a negation marker.

        "There is no federal question jurisdiction" negates *jurisdiction*; it
        says nothing about a limitations period discussed in another sentence.
        Requiring the negation to reach a shared term is what stops the old rule
        from collapsing to "exactly one side contains a negation word".
        """
        terms: set[str] = set()
        remaining = 0
        for token in self._raw_tokens(text):
            if self._NEGATION_RE.fullmatch(token):
                remaining = self._NEGATION_SCOPE
                continue
            if remaining:
                remaining -= 1
                word = token.replace("'", "")
                if len(word) > 2 and word not in self._STOP:
                    terms.add(word)
        return terms

    @staticmethod
    def _has_phrase(text: str, phrase: str) -> bool:
        """Word-boundary phrase test over normalized text.

        Not a substring test: "constitutional" must not match inside
        "unconstitutional", or two sentences that both say "unconstitutional"
        would satisfy the antonym rule and be called a contradiction.
        """
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

    def _antonym_contradiction(self, p: str, h: str) -> bool:
        for a, b in self._ANTONYMS.items():
            a_norm = self._normalize(a)
            b_norm = self._normalize(b)
            if (self._has_phrase(p, a_norm) and self._has_phrase(h, b_norm)) or (
                self._has_phrase(p, b_norm) and self._has_phrase(h, a_norm)
            ):
                return True
        return False

    def _heuristic(self, premise: str, hypothesis: str) -> Label:
        p = self._normalize(premise)
        h = self._normalize(hypothesis)

        p_neg = self._has_negation(premise)
        h_neg = self._has_negation(hypothesis)

        p_words = self._content_words(premise)
        h_words = self._content_words(hypothesis)
        overlap = p_words & h_words
        ratio = self._overlap_ratio(p_words, h_words)

        # Exact paraphrase or containment.
        if p == h or p in h or h in p:
            return "entailment"

        # Antonym-based contradiction is checked BEFORE the overlap-based
        # entailment shortcut. A morphological antonym ("constitutional" vs
        # "unconstitutional") carries no negation word, so the two sides look
        # like a same-polarity near-paraphrase and the shortcut would otherwise
        # return `entailment` for a flat contradiction.
        if self._antonym_contradiction(p, h):
            return "contradiction"

        # Entailment: high content overlap at the same polarity.
        if ratio >= self._MIN_OVERLAP and p_neg == h_neg:
            return "entailment"

        # Negation-flip contradiction: one text is the other with a negation removed.
        negated_h = re.sub(r"\b(not |no |never |fails to |does not |is not |are not )", " ", h)
        negated_p = re.sub(r"\b(not |no |never |fails to |does not |is not |are not )", " ", p)
        if p == negated_h.strip() or h == negated_p.strip():
            return "contradiction"

        # One-sided negation. This is a contradiction only when the two
        # propositions are actually about the same thing (content-word overlap)
        # AND the negation reaches a shared term. Without both guards the rule
        # fires on essentially any two English sentences: `_normalize` does not
        # drop stopwords, so raw `.split()` overlap is never empty.
        if ratio >= self._MIN_OVERLAP and p_neg != h_neg:
            negated_side = premise if p_neg else hypothesis
            if overlap & self._negation_scope_terms(negated_side):
                return "contradiction"

        return "neutral"
