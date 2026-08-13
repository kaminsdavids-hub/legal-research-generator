"""Shared text normalization for the NLI pass and the independence guard.

Both need the same notions of "what words does this proposition actually carry"
and "is this proposition negated". Keeping one definition means a change to
stopwords or negation markers cannot make the two disagree — which matters,
because the independence guard exists to catch exactly the polarity-flip pattern
the NLI pass reports as a contradiction.
"""

from __future__ import annotations

import re

#: Tokens that carry no topical content. Negation words are included so a bare
#: "not" never counts as shared subject matter; polarity is tracked separately
#: by :func:`has_negation`.
STOP = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "that", "this",
    "is", "are", "be", "as", "by", "with", "any", "such", "under", "section",
    "it", "its", "from", "at", "into", "over", "after", "before", "when", "where",
    "was", "were", "been", "has", "have", "had", "does", "did", "do", "will",
    "there", "here", "but", "than", "then", "so", "if", "not",
}

#: Negation markers. Always applied to raw (lower-cased) text, never to
#: normalized text: normalization strips apostrophes, which would make every
#: contraction alternative unreachable and read "doesn't" as non-negated.
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|nor|neither|without|lacks?|fails?|failed|"
    r"absent|denies|denied)\b|\b\w+n't\b"
)


def normalize(text: str) -> str:
    """Lower-case, punctuation-stripped, space-joined tokens."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def raw_tokens(text: str) -> list[str]:
    """Lower-case tokens with apostrophes preserved, so contractions survive."""
    return re.findall(r"[a-z0-9']+", text.lower())


def has_negation(text: str) -> bool:
    """True when *text* contains a negation marker. Pass RAW text, not normalized."""
    return bool(NEGATION_RE.search(text.lower()))


def stem(word: str) -> str:
    """Crude suffix stripping, enough to align "applies" with "apply".

    Not linguistics — just enough that a proposition and its polarity-flipped
    twin do not look like different vocabulary because one inflects a verb.
    """
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("es"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    return word


def content_words(text: str, *, stemmed: bool = False) -> set[str]:
    """Topical words: stopwords, negation markers and very short tokens dropped."""
    words = {w for w in normalize(text).split() if len(w) > 2 and w not in STOP}
    if stemmed:
        return {stem(w) for w in words}
    return words


def overlap_ratio(a: set[str], b: set[str]) -> float:
    """Share of the *smaller* set that is shared. Asymmetric by design."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    """Symmetric overlap. Use when neither side should be privileged."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(inner: set[str], outer: set[str]) -> float:
    """Share of *inner* contained in *outer*. Catches truncated restatements."""
    if not inner:
        return 0.0
    return len(inner & outer) / len(inner)
