"""Citation placeholders, and how prose may be rewritten around them.

Drafted prose carries ``{{cite:<id>}}`` placeholders that the formatter later
turns into footnotes. Every pass that rewrites prose — the anti-AI-voice
rewrite, the grammar chain, the mechanism gate — has the same obligation: the
set of placeholders must come out of the pass exactly as it went in. A pass that
drops one silently removes an authority from the paper; a pass that invents one
attaches a footnote to text no retriever ever grounded.

These helpers exist so that obligation is implemented once. They were private to
:mod:`legal_research.agents.editor` until the mechanism gate needed the same
guarantee, and a second copy of a rule this load-bearing is how the two copies
start to disagree.
"""

from __future__ import annotations

import re

CITE_TOKEN = re.compile(r"\{\{cite:[^}]+\}\}")

__all__ = [
    "CITE_TOKEN",
    "collapse_spacing",
    "collect_tokens",
    "join_paragraphs",
    "normalize_paragraph_count",
    "rebuild_with_tokens",
    "split_paragraphs",
    "split_with_tokens",
    "strip_tokens",
]


def collapse_spacing(text: str) -> str:
    """Tidy the spacing a rewrite or a removed citation leaves behind.

    Paragraph breaks survive. Two separate passes each ended with a blanket
    ``re.sub(r"\\s{2,}", " ", text)`` — the anti-AI-voice rewrite and the
    citation formatter — and because the formatter runs last, fixing only the
    first one changed nothing: three live manuscripts came out of the pipeline
    as one 400-to-550-word block per section. A collapse that treats the blank
    line between paragraphs as stray whitespace destroys the only structure the
    renderer, the Socratic coach and the mechanism gate have to work with, and
    nothing downstream can put it back.
    """

    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_tokens(text: str) -> str:
    return CITE_TOKEN.sub("", text).strip()


def collect_tokens(text: str) -> list[str]:
    return CITE_TOKEN.findall(text)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if p.strip()).strip()


def split_with_tokens(text: str) -> tuple[list[str], list[list[str]]]:
    """Split into paragraphs, returning prose and each paragraph's tokens."""

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    prose: list[str] = []
    tokens_by_paragraph: list[list[str]] = []
    for paragraph in paragraphs:
        tokens_by_paragraph.append(CITE_TOKEN.findall(paragraph))
        prose.append(CITE_TOKEN.sub("", paragraph).strip())
    return prose, tokens_by_paragraph


def rebuild_with_tokens(paragraphs: list[str], tokens_by_paragraph: list[list[str]]) -> str:
    """Reattach each paragraph's tokens to the rewritten prose, in order."""

    rebuilt: list[str] = []
    for idx, tokens in enumerate(tokens_by_paragraph):
        prose = paragraphs[idx].strip() if idx < len(paragraphs) else ""
        merged = (prose + "".join(tokens)).strip() if prose else "".join(tokens)
        if merged:
            rebuilt.append(merged)
    for extra in paragraphs[len(tokens_by_paragraph) :]:
        extra = extra.strip()
        if extra:
            rebuilt.append(extra)
    return join_paragraphs(rebuilt)


def normalize_paragraph_count(paragraphs: list[str], target: int) -> list[str]:
    """Pad or fold a rewrite so it has exactly ``target`` paragraphs.

    A rewrite that split or merged paragraphs would otherwise shift every
    subsequent paragraph's tokens onto the wrong prose.
    """

    if target <= 0:
        return []
    if len(paragraphs) == target:
        return paragraphs
    if len(paragraphs) < target:
        return [*paragraphs, *("" for _ in range(target - len(paragraphs)))]
    head = paragraphs[: target - 1]
    tail = " ".join(p.strip() for p in paragraphs[target - 1 :] if p.strip()).strip()
    return [*head, tail]
