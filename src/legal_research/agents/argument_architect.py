"""Argument Architect (spec §2.5).

Structures the paper: thesis, roadmap, sections (IRAC/CREAC where appropriate),
counterargument + rebuttal. Builds the outline from the ideas the scholar selected;
every section is anchored to at least one kept idea.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..models import SectionStatus
from .base import Agent, AgentContext, AgentResult

# A stable law-review scaffold. Section *titles* are derived from the selected ideas
# so no two papers share identical boilerplate (see the anti-template check).
_FIXED_FRONT = ["Introduction"]
_FIXED_BACK = ["Counterargument and Rebuttal", "Conclusion"]

# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #
# An idea is a sentence; a heading is a noun phrase. Truncating the sentence at
# 70 characters produced neither, and shipped three distinct defects into live
# manuscripts: "I. To avoid regulatory entanglements with U.S. tech exports
# laws like...", "II. **TOPIC: AI Algorithmic Protection**", and — the one that
# matters — "II. Publishing AI model weights may be subject to EAR
# § 3599.7(b)(4)...", citing a provision that does not exist.
#
# That last one is why this is not a cosmetic fix. **Nothing verifies a
# heading.** The CitationGuard grounds propositions in body prose and the
# Verifier re-checks them; a title is never grounded, never verified, and never
# appears in the verification report, so an invented authority in a heading
# reaches the PDF and the table of contents unchallenged. Headings therefore
# carry no citations at all: if an authority is load-bearing, it belongs in the
# text, where the guard can see it.

_MARKDOWN = re.compile(r"[*_`#]+")
#: "Topic:", "Section 3 -", "Issue:" and friends, which models emit as scaffolding.
_LEADING_LABEL = re.compile(
    r"^\s*(?:topic|section|part|issue|point|argument)\b\s*\d*\s*[:.\-–]\s*", re.IGNORECASE
)
#: A roman or arabic enumeration the model added; this function adds its own.
_ENUMERATION = re.compile(r"^\s*(?:[ivxlcIVXLC]+|\d+|[A-Za-z])\s*[.)]\s+")
#: Citation shapes. Deliberately narrow: bare "U.S." must survive, because
#: "U.S. export law" is a phrase and "550 U.S. 544" is a citation.
_CITATION = re.compile(
    r"(?:\b\d+\s+[A-Z][\w.]*\.?\s*(?:\d+[a-z]{0,2}\s+)?\d+\b)"  # 550 U.S. 544
    r"|(?:\bU\.?\s?S\.?\s?C\.?\b[^,;]*)"                        # 17 U.S.C. § 106
    r"|(?:\bC\.?\s?F\.?\s?R\.?\b[^,;]*)"
    r"|(?:§+\s*[\w.()–-]+)",                                    # § 3599.7(b)(4)
)
#: A case name, checked against the corpus before it is allowed to stay.
_CASE_NAME = re.compile(r"\b[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)*\s+v\.\s+[A-Z][\w.'-]+")
#: Where a sentence stops being a noun phrase. Cutting at the first of these
#: leaves the subject, which is what a heading is.
# "and"/"or" are deliberately absent: they join noun phrases at least as often
# as they join clauses, and breaking on them turned "the strongest textual and
# precedential objection" into "The strongest textual".
_CLAUSE_BREAK = re.compile(
    r"[,;:—]|\s+(?:that|which|because|when|if|unless|so|but|may|must|should|"
    r"shall|will|would|can|could|does|do|did|is|are|was|were|has|have|had)\s+",
    re.IGNORECASE,
)
#: Words a heading must not end on.
_TRAILING_STOPWORDS = frozenset(
    {"of", "the", "a", "an", "and", "or", "in", "on", "to", "for", "with", "under",
     "by", "from", "as", "at", "into", "over", "such", "this", "that", "these",
     "whether", "about", "between", "against", "than",
     # "re" is here so that the stub left by an unverifiable "In re X v. Y"
     # unwinds to nothing and takes the fallback, instead of shipping "In re".
     "re"}
)

MIN_HEADING_WORDS = 3
MAX_HEADING_WORDS = 9
#: Used when nothing survives. Honest rather than invented: the section is an
#: analysis of an idea whose text could not be reduced to a heading, and a
#: fabricated topic would be worse than a plain label.
FALLBACK_HEADING = "Analysis"


def heading_from(idea_text: str, known: Callable[[str], bool] | None = None) -> str:
    """Reduce an idea sentence to a section heading.

    Deterministic on purpose. Asking a model for a heading is what produced
    ``EAR § 3599.7(b)(4)`` in the first place, and a heading is precisely the
    place where no later stage would catch it.

    ``known`` decides whether a case name may stay. Stripping every citation
    shape unconditionally was the first version, and it turned "Junger v. Daley
    controls because source code is expressive" into "Controls because source
    code" — mangling a heading that names an authority the corpus actually
    holds, which is ordinary law-review practice. So a case name survives when
    the corpus knows it and is removed when it does not, which is the same rule
    the CitationGuard applies to body prose, enforced in the one place that has
    no guard. Section and reporter citations are always removed: their
    precision is exactly what a heading cannot carry and a reader cannot check.
    """

    text = _MARKDOWN.sub(" ", idea_text or "")
    text = _drop_unknown_cases(text, known)
    text = _CITATION.sub(" ", text)
    text = _LEADING_LABEL.sub("", text.strip())
    text = _ENUMERATION.sub("", text)
    text = _LEADING_LABEL.sub("", text.strip())  # "I. Topic: ..." needs both
    text = " ".join(text.split()).strip(" .,:;-–")
    if not text:
        return FALLBACK_HEADING

    core = _cut_to_noun_phrase(text)
    words = core.split()
    while words and words[-1].lower().strip(".,;:") in _TRAILING_STOPWORDS:
        words.pop()
    if len(words) < MIN_HEADING_WORDS:
        # The cut was too aggressive for this sentence; fall back to the word cap
        # rather than shipping a one-word heading.
        words = text.split()[:MAX_HEADING_WORDS]
        while words and words[-1].lower().strip(".,;:") in _TRAILING_STOPWORDS:
            words.pop()
    if not words:
        return FALLBACK_HEADING

    heading = " ".join(words).strip(" .,:;-–")
    # Sentence case, not title case: title-casing would turn "AI" into "Ai" and
    # "U.S." into "U.s.", and lowercase-preserving variants still get "Of" wrong.
    return heading[0].upper() + heading[1:] if heading else FALLBACK_HEADING


def _drop_unknown_cases(text: str, known: Callable[[str], bool] | None) -> str:
    """Truncate at the first case name the corpus does not hold.

    Truncating rather than excising, because excising leaves debris that reads
    like a heading and is not one: cutting the unverifiable "In re Seagate Tech.
    v. Fabricated Corp." out of the middle produced "In re Corp. controls the
    analysis here". Everything before the unknown authority is prose that does
    not depend on it, so it is kept; everything after was written about it.
    """

    for match in _CASE_NAME.finditer(text):
        if known and known(match.group(0)):
            continue
        return text[: match.start()]
    return text


def noun_phrase(text: str, max_words: int = MAX_HEADING_WORDS) -> str:
    """The subject, up to the first clause break that leaves enough words.

    Public because the pipeline's fallback ideas need it too: they interpolate
    the scholar's raw thesis into a template, and a thesis is a sentence, so
    "The controlling doctrinal test for" + "Publishing open model weights is
    protected expression, and the Export Administration Regulations may not
    treat that publication as a deemed export" produced an idea that is not
    grammatical, a heading built from it, and a retrieval proposition that asks
    the corpus about two claims joined by a comma.
    """

    for match in _CLAUSE_BREAK.finditer(text):
        candidate = text[: match.start()].strip()
        if len(candidate.split()) >= MIN_HEADING_WORDS:
            return " ".join(candidate.split()[:max_words])
    return " ".join(text.split()[:max_words])


_cut_to_noun_phrase = noun_phrase


class ArgumentArchitect(Agent):
    name = "Argument Architect"
    expert_role = "writer"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        bb.outline.clear()
        selected = bb.selected_ideas()

        intro = bb.add_section(_FIXED_FRONT[0])
        intro.idea_ids = [i.id for i in selected]

        if not selected:
            # Minimal viable outline even with no selected ideas.
            bb.add_section("Background and Governing Standard")
        known = _corpus_knows(ctx)
        for n, idea in enumerate(selected, start=1):
            title = self._section_title(n, idea.text, known)
            bb.add_section(title, idea_ids=[idea.id])

        for title in _FIXED_BACK:
            bb.add_section(title)

        for section in bb.outline:
            section.status = SectionStatus.IDEA

        return AgentResult(
            agent=self.name,
            summary=f"built outline with {len(bb.outline)} sections from {len(selected)} ideas",
            payload={"section_ids": [s.id for s in bb.outline]},
        )

    @staticmethod
    def _section_title(
        n: int, idea_text: str, known: Callable[[str], bool] | None = None
    ) -> str:
        return f"{_roman(n)}. {heading_from(idea_text, known)}"


def _corpus_knows(ctx: AgentContext) -> Callable[[str], bool]:
    """Does the corpus hold a record whose title contains this case name?

    Matching on the short form the heading uses ("Junger v. Daley") against the
    record's full title, both normalised, because a corpus title carries the
    reporter and year a heading never would.
    """

    titles = [" ".join(r.title.lower().split()) for r in getattr(ctx.corpus, "records", [])]

    def known(case_name: str) -> bool:
        # Surnames, not the whole string: a heading says "Junger v. Daley" and
        # the corpus record is titled "Peter D. Junger v. William Daley, United
        # States Secretary of Commerce", so containment finds nothing. Both
        # surnames, in order, is the match a reader would make.
        parts = " ".join(case_name.lower().split()).rstrip(".,;:").split(" v. ")
        if len(parts) != 2 or not parts[0].split() or not parts[1].split():
            return False
        left = parts[0].split()[-1].strip(".,;:")
        right = parts[1].split()[0].strip(".,;:")
        if len(left) < 3 or len(right) < 3:
            return False
        return any(
            left in title and right in title and title.index(left) < title.index(right)
            for title in titles
        )

    return known


def _roman(n: int) -> str:
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out
