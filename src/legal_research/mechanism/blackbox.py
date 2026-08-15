"""Deterministic black-box detection for legal prose. No LLM, no I/O, no config.

This module is the invariant. Everything else in :mod:`legal_research.mechanism`
is advisory relative to it: the LLM critic may add findings or escalate them, and
may never remove one.

The rule it enforces
--------------------
"AI" is a vocabulary of implementation, never a mechanism. A paragraph may
*discuss* a model; its argument may not *rest* on one. The operative question is
whether the sentence names what the system did, or only that something clever
happened and a legal consequence followed.

Why the rule is not stylistic
-----------------------------
Legal analysis is applied to an operation, and a passage that never names the
operation cannot be tested against any standard the paper invokes:

* **Disparate impact** — 42 U.S.C. § 2000e-2(k)(1)(B)(i) makes the plaintiff
  identify *each particular challenged practice*. "The algorithm discriminates"
  identifies none, so the argument built on it has no element to prove.
* **Expert methodology** — *Daubert v. Merrell Dow Pharmaceuticals*, 509 U.S. 579
  (1993), tests the method, not the conclusion. A method stated as "a model
  predicts recidivism" is not a method a court can examine.
* **Reasoned decisionmaking** — *Motor Vehicle Mfrs. Ass'n v. State Farm*,
  463 U.S. 29 (1983), requires the agency to articulate the basis of its
  decision. A brief arguing an agency failed that test must itself say what the
  decision procedure was.

So a term is a violation when it occupies the **mechanism position** and no
structure sits next to it. "A gradient-boosted classifier over 42 application
features, thresholded at 0.65" is structure. "An AI system" is not, and neither
is "an algorithm that intelligently prioritizes applicants".

Why this is not the patent detector
-----------------------------------
This module began as ``patentgen.blackbox``, which flags a banned term wherever
it appears, because a patent claim may not use one at all. That rule is wrong
here and would make the gate useless in exactly the papers it exists for: a
law-review article on the EU AI Act says "machine learning" on every page and
should. **Mention is not the defect; mechanism position is.** Every pattern
below therefore pairs the vocabulary with a *position* — agent of a
determinative verb, instrument of a "by/using" phrase, cause in an attribution,
or subject of a legal conclusion — and a bare topical mention never fires.

Design notes
------------
On ambiguity the detector flags. A false flag costs the author one rewrite; a
missed one ships a paragraph whose argument has no referent. Findings are
evidence for a human, not a verdict on the paper, and the module says so rather
than implying more precision than regexes can carry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Finding",
    "MECHANISM_PATTERNS",
    "ParagraphReport",
    "Severity",
    "has_structure_nearby",
    "scan",
    "scan_by_paragraph",
    "scan_text",
]


class Severity(str, Enum):
    """How a finding should be treated by the gate."""

    #: Vocabulary in mechanism position with no structure anywhere near it.
    BLACK_BOX = "BLACK_BOX"
    #: The same position, but *some* structure is adjacent and it is too thin.
    #: Kept separate from BLACK_BOX so the repair pass and the report can tell
    #: "said nothing" from "said too little" — different rewrites, different
    #: severities to a reader.
    UNDER_SPECIFIED = "UNDER_SPECIFIED"


@dataclass(frozen=True)
class Finding:
    """One located violation.

    ``span`` indexes into the text that was scanned, so a caller can underline
    the exact phrase rather than reprinting the sentence and hoping the reader
    spots it.
    """

    severity: Severity
    term: str
    span: tuple[int, int]
    excerpt: str
    reason: str
    where: str = ""

    def render(self) -> str:
        location = f"{self.where}: " if self.where else ""
        return f"[{self.severity.value}] {location}{self.reason} — {self.excerpt!r}"


# --------------------------------------------------------------------------- #
# The vocabulary, and the positions that make it a defect
# --------------------------------------------------------------------------- #
# A head noun phrase naming a system by its discipline rather than its
# operation. The optional determiner is part of the pattern so that "the model"
# matches and "the model of cooperative federalism" does not: the qualifier is
# what would have carried the structure, and an intervening "of ..." means the
# noun is not the head the verb attaches to.
_AI_HEAD = (
    r"(?:(?:the|an?|this|that|such|its|their|these|those)\s+)?"
    r"(?:"
    r"artificial[-\s]intelligence(?:\s+(?:system|tool|model|program|agent))?"
    r"|\bAI\b(?:[-\s](?:system|tool|model|engine|program|agent))?"
    r"|machine[-\s]learning(?:\s+(?:system|model|tool|algorithm|classifier))?"
    r"|deep[-\s]learning(?:\s+(?:system|model))?"
    r"|neural\s+networks?"
    r"|algorithms?"
    r"|models?"
    r"|classifiers?"
    r"|automated\s+(?:system|tool|process)"
    r"|automated\s+decision[-\s]?making(?:\s+(?:system|tool|process))?"
    r")"
)

# Verbs that make the head the *decider*. Each names an outcome and no
# operation, which is precisely the substitution the rule is about.
_DETERMINATIVE_VERBS = (
    r"(?:determines?|decides?|predicts?|identifies|identif(?:y|ied)|flags?|flagged"
    r"|scores?|scored|ranks?|ranked|selects?|selected|denies|denied|approves?|approved"
    r"|rejects?|rejected|detects?|detected|infers?|inferred|classifies|classified"
    r"|recommends?|recommended|assesses|assessed|evaluates?|evaluated|calculates?"
    r"|learns?|learned|understands?|knows?|chooses?|assigns?|assigned|allocates?"
    r"|screens?|screened|matches|matched|generates?|generated|discriminates?"
    r"|discriminated)"
)

#: Pattern -> why that position is the defect. Order is documentation, not
#: precedence: every pattern is applied to every scan.
MECHANISM_PATTERNS: dict[str, str] = {
    # 1. Agent position: the system is the subject of a determinative verb.
    rf"{_AI_HEAD}\s+(?:\w+ly\s+)?{_DETERMINATIVE_VERBS}\b":
        "names the decider, not the decision rule — state the inputs, the "
        "operation performed on them, and the output the argument relies on",
    # 2. Instrument position: the system is how the act was accomplished.
    rf"\b(?:by|through|via|using|by\s+means\s+of|on\s+the\s+basis\s+of)\s+{_AI_HEAD}\b":
        "names an instrument, not the operation — 'by an algorithm' identifies "
        "no practice a court could examine",
    # 3. Cause position: an outcome is attributed to the system.
    rf"\b(?:because\s+of|due\s+to|attributable\s+to|caused\s+by|the\s+result\s+of)"
    rf"\s+{_AI_HEAD}\b":
        "attributes an outcome to a system whose operation the passage never "
        "states, so the causal claim cannot be tested",
    # 4. Legal conclusion about an unstated operation. This is the form that
    #    does the most damage in a law-review draft, because it reads as an
    #    argument and carries none: the element it would have to prove is
    #    exactly the operation it omits.
    rf"{_AI_HEAD}\s+(?:is|was|are|were)\s+"
    r"(?:biased|discriminatory|unlawful|arbitrary|unfair|unreliable)\b":
        "a legal conclusion about a system whose operation is unstated — name "
        "the practice before applying the standard to it",
    # 5. Capability claims that stand in for a step.
    r"\bintelligently\s+\w+":
        "'intelligently' names a result, not a step",
    r"\bautomatically\s+(?:infers?|determines?|learns?|discovers?|understands?|decides?)\b":
        "'automatically' hides the operation it replaces",
    # Adjectival forms fire only on an operative act, so that "AI-driven
    # regulation" as a subject of discussion does not, and "the AI-driven
    # denial" — where the denial is the thing being analysed — does.
    r"\b(?:AI|ML|algorithmically|machine[-\s]learning)[-\s]"
    r"(?:powered|driven|based|enabled|assisted)\s+"
    r"(?:determination|decision|assessment|screening|selection|denial|approval"
    r"|adjudication|sentencing|scoring)s?\b":
        "the modifier names a discipline, not the operation that produced the "
        "act being analysed",
    r"\bsmart\s+(?:system|engine|module|component)\b":
        "'smart' is marketing, not an operation",
    r"\bcognitive\s+(?:engine|module|system)\b":
        "'cognitive' names no computation",
}

_COMPILED: dict[str, re.Pattern[str]] = {
    pattern: re.compile(pattern, re.IGNORECASE) for pattern in MECHANISM_PATTERNS
}

# Concrete structure. Any of these near a hit downgrades BLACK_BOX to
# UNDER_SPECIFIED — still a finding, but a different rewrite.
#
# Note what is *not* here: a citation. A footnote to the study that describes
# the system tells the reader where the operation is written down; it does not
# put the operation in the sentence the argument is made in, and it is the
# sentence that has to survive the standard being applied.
_STRUCTURE_MARKERS: tuple[re.Pattern[str], ...] = (
    # Named architectures and estimators.
    re.compile(
        r"\b(?:transformer|encoder|decoder|convolution\w*|recurrent|LSTM|GRU|ResNet"
        r"|attention\s+head|gating\s+head|softmax|MLP|random\s+forest"
        r"|gradient[-\s]boost\w*|logistic\s+regression|linear\s+regression|SVM"
        r"|support\s+vector|k-?means|nearest\s+neighbou?rs?|decision\s+tree"
        r"|Kalman\s+filter|regression\s+discontinuity)\b",
        re.IGNORECASE,
    ),
    # Objective, training data, and provenance of the learned behaviour.
    re.compile(
        r"\b(?:trained\s+on|training\s+(?:data|set|corpus)|fine[-\s]tuned"
        r"|objective\s+function|loss\s+function|cross[-\s]entropy|ground\s+truth"
        r"|labell?ed\s+(?:data|examples)|validation\s+set|held[-\s]out)\b",
        re.IGNORECASE,
    ),
    # The decision rule itself: inputs, cut-off, output.
    re.compile(
        r"\b(?:input\s+(?:variables?|features?)|feature\s+(?:set|vector|s)?"
        r"|predictor\s+variables?|covariates?|decision\s+rule|cut-?off"
        r"|threshold|risk\s+score|probability\s+of|weights?\s+assigned"
        r"|ranked\s+list|outputs?\s+a)\b",
        re.IGNORECASE,
    ),
    # Quantified structure. A number attached to the mechanism is the cheapest
    # evidence that the author looked at one.
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|features?|variables?|layers?|parameters?"
        r"|inputs?|categories|applicants?|records?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:threshold|cut-?off|score|accuracy|precision|recall|AUC)\b"
        r"\s*(?:of|=|:|above|below)?\s*[\d.]+",
        re.IGNORECASE,
    ),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Character bounds of each sentence, in document order."""

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for piece in _SENTENCE_SPLIT.split(text):
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:  # pragma: no cover - split pieces always occur in order
            continue
        bounds.append((start, start + len(piece)))
        cursor = start + len(piece)
    return bounds or [(0, len(text))]


def has_structure_nearby(text: str, span: tuple[int, int]) -> bool:
    """Whether concrete structure appears in the hit's sentence or the next one.

    Adjacency is measured in sentences rather than characters because prose has
    sentences and claim limitations do not; the patent version's 240-character
    window would reach backwards into an unrelated argument as readily as
    forwards into the explanation.

    The *next* sentence counts deliberately. Authors write "The model determines
    eligibility. It is a logistic regression over six application variables,
    thresholded at 0.4." Demanding both in one sentence would flag prose that
    does exactly what the rule asks, and the author's only fix would be to write
    a worse sentence.
    """

    bounds = _sentence_bounds(text)
    for index, (start, end) in enumerate(bounds):
        if start <= span[0] < end:
            window_end = bounds[index + 1][1] if index + 1 < len(bounds) else end
            neighbourhood = text[start:window_end]
            return any(marker.search(neighbourhood) for marker in _STRUCTURE_MARKERS)
    return any(marker.search(text) for marker in _STRUCTURE_MARKERS)


def scan_text(text: str, *, where: str = "") -> list[Finding]:
    """Scan one blob and return every violation, in document order.

    Overlapping hits are collapsed to the longest one: "using an AI system to
    score applicants" matches both the instrument and the agent pattern, and
    reporting it twice would make the report look like two defects.
    """

    findings: list[Finding] = []
    for pattern, reason in MECHANISM_PATTERNS.items():
        for match in _COMPILED[pattern].finditer(text):
            span = (match.start(), match.end())
            severity = (
                Severity.UNDER_SPECIFIED
                if has_structure_nearby(text, span)
                else Severity.BLACK_BOX
            )
            findings.append(
                Finding(
                    severity=severity,
                    term=match.group(0).strip(),
                    span=span,
                    excerpt=_excerpt(text, span),
                    reason=reason,
                    where=where,
                )
            )
    return _dedupe_overlaps(findings)


def _dedupe_overlaps(findings: list[Finding]) -> list[Finding]:
    ordered = sorted(findings, key=lambda f: (f.span[0], -(f.span[1] - f.span[0])))
    kept: list[Finding] = []
    for finding in ordered:
        if any(
            finding.span[0] < prior.span[1] and prior.span[0] < finding.span[1]
            for prior in kept
        ):
            continue
        kept.append(finding)
    return kept


def scan(sections: dict[str, str]) -> list[Finding]:
    """Scan a mapping of ``label -> text``. Labels appear in ``Finding.where``."""

    findings: list[Finding] = []
    for label, text in sections.items():
        findings.extend(scan_text(text, where=label))
    return findings


def _excerpt(text: str, span: tuple[int, int], *, pad: int = 70) -> str:
    start = max(0, span[0] - pad)
    end = min(len(text), span[1] + pad)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{' '.join(text[start:end].split())}{suffix}"


@dataclass
class ParagraphReport:
    """Per-paragraph detail, so a caller can point at the prose to rewrite."""

    index: int
    paragraph: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings


def scan_by_paragraph(text: str, *, where: str = "") -> list[ParagraphReport]:
    """Split into paragraphs first, then scan each independently.

    The repair pass rewrites one paragraph at a time, so it needs findings
    indexed to a paragraph rather than to the section they were found in.
    Scanning per paragraph also stops structure in a later paragraph from
    rescuing an earlier bare assertion, which the sentence window would allow
    across a paragraph break.
    """

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return [
        ParagraphReport(
            index=index,
            paragraph=paragraph,
            findings=scan_text(paragraph, where=where),
        )
        # Indexed over non-empty paragraphs so the index matches the section
        # splitting the writer and editor already use; a blank line must not
        # shift the paragraph the repair pass rewrites.
        for index, paragraph in enumerate(paragraphs)
    ]
