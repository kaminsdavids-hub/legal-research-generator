"""Anti-"AI voice" lint.

Flags the formulaic tells of machine prose: over-hedging, clustered transition words
(``Moreover``/``Furthermore`` ...), empty summarizing sentences, and uniform sentence
length. Returns a structured report; the pipeline blocks prose that fails.
"""

from __future__ import annotations

import re
import statistics
from enum import Enum

from pydantic import BaseModel, Field

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")

HEDGES = [
    "arguably", "perhaps", "it seems", "it appears", "may suggest", "might suggest",
    "could be argued", "somewhat", "relatively", "to some extent", "in a sense",
    "it could be said", "one might argue",
]
TRANSITIONS = ["moreover", "furthermore", "additionally", "in addition", "notably", "importantly"]
EMPTY_SUMMARY = [
    "in conclusion", "in summary", "to summarize", "it is important to note that",
    "it is worth noting that", "at the end of the day", "needless to say",
    "as we can see", "it goes without saying",
]


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


class Finding(BaseModel):
    kind: str
    message: str
    severity: Severity
    evidence: str = ""


class AiVoiceReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    score: float = 1.0  # 1.0 = most human-like
    passed: bool = True

    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAIL]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _count_phrases(text_lower: str, phrases: list[str]) -> list[str]:
    hits: list[str] = []
    for p in phrases:
        if p in text_lower:
            hits.append(p)
    return hits


def lint_ai_voice(text: str) -> AiVoiceReport:
    report = AiVoiceReport()
    sentences = _sentences(text)
    words = _WORD.findall(text)
    n_words = max(1, len(words))
    lower = text.lower()

    # 1) Over-hedging.
    hedge_hits = [h for h in HEDGES if h in lower]
    hedge_density = len(re.findall("|".join(re.escape(h) for h in HEDGES), lower)) / n_words * 100
    if hedge_density > 1.5:
        report.findings.append(
            Finding(
                kind="over_hedging",
                message=f"hedging density {hedge_density:.1f}/100 words is too high",
                severity=Severity.FAIL,
                evidence=", ".join(sorted(set(hedge_hits))),
            )
        )

    # 2) Clustered transitions (three+ sentences opening with a transition word, or
    #    two consecutive such openers).
    openers = [s.split()[0].lower().strip(",") if s.split() else "" for s in sentences]
    transition_openers = [i for i, o in enumerate(openers) if o in TRANSITIONS]
    consecutive = any(
        b - a == 1 for a, b in zip(transition_openers, transition_openers[1:], strict=False)
    )
    if len(transition_openers) >= 3 or consecutive:
        report.findings.append(
            Finding(
                kind="transition_clustering",
                message="transition words (Moreover/Furthermore/...) cluster unnaturally",
                severity=Severity.FAIL,
                evidence=f"{len(transition_openers)} transition-opening sentences",
            )
        )

    # 3) Empty summarizing sentences.
    empty_hits = _count_phrases(lower, EMPTY_SUMMARY)
    if empty_hits:
        report.findings.append(
            Finding(
                kind="empty_summary",
                message="contains empty summarizing filler",
                severity=Severity.FAIL,
                evidence=", ".join(empty_hits),
            )
        )

    # 4) Uniform sentence length (low variance is a strong AI tell).
    if len(sentences) >= 4:
        lengths = [len(_WORD.findall(s)) for s in sentences]
        stdev = statistics.pstdev(lengths)
        if stdev < 3.0:
            report.findings.append(
                Finding(
                    kind="uniform_cadence",
                    message=f"sentence-length variety is too low (stdev={stdev:.1f})",
                    severity=Severity.FAIL,
                    evidence=f"lengths={lengths}",
                )
            )

    n_fail = len(report.failures())
    report.passed = n_fail == 0
    report.score = round(max(0.0, 1.0 - 0.25 * n_fail), 3)
    return report


def rewrite_for_voice(text: str, style_sample: str | None = None) -> str:
    """Rewrite flagged prose toward a more human cadence.

    In ``openai`` mode the Writer engine rewrites seeded by ``style_sample``. In mock
    mode we apply deterministic structural fixes so the hook is testable and actually
    clears the lint: strip clustered transitions, drop empty-summary filler, and, when
    sentence lengths are too uniform, collapse the middle sentences into one longer
    clause so the paragraph reads as short / long / short rather than a metronome.
    """

    out = text
    for t in TRANSITIONS:
        out = re.sub(rf"(?i)\b{re.escape(t)}\b,\s*", "", out)
    for phrase in EMPTY_SUMMARY:
        out = re.sub(rf"(?i){re.escape(phrase)},?\s*", "", out)

    if any(f.kind == "uniform_cadence" for f in lint_ai_voice(out).failures()):
        sents = _sentences(out)
        if len(sents) >= 4:
            head, tail = sents[0], sents[-1]
            middle = "; ".join(s.rstrip(".!?") for s in sents[1:-1])
            out = " ".join([head, f"{middle}." if middle else "", tail]).strip()

    return re.sub(r"\s{2,}", " ", out).strip()
