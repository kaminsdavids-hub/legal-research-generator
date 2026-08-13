"""Learning which questions are worth asking — from the author, not the gates.

The question policy ships with a hand-ordered `PRIORITY`. This adapts it to the
person actually using it, and the whole design turns on one choice: **what counts
as evidence that a question was worth asking.**

**The gates are not the teacher.** Ranking questions by whether their answers
passed the gates would optimise for questions whose answers the machinery
happens to like, and the machinery would then be grading its own curriculum. That
is the collapse §18 found in retrieval, one layer up, and it is more tempting
here because the gate verdicts are right there in `StepResult`. They are
deliberately not read.

**A refused answer is a successful question.** If the author wrote a paragraph
and the grounding gate rejected the patch, the question did its job — it
provoked work. The refusal is a fact about the machinery, not about the question.
Only *declining* is negative evidence, because only the author can decide a
question was not worth their time.

So the signal is the author's revealed preference: did they answer, did they
decline, and how much did they write. Nothing else.

**The journal outlives the manuscript.** What is being learned is a property of
the *author* — which questions they find worth their time — not of one paper. It
is stored separately from the session for that reason, and because it has to be:
a single offline manuscript runs dry after two or three questions, which is below
the evidence floor for any one gap kind, so a per-manuscript journal could never
accumulate enough to act on. It would have been a feature that shipped and never
once activated.

**Evidence nudges; it does not overrule.** An adjustment is bounded at
``MAX_SHIFT`` positions, so the declared order dominates over any distance
greater than that: a `DEPENDS_ON` cycle the author keeps skipping still outranks
an unverified citation they enjoy fixing, because five positions is further than
evidence can travel. *Adjacent* kinds can and do swap — an engaged-with
unanswered attack will overtake a self-grounding cycle nobody has been asked
about yet — and that is intended rather than tolerated: neighbouring priorities
were close to a judgement call in the first place, and the author's behaviour is
better evidence than the ordering guess. What evidence cannot do is rewrite the
order wholesale.

No gap kind is ever suppressed outright either: a kind that stopped being asked
could never earn its way back, and the loop would silently narrow to whatever the
author answered first.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .socratic import PRIORITY, Gap, GapKind

#: Widest a learned adjustment may move a gap kind, in priority positions. Small
#: on purpose: this tunes an order that was reasoned about, it does not replace
#: the reasoning.
MAX_SHIFT = 1.5

#: Episodes of a kind before its evidence is used at all. One answer is a mood.
MIN_EVIDENCE = 3

#: An answer shorter than this is engagement without substance — the author
#: typing something to move on. Counted as answered, but not as a good answer.
TERSE_WORDS = 8


class Outcome(StrEnum):
    #: The author wrote an answer. Whether it merged is not this module's
    #: business — see the module docstring.
    ANSWERED = "answered"
    #: The author declined the question. The only negative signal there is.
    DECLINED = "declined"


@dataclass(frozen=True)
class Episode:
    gap_kind: GapKind
    outcome: Outcome
    section: str = ""
    answer_words: int = 0

    @property
    def substantive(self) -> bool:
        return self.outcome is Outcome.ANSWERED and self.answer_words >= TERSE_WORDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_kind": self.gap_kind.value,
            "outcome": self.outcome.value,
            "section": self.section,
            "answer_words": self.answer_words,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        return cls(
            gap_kind=GapKind(data["gap_kind"]),
            outcome=Outcome(data["outcome"]),
            section=data.get("section", ""),
            answer_words=int(data.get("answer_words", 0)),
        )


@dataclass
class Journal:
    """What the author did with the questions they were asked."""

    episodes: list[Episode] = field(default_factory=list)

    def record(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def of_kind(self, kind: GapKind) -> list[Episode]:
        return [e for e in self.episodes if e.gap_kind is kind]

    def engagement(self, kind: GapKind) -> float | None:
        """Share of questions of this kind that drew a substantive answer.

        ``None`` when there is not enough evidence to say — which is different
        from zero engagement, and is the distinction §11.8 was about. A kind
        nobody has been asked about is unmeasured, not unwanted.
        """
        episodes = self.of_kind(kind)
        if len(episodes) < MIN_EVIDENCE:
            return None
        return sum(1 for e in episodes if e.substantive) / len(episodes)

    def summary(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for kind in GapKind:
            episodes = self.of_kind(kind)
            if not episodes:
                continue
            rows[kind.value] = {
                "asked": len(episodes),
                "answered": sum(1 for e in episodes if e.outcome is Outcome.ANSWERED),
                "substantive": sum(1 for e in episodes if e.substantive),
                "engagement": self.engagement(kind),
            }
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"episodes": [e.to_dict() for e in self.episodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Journal:
        return cls(episodes=[Episode.from_dict(e) for e in data.get("episodes", [])])

    # ------------------------------------------------------------------ #
    # Persistence, deliberately separate from the manuscript
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """Write-then-rename, so an interrupted save cannot truncate a history."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Journal:
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class LearnedPolicy:
    """Ranks gap kinds: the declared priority, nudged by observed engagement."""

    def __init__(self, journal: Journal | None = None, *, max_shift: float = MAX_SHIFT):
        self.journal = journal or Journal()
        self.max_shift = max_shift

    def rank(self, kind: GapKind) -> float:
        """Lower sorts first, matching `PRIORITY.index`.

        The base is the declared position. Engagement moves it by at most
        ``max_shift``, centred so that average engagement is no adjustment at
        all: a kind the author engages with rises, one they answer tersely or
        skip falls, and neither can travel far enough to reorder the reasoning
        that set the base.
        """
        base = float(PRIORITY.index(kind))
        engagement = self.journal.engagement(kind)
        if engagement is None:
            return base
        # engagement 1.0 -> -max_shift (earlier), 0.0 -> +max_shift (later).
        return base + self.max_shift * (1.0 - 2.0 * engagement)

    def order(self, kinds: list[GapKind]) -> list[GapKind]:
        return sorted(kinds, key=lambda k: (self.rank(k), PRIORITY.index(k)))

    def explain(self) -> list[str]:
        """Why the order is what it is. A policy that cannot say why it moved is
        one nobody can argue with."""
        lines = []
        kinds: list[GapKind] = list(GapKind)
        for kind in sorted(kinds, key=self.rank):
            engagement = self.journal.engagement(kind)
            base = PRIORITY.index(kind)
            if engagement is None:
                lines.append(
                    f"{kind.value:22s} base {base}  (not enough evidence yet)"
                )
            else:
                lines.append(
                    f"{kind.value:22s} base {base} -> {self.rank(kind):.2f}  "
                    f"({engagement:.0%} substantive over "
                    f"{len(self.journal.of_kind(kind))} asked)"
                )
        return lines


def episodes_by_section(journal: Journal) -> dict[str, int]:
    """Substantive answers per section — where this author's work actually is."""
    counts: dict[str, int] = defaultdict(int)
    for episode in journal.episodes:
        if episode.substantive and episode.section:
            counts[episode.section] += 1
    return dict(counts)


def gap_episode(gap: Gap, outcome: Outcome, answer: str = "") -> Episode:
    return Episode(
        gap_kind=gap.kind,
        outcome=outcome,
        section=gap.section,
        answer_words=len(answer.split()),
    )
