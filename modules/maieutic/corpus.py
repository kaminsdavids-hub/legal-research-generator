"""Keeping the author's own exchanges, so a model can later be trained on them.

The proposal this implements is one line long: fine-tune the dialectic model on
accumulated ``(question → author's answer → synthesis)`` triples, so the system
learns *this* author's reasoning rather than the average of everyone's. The
work is in what that sentence leaves out.

**It is off unless the author turns it on.** Everything else in this package
writes about the author — how many questions they answered, how long the answers
were. This writes *what they wrote*, verbatim, into a second file that outlives
the manuscript and exists to be fed to a training run. That is a different
category of thing to keep, so it takes an explicit
``LRG_MAIEUTIC_TRAINING_CAPTURE=1`` and a path; there is no default location, and
:func:`from_env` returning ``None`` is the normal case.

**The gate verdict is recorded here, and it is not recorded in the journal.**
:mod:`~modules.maieutic.learn` refuses to read gate verdicts because ranking
*questions* by whether their answers passed would let the machinery grade its own
curriculum. That argument does not transfer: this corpus is about the *text*, and
whether a paragraph survived grounding and citation checks is real evidence about
the text. So ``merged`` is stored — and stored rather than filtered on, because
which subset to train on is a decision for whoever runs the training, made in the
open, not one this module makes silently at capture time.

**The synthesis is stored as what it is.** The machine's synthesis is never
merged into the manuscript (see :mod:`~modules.maieutic.dialectic_adapter`), so
no synthesis here was ever "accepted" in the sense of becoming part of the
paper. ``merged`` says only that the exchange it belonged to produced a patch
that survived every gate. Calling that an accepted synthesis in the file format
would encode a claim the loop never made, and a training run reading this file a
year from now has no way to check it.

**One exchange, one record.** Re-answering the same question replaces nothing —
both attempts are kept, because the second is evidence about the first. Exact
duplicates are dropped, since replaying a session should not multiply its weight.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Triple", "TrainingCorpus", "from_env"]

#: Below this the answer is a keystroke, not a sample worth training on. Matches
#: ``learn.TERSE_WORDS``, where the same threshold separates engagement from
#: substance; kept as its own constant because the two could reasonably diverge.
MIN_SAMPLE_WORDS = 8


@dataclass(frozen=True)
class Triple:
    """One exchange, in the author's words and the machine's."""

    question: str
    answer: str
    gap_kind: str = ""
    section: str = ""
    synthesis: str = ""
    #: Whether the patch built from this answer survived every gate. See the
    #: module docstring: stored, never filtered on here.
    merged: bool = False
    #: Which manuscript this came from, so a corpus assembled from several can
    #: still be split back apart.
    manuscript: str = ""

    @property
    def digest(self) -> str:
        payload = f"{self.question}\x00{self.answer}\x00{self.synthesis}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def substantive(self) -> bool:
        return len(self.answer.split()) >= MIN_SAMPLE_WORDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "question": self.question,
            "answer": self.answer,
            "gap_kind": self.gap_kind,
            "section": self.section,
            "synthesis": self.synthesis,
            "merged": self.merged,
            "manuscript": self.manuscript,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Triple:
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            gap_kind=str(data.get("gap_kind", "")),
            section=str(data.get("section", "")),
            synthesis=str(data.get("synthesis", "")),
            merged=bool(data.get("merged", False)),
            manuscript=str(data.get("manuscript", "")),
        )


@dataclass
class TrainingCorpus:
    """Append-only JSONL of exchanges, and the SFT export built from it."""

    path: Path
    triples: list[Triple] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def load(cls, path: Path | str) -> TrainingCorpus:
        corpus = cls(path=Path(path))
        if corpus.path.exists():
            for line in corpus.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    triple = Triple.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    # One corrupt line must not cost the rest of the history.
                    continue
                corpus.triples.append(triple)
                corpus._seen.add(triple.digest)
        return corpus

    def record(self, triple: Triple) -> bool:
        """Append one exchange. Returns whether it was new.

        Declines never reach here: there is no answer, so there is nothing to
        train on. An answer too short to be a sample is dropped for the same
        reason, and both facts stay visible in the journal, which counts every
        question either way.
        """

        if not triple.answer.strip() or not triple.substantive:
            return False
        if triple.digest in self._seen:
            return False
        self.triples.append(triple)
        self._seen.add(triple.digest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(triple.to_dict(), ensure_ascii=False) + "\n")
        return True

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export_sft(
        self,
        path: Path | str,
        *,
        merged_only: bool = False,
        include_synthesis: bool = False,
    ) -> int:
        """Write chat-format SFT samples and return how many were written.

        The assistant turn is the **author's** answer: the point of the corpus
        is a model that argues the way they do, so their text is the target and
        the machine's synthesis is not. ``include_synthesis`` adds the synthesis
        as a following turn for anyone training the dialectic role itself, and
        is off by default so the common export cannot accidentally teach a model
        to imitate its own output.

        ``merged_only`` restricts the export to exchanges whose patch survived
        every gate. Whoever runs the training decides that; capture does not.
        """

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out.open("w", encoding="utf-8") as handle:
            for triple in self.triples:
                if merged_only and not triple.merged:
                    continue
                messages = [
                    {"role": "user", "content": triple.question},
                    {"role": "assistant", "content": triple.answer},
                ]
                if include_synthesis and triple.synthesis.strip():
                    messages.append({"role": "assistant", "content": triple.synthesis})
                handle.write(
                    json.dumps(
                        {
                            "messages": messages,
                            "metadata": {
                                "gap_kind": triple.gap_kind,
                                "section": triple.section,
                                "merged": triple.merged,
                                "manuscript": triple.manuscript,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        return written

    def summary(self) -> dict[str, Any]:
        """Enough to decide whether there is a training set here yet.

        A count on its own reads as readiness. ``merged`` and ``authored_words``
        are what say whether the samples are the author's substantive work or a
        long tail of one-line replies.
        """

        return {
            "samples": len(self.triples),
            "merged": sum(1 for t in self.triples if t.merged),
            "with_synthesis": sum(1 for t in self.triples if t.synthesis.strip()),
            "authored_words": sum(len(t.answer.split()) for t in self.triples),
            "manuscripts": len({t.manuscript for t in self.triples if t.manuscript}),
            "gap_kinds": sorted({t.gap_kind for t in self.triples if t.gap_kind}),
        }


def from_env(env: dict[str, str] | None = None) -> TrainingCorpus | None:
    """The corpus the author opted into, or ``None`` — which is the default.

    Both variables are required and neither has a default path. A capture that
    could switch itself on, or write somewhere the author did not name, is not
    an opt-in.
    """

    source = os.environ if env is None else env
    if str(source.get("LRG_MAIEUTIC_TRAINING_CAPTURE", "")).strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None
    path = str(source.get("LRG_MAIEUTIC_TRAINING_PATH", "")).strip()
    if not path:
        return None
    return TrainingCorpus.load(path)
