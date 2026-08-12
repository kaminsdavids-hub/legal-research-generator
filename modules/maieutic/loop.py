"""The cycle: ask, answer, gate, merge.

One place where the whole loop is assembled, kept separate from the CLI so that
what happens is testable without going through argv and stdout.

The ordering of the four gates is not arbitrary. Coherence runs first because a
structurally broken patch makes the other verdicts meaningless — an edge pointing
at nothing has no relation to judge. Grounding runs before novelty because a
fabricated citation should be reported as fabrication, not as restatement.
Banality runs last because it is the weakest and its verdict is the least
actionable.

**Every gate runs even when an earlier one has already failed.** Reporting the
first refusal and stopping would make the author fix one problem, resubmit, and
discover the next — and the whole system's premise is that their attention is the
scarce resource.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from modules.dialectic.models import DialecticTurn

from .banality import BanalityGate, PatchBanality, Source
from .coherence import CoherenceGate, PatchCoherence
from .dialectic_adapter import Adaptation, adapt
from .graph import ArgumentGraph, GraphPatch, Node, NodeType
from .grounding import GroundingGate, PatchGrounding
from .novelty import LexicalEmbedder, NoveltyGate, PatchNovelty
from .render import Audience, Manuscript, render
from .socratic import AskedLog, Gap, GapKind, Phrasing, Question, SocraticEngine, analyse


class TurnProvider(Protocol):
    """Runs a dialectic exchange over an answer. Absent by default: see D11."""

    def __call__(self, question: str, answer: str) -> DialecticTurn | None: ...


@dataclass
class GateReport:
    coherence: PatchCoherence
    grounding: PatchGrounding
    novelty: PatchNovelty
    banality: PatchBanality

    @property
    def passed(self) -> bool:
        return (
            self.coherence.passed
            and self.grounding.passed
            and self.novelty.passed
            and self.banality.passed
        )

    @property
    def refusals(self) -> list[str]:
        """Every reason this patch cannot merge, not just the first one."""
        reasons = []
        for finding in self.coherence.blocking:
            reasons.append(f"coherence: {finding.kind.value} — {finding.detail}")
        for result in self.grounding.failures:
            kind = result.failure.value if result.failure else "ungrounded"
            reasons.append(f"grounding: {kind} — {result.detail}")
        if not self.novelty.passed:
            reasons.append("novelty: nothing in this patch is new to the manuscript")
        for assessment in self.banality.assessments:
            if assessment.blocking:
                reasons.append(f"banality: {assessment.verdict.value} — {assessment.detail}")
        return reasons

    @property
    def advisories(self) -> list[str]:
        """Reported, never blocking. A model's opinion is not a veto."""
        return [
            f"coherence: {f.kind.value} — {f.detail}" for f in self.coherence.advisory
        ] + [
            f"banality: {a.verdict.value} — {a.detail}" for a in self.banality.advisory
        ]


@dataclass
class Gates:
    coherence: CoherenceGate = field(default_factory=CoherenceGate)
    grounding: GroundingGate = field(default_factory=GroundingGate)
    novelty: NoveltyGate = field(default_factory=lambda: NoveltyGate(LexicalEmbedder()))
    banality: BanalityGate = field(default_factory=BanalityGate)

    @classmethod
    def offline(cls, sources: list[Source] | None = None) -> Gates:
        """The default: no network, no models, and honest about it.

        The grounding gate has no verifier, so it fails closed on any AUTHORITY
        node. That is correct rather than inconvenient — offline, nothing can
        confirm a citation, and merging one on trust is the failure the gate
        exists to prevent.
        """
        return cls(banality=BanalityGate(sources or []))

    def check(self, patch: GraphPatch, graph: ArgumentGraph) -> GateReport:
        return GateReport(
            coherence=self.coherence.assess(patch, graph),
            grounding=self.grounding.assess_patch(patch, graph),
            novelty=self.novelty.assess_patch(patch, graph),
            banality=self.banality.assess_patch(patch, graph),
        )


@dataclass
class StepResult:
    question: Question | None = None
    answer: str = ""
    merged: bool = False
    added: list[str] = field(default_factory=list)
    report: GateReport | None = None
    #: True when the answer could not close the gap it responded to.
    unresolved: bool = False
    adaptation: Adaptation | None = None

    @property
    def refusals(self) -> list[str]:
        return self.report.refusals if self.report else []


@dataclass
class Session:
    """A manuscript in progress, and what has already been asked about it."""

    graph: ArgumentGraph = field(default_factory=ArgumentGraph)
    asked: AskedLog = field(default_factory=AskedLog)
    #: Sections where a merge produced no substantive claim. Accumulated across
    #: cycles, because one patch is too small a sample to act on.
    barren: set[str] = field(default_factory=set)
    #: The question awaiting an answer, so `ask` and `answer` cannot disagree.
    pending: Question | None = None

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def begin(self, thesis: str, section: str = "") -> Node:
        """Open a manuscript with the author's own thesis."""
        if self.graph.nodes:
            raise ValueError("this manuscript has already been started")
        node = Node.from_human(NodeType.THESIS, thesis.strip(), section=section)
        self.graph.apply(GraphPatch(nodes=[node], rationale="opening thesis"))
        return node

    def ask(self, engine: SocraticEngine | None = None) -> Question | None:
        """The next question, or None when there is no unasked gap left."""
        engine = engine or SocraticEngine()
        questions = engine.ask(
            self.graph, self.asked, limit=1, avoid_sections=frozenset(self.barren)
        )
        self.pending = questions[0] if questions else None
        return self.pending

    def answer(
        self,
        text: str,
        gates: Gates | None = None,
        turns: TurnProvider | None = None,
    ) -> StepResult:
        """Record an answer, run every gate, and merge only if all of them pass."""
        question = self.pending
        if question is None:
            raise ValueError("no question is pending; call ask() first")

        gates = gates or Gates.offline()
        turn = turns(question.text, text) if turns else None
        adaptation = adapt(question, text, turn)

        report = gates.check(adaptation.patch, self.graph)
        result = StepResult(
            question=question,
            answer=text,
            report=report,
            unresolved=adaptation.unresolved,
            adaptation=adaptation,
        )
        if not report.passed:
            # The question stays pending. A refused patch means the author has
            # not yet answered, not that the gap has been dealt with.
            return result

        result.added = self.graph.apply(adaptation.patch)
        result.merged = True
        self.asked.record(question)
        self.barren |= report.banality.barren_sections()
        self.pending = None
        return result

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def gaps(self) -> list[Gap]:
        return analyse(self.graph)

    def outstanding(self) -> list[Gap]:
        return self.asked.outstanding(self.graph)

    def manuscript(self, audience: Audience = Audience.MANUSCRIPT) -> Manuscript:
        return render(self.graph, audience)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "asked": sorted(self.asked.keys),
            "barren": sorted(self.barren),
            "pending": _question_to_dict(self.pending) if self.pending else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        pending = data.get("pending")
        return cls(
            graph=ArgumentGraph.from_dict(data.get("graph", {})),
            asked=AskedLog(keys=set(data.get("asked", []))),
            barren=set(data.get("barren", [])),
            pending=_question_from_dict(pending) if pending else None,
        )

    def save(self, path: Path) -> None:
        """Write-then-rename, so an interrupted save cannot truncate a manuscript."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Session:
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _question_to_dict(question: Question) -> dict[str, Any]:
    return {
        "text": question.text,
        "phrasing": question.phrasing.value,
        "gap": {
            "kind": question.gap.kind.value,
            "node_ids": list(question.gap.node_ids),
            "detail": question.gap.detail,
            "section": question.gap.section,
        },
    }


def _question_from_dict(data: dict[str, Any]) -> Question:
    gap = data["gap"]
    return Question(
        gap=Gap(
            kind=GapKind(gap["kind"]),
            node_ids=tuple(gap["node_ids"]),
            detail=gap.get("detail", ""),
            section=gap.get("section", ""),
        ),
        text=data["text"],
        phrasing=Phrasing(data.get("phrasing", "template")),
    )
