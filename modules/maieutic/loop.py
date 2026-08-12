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
from .graph import ArgumentGraph, GraphPatch, Node, NodeType, Provenance
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

    @classmethod
    def live(cls, settings: Any) -> Gates:
        """Gates matched to a live exchange.

        A live run retrieves and verifies citations, so it produces AUTHORITY
        nodes — and the offline grounding gate has no verifier and refuses every
        one of them. Running the live engine behind offline gates therefore
        refuses whole exchanges at the fabrication wall for want of a verifier,
        not for want of an authority. The two halves have to be configured
        together (REMEDIATION §12.2).

        The corpus doubles as the banality gate's view of the literature: the
        same passages that confirm an authority are what the field already says.
        """
        from legal_research.citations.corpus import load_corpus

        from .service import build_grounding_gate

        sources: list[Source] = []
        try:
            corpus = load_corpus(settings.corpus_path)
        except Exception:  # noqa: BLE001 - no corpus is a real deployment state
            corpus = None
        if corpus is not None:
            for record in getattr(corpus, "records", []):
                for passage in getattr(record, "passages", []):
                    sources.append(Source(str(getattr(record, "title", "")), passage))

        return cls(
            grounding=build_grounding_gate(settings),
            banality=BanalityGate(sources),
        )

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
    #: Set when the dialectic exchange failed. The answer still merges; what was
    #: lost is the machine's pressure, and the author must be told which.
    turn_error: str = ""
    #: Machine-proposed nodes dropped so the author's answer could merge.
    dropped: list[str] = field(default_factory=list)

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

        turn = None
        turn_error = ""
        if turns is not None:
            try:
                turn = turns(question.text, text)
            except Exception as exc:  # noqa: BLE001 - see below
                # A model being down costs the machine's pressure, never the
                # author's own answer. Their work is not hostage to an exchange
                # that failed, so the patch proceeds carrying what they wrote.
                turn_error = f"{type(exc).__name__}: {exc}"

        adaptation = adapt(question, text, turn)
        patch = adaptation.patch

        report = gates.check(patch, self.graph)
        dropped: list[str] = []
        if not report.passed:
            reduced = _without_ungrounded_machine_nodes(patch, self.graph, report)
            if reduced is not None:
                patch, dropped = reduced
                report = gates.check(patch, self.graph)

        result = StepResult(
            question=question,
            answer=text,
            report=report,
            unresolved=adaptation.unresolved,
            adaptation=adaptation,
            turn_error=turn_error,
            dropped=dropped,
        )
        if not report.passed:
            # The question stays pending. A refused patch means the author has
            # not yet answered, not that the gap has been dealt with.
            return result

        # `patch`, not `adaptation.patch`: the reduced one is what the gates
        # judged, and merging anything else means merging something unjudged.
        result.added = self.graph.apply(patch)
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


def _without_ungrounded_machine_nodes(
    patch: GraphPatch, graph: ArgumentGraph, report: GateReport
) -> tuple[GraphPatch, list[str]] | None:
    """Drop the machine's ungrounded nodes so the author's answer can still merge.

    Grounding is all-or-nothing across a patch, and that rule was written to stop
    a fabricated claim entering the manuscript beside verified material. In the
    assembled loop it had a consequence nobody chose: the party that cites badly
    is the *machine*, and the party that loses their work is the *author*. Every
    live exchange failed this way, because a model that retrieves two authorities
    the source does not support takes the author's own paragraph down with them
    (REMEDIATION §12.3).

    What actually matters is that nothing ungrounded reaches the manuscript, and
    dropping the offending nodes secures that just as well as refusing the patch.
    So the atomicity of a patch gives way, and only across parties: if any node
    the *author* wrote fails grounding, the patch still fails whole.

    Returns None when this cannot apply — when an author node failed, when a
    non-grounding gate failed, or when nothing would be left.
    """
    failures = {r.node_id for r in report.grounding.failures}
    if not failures:
        return None
    # Only grounding may be resolved this way. A coherence failure is structural
    # and a banality failure is the author's own, and neither is repaired by
    # deleting somebody else's node.
    if not report.coherence.passed or not report.banality.passed:
        return None

    by_id = {n.id: n for n in patch.nodes}
    if any(
        by_id[node_id].provenance is Provenance.HUMAN
        for node_id in failures
        if node_id in by_id
    ):
        return None

    kept = [n for n in patch.nodes if n.id not in failures]
    if not kept:
        return None
    known = {n.id for n in kept} | set(graph.nodes)
    edges = [e for e in patch.edges if e.source in known and e.target in known]
    return (
        GraphPatch(nodes=kept, edges=edges, rationale=patch.rationale),
        sorted(failures),
    )


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
