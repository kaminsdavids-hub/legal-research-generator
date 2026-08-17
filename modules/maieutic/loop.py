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
from .corpus import TrainingCorpus, Triple
from .dialectic_adapter import Adaptation, adapt
from .graph import ArgumentGraph, Edge, EdgeType, GraphPatch, Node, NodeType, Provenance
from .grounding import GroundingGate, PatchGrounding
from .learn import Journal, LearnedPolicy, Outcome, gap_episode
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


class SelfGroundingEdge(Exception):
    """Raised when a stated dependency would close a DEPENDS_ON cycle.

    Refused rather than recorded: a cycle is the one structural defect no
    further material fixes, and a graph carrying one makes every reading order
    downstream a fiction.
    """


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
    #: What the author did with the questions they were asked. Read only by the
    #: learned policy, and never by the gates. Held here but persisted
    #: separately: it is a fact about the author, not about this manuscript, and
    #: one manuscript cannot supply enough evidence to act on.
    journal: Journal = field(default_factory=Journal)
    #: Verbatim exchanges kept for a future fine-tune, or ``None`` — the
    #: default. Off unless the author opted in; see
    #: :mod:`modules.maieutic.corpus` for why this one is not on by default when
    #: the journal is.
    training: TrainingCorpus | None = None
    #: Names this manuscript in the training corpus, so a corpus assembled from
    #: several can be split back apart.
    manuscript_id: str = ""

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
        engine = engine or SocraticEngine(rank=LearnedPolicy(self.journal).rank)
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
            #
            # The episode is still recorded as ANSWERED: they wrote a paragraph
            # and the machinery rejected it, which is a fact about the gates and
            # not about whether the question was worth asking.
            self.journal.record(gap_episode(question.gap, Outcome.ANSWERED, text))
            self._capture(question, text, adaptation, merged=False)
            return result

        # `patch`, not `adaptation.patch`: the reduced one is what the gates
        # judged, and merging anything else means merging something unjudged.
        # Recorded before the merge, and regardless of it: the author answered,
        # which is the only thing this journal is about.
        self.journal.record(gap_episode(question.gap, Outcome.ANSWERED, text))
        result.added = self.graph.apply(patch)
        result.merged = True
        self.asked.record(question)
        self.barren |= report.banality.barren_sections()
        self.pending = None
        self._capture(question, text, adaptation, merged=True)
        return result

    def _capture(
        self, question: Question, text: str, adaptation: Adaptation, *, merged: bool
    ) -> None:
        """Keep the exchange verbatim, if the author asked for it to be kept.

        Recorded on both paths. A refused patch still contains the author's own
        prose, and the reason the gates rejected it is a property of the
        citations it carried, not of how they write — dropping it here would
        train on their successes only and quietly narrow the sample to whatever
        the machinery happened to like.
        """

        if self.training is None:
            return
        self.training.record(
            Triple(
                question=question.text,
                answer=text,
                gap_kind=getattr(question.gap.kind, "value", str(question.gap.kind)),
                section=question.gap.section,
                synthesis=adaptation.synthesis,
                merged=merged,
                manuscript=self.manuscript_id,
            )
        )

    def depends_on(self, dependent: str, prerequisite: str) -> Edge:
        """The author states that one claim rests on another. An edit, not a patch.

        ``DEPENDS_ON`` is the only edge type nothing in this package creates.
        That is deliberate and it is why this method exists: the edge asserts
        that a reader *must accept one claim before another*, which is a claim
        about the argument's structure, and the machine inferring it would be
        the machine deciding what the paper's reasoning depends on. Every
        DEPENDS_ON edge in a graph is therefore the author's own — an invariant
        worth more than an edge-level provenance field, because it holds by
        construction rather than by record-keeping.

        It is also an *edit*: it adds no node, and :class:`GraphPatch` refuses a
        patch that asserts nothing new (see ``dialectic_adapter``, which names
        this case exactly). So it goes through here, where the one thing that
        can go wrong is checked.

        That one thing is a cycle. An argument that grounds itself is the gap
        the Socratic engine ranks first, and it is far easier to create by
        adding an edge than by adding a node. The edge is applied only if the
        result is acyclic; otherwise nothing changes and the caller is told
        which chain it would have closed.
        """

        for node_id in (dependent, prerequisite):
            if node_id not in self.graph.nodes:
                raise KeyError(f"no such node: {node_id}")
        if dependent == prerequisite:
            raise ValueError("a claim cannot depend on itself")

        edge = Edge(source=dependent, target=prerequisite, type=EdgeType.DEPENDS_ON)
        if any(
            e.source == edge.source and e.target == edge.target and e.type is edge.type
            for e in self.graph.edges
        ):
            return edge  # already stated; saying it twice asserts nothing new

        self.graph.edges.append(edge)
        cycle = self.graph.depends_on_cycle()
        if cycle:
            self.graph.edges.remove(edge)
            raise SelfGroundingEdge(
                f"{dependent} depending on {prerequisite} would close a cycle: "
                + " -> ".join(cycle)
                + ". One of these has to be prior; the argument cannot ground itself."
            )
        return edge

    def retract_dependency(self, dependent: str, prerequisite: str) -> bool:
        """Take back a stated dependency. Returns whether one was removed.

        The complement of :meth:`depends_on`, and needed for the same reason it
        exists: if the author is the only one who may say a claim rests on
        another, they are also the only one who can say it does not. A wrong
        dependency is not inert — it reorders the manuscript and makes the
        renderer assert a reasoning path ("as argued in Part I") that the
        argument does not contain.

        Retracting is safe in a way stating is not. Removing a precedence
        constraint can never make an order infeasible, so there is no cycle
        check here and nothing to roll back. What it does change is the reading
        order, which the caller should say out loud.

        **Only DEPENDS_ON.** SUPPORTS and ATTACKS are the argument's substance;
        removing one is deleting a relation the author or the dialectic
        established, which is a different act with different consequences and
        does not belong behind a flag on this command.
        """

        for node_id in (dependent, prerequisite):
            if node_id not in self.graph.nodes:
                raise KeyError(f"no such node: {node_id}")

        before = len(self.graph.edges)
        self.graph.edges = [
            e
            for e in self.graph.edges
            if not (
                e.type is EdgeType.DEPENDS_ON
                and e.source == dependent
                and e.target == prerequisite
            )
        ]
        return len(self.graph.edges) < before

    def decline(self) -> Question | None:
        """The author passes on the pending question.

        The only negative signal the learned policy gets, and the reason it
        exists: without a decline path a skipped question is indistinguishable
        from one never reached, and the policy would learn nothing from the
        questions that were not worth asking.
        """
        question = self.pending
        if question is None:
            return None
        self.journal.record(gap_episode(question.gap, Outcome.DECLINED))
        self.asked.record(question)
        self.pending = None
        return question

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
            # The label, not the corpus: what was captured lives in its own
            # file, and a session that is copied or shared must not carry the
            # author's exchanges along inside it.
            "manuscript_id": self.manuscript_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        pending = data.get("pending")
        return cls(
            graph=ArgumentGraph.from_dict(data.get("graph", {})),
            asked=AskedLog(keys=set(data.get("asked", []))),
            barren=set(data.get("barren", [])),
            pending=_question_from_dict(pending) if pending else None,
            manuscript_id=str(data.get("manuscript_id", "")),
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
