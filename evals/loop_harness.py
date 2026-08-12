"""Measuring the maieutic loop, as opposed to the dialectic module inside it.

`evals/harness.py` asks whether a dialectic exchange is any good. This asks a
different question: given an author who answers, what does the loop actually do
for them? How much of the machine's output survives the gates, how many
objections worth answering does an exchange yield, how deep can the loop go
before it runs dry.

**The author's answers are scripted, and that is the whole design.** A harness
that let a model play the author would have the system ask itself questions,
answer them, and grade the result — the self-satisfying arrangement that has
already cost this repository three gates (REMEDIATION §5, §11.5, §11.11a). The
answers come from a fixture, outside the loop, and the machinery faces text it
did not write. See D22 for what that establishes and what it does not.

**Missing data is not zero.** The lesson from §11.8, where collapsing a broken
judge to 0.0 turned the best cluster into the worst. Authority survival is
``None`` when no authority was proposed — nothing was tested, so nothing is
known. It is 0.0 only when authorities were proposed and none survived. The
distinction is the difference between "the wall held" and "the wall was never
approached".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from modules.maieutic.graph import NodeType, Provenance
from modules.maieutic.loop import Gates, Session, StepResult, TurnProvider


@dataclass(frozen=True)
class SessionScript:
    """A thesis and the answers an author gives, in order."""

    id: str
    thesis: str
    answers: list[str]
    section: str = ""
    #: Where the answers came from. Recorded on every report, because a run over
    #: stand-in answers and a run over a real transcript are not comparable.
    source: str = "fixture"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionScript:
        return cls(
            id=str(data["id"]),
            thesis=str(data["thesis"]),
            answers=[str(a) for a in data["answers"]],
            section=str(data.get("section", "")),
            source=str(data.get("source", "fixture")),
        )


def load_scripts(path: Path) -> list[SessionScript]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SessionScript.from_dict(raw) for raw in data["sessions"]]


@dataclass
class ExchangeRecord:
    """What one answer cost and what it bought."""

    index: int
    gap_kind: str = ""
    merged: bool = False
    #: Nodes added, split by who authored them.
    author_nodes: int = 0
    machine_nodes: int = 0
    objections: int = 0
    #: AUTHORITY nodes the exchange proposed, and how many cleared grounding.
    authorities_proposed: int = 0
    authorities_grounded: int = 0
    #: Every proposed authority: its claim, its citation, and whether it merged.
    #: Counts alone say a wall held; the pairs say what it held against, and are
    #: what makes a scorer comparison possible without re-running the models.
    authorities: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0
    refusals: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    novelty_delta: float | None = None
    turn_error: str = ""
    #: Propositions the adapter refused at the boundary (citation strings,
    #: rescinded authority). Never silently discarded, so never silently unmeasured.
    boundary_refusals: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def authority_survival(self) -> float | None:
        """Share of proposed authorities that cleared grounding.

        ``None`` when none were proposed: the wall was never approached, which
        is not the same as it holding. Averaging those in as 0.0 would report an
        offline run — where no authority is ever proposed — as total failure.
        """
        if self.authorities_proposed == 0:
            return None
        return self.authorities_grounded / self.authorities_proposed


@dataclass
class SessionReport:
    script_id: str
    source: str
    mode: str
    exchanges: list[ExchangeRecord] = field(default_factory=list)
    #: True when the loop had no unasked gap left before the answers ran out.
    ran_dry: bool = False
    answers_unused: int = 0
    nodes: int = 0
    open_problems: int = 0
    error: str = ""

    @property
    def answer_survival(self) -> float | None:
        """Share of answers that merged. The author's side of the bargain.

        This is the number REMEDIATION §12.3 was about: it sat at 0.0 for every
        live exchange while the machine's bad citations took the author's
        paragraph down with them.
        """
        attempted = [e for e in self.exchanges if not e.error]
        if not attempted:
            return None
        return sum(1 for e in attempted if e.merged) / len(attempted)

    @property
    def objections_per_exchange(self) -> float | None:
        """Pressure delivered. A real 0 is meaningful: the machine said nothing."""
        merged = [e for e in self.exchanges if e.merged]
        if not merged:
            return None
        return mean(e.objections for e in merged)

    @property
    def authority_survival(self) -> float | None:
        proposed = sum(e.authorities_proposed for e in self.exchanges)
        if proposed == 0:
            return None
        return sum(e.authorities_grounded for e in self.exchanges) / proposed

    @property
    def seconds(self) -> float:
        return sum(e.seconds for e in self.exchanges)


@dataclass
class LoopReport:
    mode: str
    sessions: list[SessionReport] = field(default_factory=list)

    def _means(self, attribute: str) -> float | None:
        values = [
            value
            for session in self.sessions
            if (value := getattr(session, attribute)) is not None
        ]
        return mean(values) if values else None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sessions": len(self.sessions),
            "sessions_errored": sum(1 for s in self.sessions if s.error),
            "exchanges": sum(len(s.exchanges) for s in self.sessions),
            "answer_survival": self._means("answer_survival"),
            "objections_per_exchange": self._means("objections_per_exchange"),
            "authority_survival": self._means("authority_survival"),
            "ran_dry": sum(1 for s in self.sessions if s.ran_dry),
            "seconds": round(sum(s.seconds for s in self.sessions), 1),
        }


def run_script(
    script: SessionScript,
    gates: Gates | None = None,
    turns: TurnProvider | None = None,
    mode: str = "offline",
) -> SessionReport:
    """Replay one scripted session through the loop.

    Errors are isolated per exchange: one answer that blows up must not discard
    the measurements from the answers before it.
    """
    report = SessionReport(script_id=script.id, source=script.source, mode=mode)
    session = Session()
    try:
        session.begin(script.thesis, script.section)
    except Exception as exc:  # noqa: BLE001 - a bad script is data, not a crash
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    for index, answer in enumerate(script.answers):
        if session.ask() is None:
            report.ran_dry = True
            report.answers_unused = len(script.answers) - index
            break

        record = ExchangeRecord(index=index)
        record.gap_kind = session.pending.gap.kind.value if session.pending else ""
        started = time.monotonic()
        try:
            record = _measure(record, session, session.answer(answer, gates, turns))
        except Exception as exc:  # noqa: BLE001 - isolate, do not discard the run
            record.error = f"{type(exc).__name__}: {exc}"
        record.seconds = time.monotonic() - started
        report.exchanges.append(record)

    report.nodes = len(session.graph.nodes)
    report.open_problems = len(session.manuscript().open_problems)
    return report


def _measure(
    record: ExchangeRecord, session: Session, result: StepResult
) -> ExchangeRecord:
    record.merged = result.merged
    record.dropped = len(result.dropped)
    record.turn_error = result.turn_error
    record.refusals = list(result.refusals)
    if result.report:
        record.advisories = list(result.report.advisories)
        record.novelty_delta = result.report.novelty.delta
    if result.adaptation:
        record.boundary_refusals = len(result.adaptation.refused)
        proposed = [
            n for n in result.adaptation.patch.nodes if n.type is NodeType.AUTHORITY
        ]
        record.authorities_proposed = len(proposed)
        # Grounded means it survived to the graph, which is the only claim worth
        # making: the gate's verdict and the merge can disagree, and what the
        # manuscript contains is what happened.
        record.authorities_grounded = sum(1 for n in proposed if n.id in session.graph.nodes)
        record.authorities = [
            {
                "citation": n.citation,
                "claim": n.text,
                "grounded": n.id in session.graph.nodes,
            }
            for n in proposed
        ]

    for node_id in result.added:
        node = session.graph.nodes[node_id]
        if node.provenance is Provenance.HUMAN:
            record.author_nodes += 1
        else:
            record.machine_nodes += 1
        if node.type is NodeType.OBJECTION and node.provenance is not Provenance.HUMAN:
            record.objections += 1
    return record


def run_all(
    scripts: list[SessionScript],
    gates: Gates | None = None,
    turns: TurnProvider | None = None,
    mode: str = "offline",
) -> LoopReport:
    report = LoopReport(mode=mode)
    for script in scripts:
        report.sessions.append(run_script(script, gates, turns, mode))
    return report
