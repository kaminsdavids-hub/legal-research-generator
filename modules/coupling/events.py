"""The event log the two modules talk through, and the epoch discipline on it.

Divergence and Portfolio are coupled and must not call each other. Divergence is
expensive and episodic — it runs when a rule predicate changed. Portfolio is
cheap and runs at render time. A direct call would tie the cheap thing to the
expensive one and make render-time latency depend on a search.

So they exchange events, and the exchange is damped.

**Why damping is not optional.** The edges form a cycle: a MishandledPrecedent
creates a proposition needing authority, Portfolio reports that proposition as
thinly covered, ThinCoverage raises the search prior on the region the
proposition occupies, and Divergence generates more findings there. Undamped,
the two modules would spend every epoch elaborating one corner of the feature
space, each responding to the other's last output, and the archive would stop
being a survey of hard cases.

The discipline: Divergence runs **once per epoch against a frozen snapshot**,
and signals raised during an epoch apply only to the **next** one. Nothing a run
produces can change what that same run explores.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = ["Event", "EventKind", "EventLog", "EpochManager"]


class EventKind(StrEnum):
    #: Divergence found a coded real case its rules classify inconsistently.
    #: Creates a proposition the manuscript now has to address.
    MISHANDLED_PRECEDENT = "mishandled_precedent"
    #: Portfolio found a proposition supportable only thinly. Raises the search
    #: prior on the feature region that proposition occupies -- next epoch.
    THIN_COVERAGE = "thin_coverage"
    #: An epoch boundary. Recorded so an artifact can be placed in time.
    EPOCH_OPENED = "epoch_opened"
    #: A rule predicate or the schema changed, which is what forces a new epoch.
    EVALUATOR_CHANGED = "evaluator_changed"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    epoch: int
    payload: dict[str, Any] = field(default_factory=dict)
    #: Which module emitted it. A log a reader cannot attribute is a log that
    #: cannot be argued with.
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "epoch": self.epoch,
            "source": self.source,
            "payload": self.payload,
        }


@dataclass
class EventLog:
    """Append-only. Events are facts about what happened, not state to mutate."""

    events: list[Event] = field(default_factory=list)
    path: Path | None = None

    def append(self, event: Event) -> Event:
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
        return event

    def of_kind(self, kind: EventKind, epoch: int | None = None) -> list[Event]:
        return [
            e
            for e in self.events
            if e.kind is kind and (epoch is None or e.epoch == epoch)
        ]

    @classmethod
    def load(cls, path: str | Path) -> EventLog:
        file = Path(path)
        log = cls(path=file)
        if not file.exists():
            return log
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            log.events.append(
                Event(
                    kind=EventKind(data["kind"]),
                    epoch=int(data["epoch"]),
                    payload=data.get("payload", {}),
                    source=data.get("source", ""),
                )
            )
        return log


@dataclass
class EpochManager:
    """Whose turn it is, and what each side is allowed to see.

    The single rule this class exists to enforce: a run reads signals from
    epochs strictly *before* its own. Everything else here is bookkeeping.
    """

    log: EventLog
    epoch: int = 0

    def open_epoch(self, reason: str = "") -> int:
        self.epoch += 1
        self.log.append(
            Event(
                kind=EventKind.EPOCH_OPENED,
                epoch=self.epoch,
                payload={"reason": reason},
                source="coupling",
            )
        )
        return self.epoch

    def evaluator_changed(self, schema_digest: str, rules_digest: str) -> int:
        """A rule or the schema changed: start a new epoch and say why.

        Archives from before this point are not comparable to those after it --
        their cells were scored by different functions -- and the log is where
        that boundary is recorded so nobody merges across it later.
        """

        self.log.append(
            Event(
                kind=EventKind.EVALUATOR_CHANGED,
                epoch=self.epoch,
                payload={"schema_digest": schema_digest, "rules_digest": rules_digest},
                source="coupling",
            )
        )
        return self.open_epoch(reason="evaluator changed")

    # ------------------------------------------------------------------ #
    # The damped exchange
    # ------------------------------------------------------------------ #
    def record_precedents(self, findings: Iterable[dict[str, Any]]) -> list[Event]:
        return [
            self.log.append(
                Event(
                    kind=EventKind.MISHANDLED_PRECEDENT,
                    epoch=self.epoch,
                    payload=dict(finding),
                    source="divergence",
                )
            )
            for finding in findings
        ]

    def record_thin_coverage(self, findings: Iterable[dict[str, Any]]) -> list[Event]:
        return [
            self.log.append(
                Event(
                    kind=EventKind.THIN_COVERAGE,
                    epoch=self.epoch,
                    payload=dict(finding),
                    source="portfolio",
                )
            )
            for finding in findings
        ]

    def priors_for(self, epoch: int) -> list[dict[str, Any]]:
        """ThinCoverage signals a run at ``epoch`` may act on.

        Strictly earlier epochs. This one line is the damping: a signal raised
        during the current epoch is invisible to the current run, so Divergence
        cannot chase a gap that its own findings created moments earlier.
        """

        return [
            event.payload
            for event in self.log.of_kind(EventKind.THIN_COVERAGE)
            if event.epoch < epoch
        ]

    def propositions_for(self, epoch: int) -> list[dict[str, Any]]:
        """Coverage requirements created by Divergence before ``epoch``.

        Same rule in the other direction: a precedent found this epoch becomes
        Portfolio's problem next epoch, not mid-run.
        """

        return [
            event.payload
            for event in self.log.of_kind(EventKind.MISHANDLED_PRECEDENT)
            if event.epoch < epoch
        ]

    def snapshot(self, epoch: int) -> dict[str, Any]:
        """What a run at ``epoch`` is allowed to see, as one frozen object."""

        return {
            "epoch": epoch,
            "priors": self.priors_for(epoch),
            "propositions": self.propositions_for(epoch),
        }


def regions_from_priors(
    priors: Sequence[dict[str, Any]], key: str = "features"
) -> list[dict[str, str]]:
    """Feature regions named by ThinCoverage signals, deduplicated.

    A signal without coded features contributes nothing rather than being
    guessed at: "this proposition is thinly covered" says where in the argument
    the weakness is, and only a coding says where in the feature space it sits.
    """

    seen: list[dict[str, str]] = []
    for prior in priors:
        features = prior.get(key)
        if isinstance(features, dict) and features:
            candidate = {str(k): str(v) for k, v in features.items()}
            if candidate not in seen:
                seen.append(candidate)
    return seen
