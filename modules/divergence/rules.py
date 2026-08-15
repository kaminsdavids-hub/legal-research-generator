"""Loading and freezing the rule predicates. The scholarly contribution.

Each formalised reading of the law is a pure function ``FactVector -> Verdict``
living in its own file under a rules directory. A model may draft one; a human
reads it, decides it says what the reading actually says, and freezes it. The
frozen set's hash is stamped on every artifact, so a result can never be quietly
compared against one produced under different law.

**Nothing here calls a model.** The loader imports Python and calls functions.
That is the property that makes the search safe: the fitness function is
executed, and no rewording of anything changes what it returns.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from modules.casework.schema import (
    INAPPLICABLE,
    UNDETERMINED,
    WILDCARDS,
    Outcome,
    Scrutiny,
    Verdict,
    digest_of,
)

from .genotype import FactVector

__all__ = ["RuleSet", "RulePredicate", "load_rules"]


class _TrackedFacts:
    """A fact vector that remembers which axes a predicate consulted.

    The rules are frozen author artifacts and must not have to know that
    ``UNDETERMINED`` exists. This finds out, per evaluation, whether the reading
    actually depended on an axis the source is silent about -- which is the only
    honest basis for saying its verdict does not apply. Marking every rule
    uncertain whenever any axis is undetermined would erase real collisions
    (three of the four readings never look at ``harm_proximity``); leaving them
    alone would let a reading branch on the empty string and answer confidently.
    """

    __slots__ = ("_facts", "read")

    def __init__(self, facts: FactVector) -> None:
        self._facts = facts
        self.read: list[str] = []

    def __getitem__(self, axis: str) -> str:
        self.read.append(axis)
        return self._facts[axis]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._facts, name)


class RulePredicate(Protocol):
    """One formalised reading. Pure: same vector in, same verdict out."""

    def __call__(self, facts: FactVector) -> Verdict: ...


@dataclass(frozen=True)
class LoadedRule:
    name: str
    predicate: RulePredicate
    source: Path
    #: The author's one-line statement of whose reading this is. Required: an
    #: unattributed predicate is a rule nobody has taken responsibility for.
    reading: str


@dataclass
class RuleSet:
    rules: list[LoadedRule]
    digest: str

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def names(self) -> list[str]:
        return [rule.name for rule in self.rules]

    def verdicts(self, facts: FactVector) -> dict[str, Verdict]:
        """Every reading's verdict on one fact pattern.

        A predicate that raises is not silently dropped -- that would make a
        broken rule look like agreement, which is the failure mode with the
        worst consequences here. It returns ``uncertain`` with the exception in
        the rationale, so the disagreement measure sees a distinct verdict and
        the report says which rule failed and why.
        """

        # Generated fact patterns never carry UNDETERMINED, so the search path
        # -- millions of evaluations -- pays one scan of eight strings and then
        # runs exactly as before, untracked.
        silent_axes = {
            name: value for name, value in facts.values.items() if value in WILDCARDS
        }

        out: dict[str, Verdict] = {}
        for rule in self.rules:
            tracked = _TrackedFacts(facts) if silent_axes else None
            failure: Verdict | None = None
            try:
                # cast, not a subclass: the tracker deliberately does not
                # inherit FactVector, so a rule reaching past __getitem__
                # for an attribute is a real difference and not silently
                # untracked.
                seen = cast(FactVector, tracked) if tracked is not None else facts
                verdict = rule.predicate(seen)
            except Exception as exc:  # noqa: BLE001 - a broken rule must be visible
                verdict = failure = Verdict(
                    outcome=Outcome.UNCERTAIN,
                    scrutiny=Scrutiny.NONE,
                    rationale=f"rule raised {type(exc).__name__}: {exc}",
                )

            if tracked is not None:
                # Order-preserving, deduplicated: the rationale should read the
                # way the predicate was written.
                consulted = list(dict.fromkeys(a for a in tracked.read if a in silent_axes))
                if consulted:
                    # Reported ahead of any exception, because on an
                    # undetermined axis the exception *is* the silence -- a rule
                    # doing int("undetermined") has not malfunctioned, it has
                    # been asked a question its source cannot answer.
                    # Same verdict, different diagnosis, and the difference
                    # is what an author does next: chase the record, or accept
                    # that this reading has nothing to say about this case.
                    undetermined = [a for a in consulted if silent_axes[a] == UNDETERMINED]
                    inapplicable = [a for a in consulted if silent_axes[a] == INAPPLICABLE]
                    reasons = []
                    if undetermined:
                        reasons.append(
                            f"reads {', '.join(undetermined)}, which the source "
                            f"does not settle"
                        )
                    if inapplicable:
                        reasons.append(
                            f"reads {', '.join(inapplicable)}, which does not "
                            f"arise in this case"
                        )
                    verdict = Verdict(
                        outcome=Outcome.UNCERTAIN,
                        scrutiny=Scrutiny.NONE,
                        rationale="not applicable: " + "; ".join(reasons),
                    )
                    failure = None
            out[rule.name] = failure if failure is not None else verdict
        return out

    def labels(self, facts: FactVector) -> list[str]:
        return [verdict.label for verdict in self.verdicts(facts).values()]


def load_rules(directory: str | Path) -> RuleSet:
    """Import every ``rules/*.py`` and freeze the set.

    Files are loaded in sorted order and the digest covers their contents, so
    editing a predicate changes the stamp and starts a new epoch -- which is the
    intended friction. Comparing an archive across a rule change is comparing
    two different questions.
    """

    path = Path(directory)
    files = sorted(p for p in path.glob("*.py") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"no rule predicates in {path}")

    rules: list[LoadedRule] = []
    for file in files:
        module_name = f"_divergence_rule_{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
            raise ImportError(f"cannot load rule module: {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        predicate: Any = getattr(module, "verdict", None)
        if predicate is None or not callable(predicate):
            raise AttributeError(
                f"{file.name} defines no `verdict(facts) -> Verdict`; a rule file "
                "must expose exactly that"
            )
        reading = str(getattr(module, "READING", "")).strip()
        if not reading:
            raise AttributeError(
                f"{file.name} has no READING string. A predicate is a claim about "
                "what a reading of the law holds; it needs an author's statement "
                "of whose reading it is."
            )
        rules.append(
            LoadedRule(
                name=str(getattr(module, "NAME", file.stem)),
                predicate=predicate,
                source=file,
                reading=reading,
            )
        )

    return RuleSet(rules=rules, digest=digest_of(files))


def rule_files(directory: str | Path) -> Sequence[Path]:
    return sorted(p for p in Path(directory).glob("*.py") if not p.name.startswith("_"))
