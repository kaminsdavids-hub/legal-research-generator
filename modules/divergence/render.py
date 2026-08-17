"""Turning a scored genotype into a readable hypothetical. Strictly downstream.

The search is over. This stage takes a fact pattern the frozen predicates have
*already* scored and writes it as prose a reader can react to — because
``distribution_modality=weights, actor_type=foreign_state_entity, capability_tier=4``
is a point in a search space, and a referee needs a paragraph.

**It cannot reach the fitness function, and that is structural rather than
promised.** This module imports no rule, no mask and no scorer; it takes an
already-computed :class:`Scores` and a genotype. There is nothing here for an
objective to call, and a test asserts that :mod:`objectives` and :mod:`mapelites`
do not import this module. If prose could influence a score, an evolutionary
search would optimise the prose.

Two guards on what comes back, because this is the one place a model writes
into the artifact:

* **No citations.** A rendered hypothetical that cites a case has invented one —
  nothing in the genotype carries authority, so any reporter cite in the output
  came from the model's weights. Reused from ``modules.dialectic.channel``,
  which enforces the same rule on the dialectic turns.
* **No verdict.** The prose states the facts; it does not say which reading
  wins. That is the question the hypothetical exists to pose, and a model
  answering it would be supplying the analysis the author is supposed to write.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from modules.dialectic.channel import CitationChannel, CitationDetected

__all__ = ["Hypothetical", "RenderError", "render_hypothetical", "template_prose"]


class RenderError(Exception):
    """Raised when a rendered hypothetical breaks one of the two guards."""


#: Phrases that answer the question instead of posing it. Deliberately narrow --
#: this catches a model stating the outcome, not a model using the word
#: "permitted" while describing what a party sought.
_VERDICT_LANGUAGE = re.compile(
    r"\b(?:therefore|so|thus|accordingly|it follows that)\b[^.]{0,80}\b"
    r"(?:is|would be|must be|should be)\s+(?:permitted|restricted|lawful|unlawful)\b"
    r"|\bthe (?:better|correct|right) (?:reading|view|answer)\b"
    r"|\bthe court (?:should|would|must) (?:hold|find|conclude)\b",
    re.IGNORECASE,
)

_SYSTEM = (
    "TASK: write\n"
    "You turn a coded fact pattern into one paragraph of neutral prose for a "
    "law-review hypothetical.\n"
    "State the facts the coding gives you and nothing else. Do NOT say how the "
    "law applies, which reading is correct, or what a court would hold — the "
    "hypothetical exists to pose that question.\n"
    "Cite nothing. You have no authorities; any case or statute you name would "
    "be invented.\n"
    "Return the paragraph only."
)


class ChatClient(Protocol):
    def chat(self, messages: Any, config: Any = None) -> str: ...


@dataclass(frozen=True)
class Hypothetical:
    """A scored fact pattern, written out."""

    facts: Mapping[str, str]
    prose: str
    #: "model" or "template". A reader must be able to tell which, because the
    #: template is deliberately flat and the model's version is not.
    source: str
    scores: Mapping[str, float] = field(default_factory=dict)
    verdicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [self.prose, ""]
        if self.verdicts:
            # The verdicts sit beside the prose, never inside it: what the
            # readings say is data from the frozen predicates, and mixing it
            # into the paragraph would make it look like part of the narrative.
            lines.append(f"Readings split: {', '.join(self.verdicts)}")
        if self.scores:
            lines.append(
                "disagreement={disagreement:.2f} brittleness={brittleness:.0f}".format(
                    disagreement=self.scores.get("disagreement", 0.0),
                    brittleness=self.scores.get("brittleness", 0.0),
                )
            )
        for warning in self.warnings:
            lines.append(f"[{warning}]")
        return "\n".join(lines).strip()


def template_prose(facts: Mapping[str, str]) -> str:
    """Deterministic prose. The fallback, and the thing the model is checked against.

    Flat by design: it names every axis and asserts nothing. When no model is
    configured this is what ships, and the artifact says ``source=template`` so
    nobody mistakes a list of features for a written hypothetical.
    """

    readable = {name.replace("_", " "): value.replace("_", " ") for name, value in facts.items()}
    parts = [f"{name}: {value}" for name, value in sorted(readable.items())]
    return "Consider a case with the following features — " + "; ".join(parts) + "."


def _check(prose: str, facts: Mapping[str, str]) -> list[str]:
    """The two guards. Raises on a citation; warns on thin coverage of the facts."""

    channel = CitationChannel()
    try:
        channel.scan(prose, role="assistant")
    except CitationDetected as exc:
        raise RenderError(
            "the rendered hypothetical cites authority it cannot have: "
            f"{exc.hits}. Nothing in a genotype carries a citation, so this came "
            "from the model's weights."
        ) from exc

    if _VERDICT_LANGUAGE.search(prose):
        raise RenderError(
            "the rendered hypothetical states an outcome. The prose poses the "
            "question; the frozen predicates answer it, and their verdicts are "
            "reported separately."
        )

    warnings: list[str] = []
    lowered = prose.lower()
    missing = [
        name
        for name, value in facts.items()
        if value.replace("_", " ").lower() not in lowered
        and value.lower() not in lowered
    ]
    if missing:
        # A warning rather than an error: a fluent paragraph may render
        # "capability_tier=5" as "a frontier-scale model" and be better prose
        # for it. The author should still be told which axes went unmentioned,
        # because an unmentioned axis is one the reader cannot react to.
        warnings.append(
            "not visibly mentioned in the prose: " + ", ".join(sorted(missing))
        )
    return warnings


def render_hypothetical(
    facts: Mapping[str, str],
    *,
    scores: Mapping[str, float] | None = None,
    verdicts: Sequence[str] = (),
    client: ChatClient | None = None,
    config: Any = None,
) -> Hypothetical:
    """Prose for one already-scored fact pattern.

    ``client`` is optional and the degradation is visible: with no model this
    returns the template with ``source="template"``. A silent fallback would let
    a run report "rendered hypotheticals" that are feature lists.

    A model failure is not fatal either — the fact pattern and its scores are
    the finding; the paragraph is presentation, and losing it should not lose
    the result.
    """

    template = template_prose(facts)
    if client is None:
        return Hypothetical(
            facts=dict(facts),
            prose=template,
            source="template",
            scores=dict(scores or {}),
            verdicts=tuple(verdicts),
            warnings=("no model configured; this is the deterministic template",),
        )

    from legal_research.llm.base import ChatMessage, DecodingPolicy

    payload = "\n".join(f"{name}: {value}" for name, value in sorted(facts.items()))
    try:
        prose = str(
            client.chat(
                [
                    ChatMessage("system", _SYSTEM),
                    ChatMessage("user", f"Coded fact pattern:\n{payload}"),
                ],
                config or DecodingPolicy.BALANCED.config,
            )
        ).strip()
    except Exception as exc:  # noqa: BLE001 - presentation must not cost the finding
        return Hypothetical(
            facts=dict(facts),
            prose=template,
            source="template",
            scores=dict(scores or {}),
            verdicts=tuple(verdicts),
            warnings=(f"model unavailable ({type(exc).__name__}); fell back to template",),
        )

    if not prose:
        return Hypothetical(
            facts=dict(facts),
            prose=template,
            source="template",
            scores=dict(scores or {}),
            verdicts=tuple(verdicts),
            warnings=("model returned nothing; fell back to template",),
        )

    warnings = _check(prose, facts)
    return Hypothetical(
        facts=dict(facts),
        prose=prose,
        source="model",
        scores=dict(scores or {}),
        verdicts=tuple(verdicts),
        warnings=tuple(warnings),
    )
