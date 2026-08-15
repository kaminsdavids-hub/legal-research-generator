"""Ideator (spec §2.2).

HOT decoding: generates candidate theses, framings and novel angles, and proposes
what would make the paper an original contribution. Feeds the idea board. Runs on
the Hermes role, backed by NVIDIA Nemotron 3 Nano, reasoning engine.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import HERMES
from ..models import Idea
from .base import Agent, AgentContext, AgentResult
from .legal_researcher import is_assertable


def _split_topic_and_claim(body: str) -> tuple[str, str]:
    """Split a ``TOPIC || CLAIM`` bullet.

    A generator that ignores the format still yields a usable topic, so the
    separator being absent is not an error -- it means no claim was offered.
    """
    if _CLAIM_SEPARATOR in body:
        topic, _, claim = body.partition(_CLAIM_SEPARATOR)
        return _unlabel(topic), _unlabel(claim)
    return _unlabel(body), ""


#: List markers a model uses interchangeably for one list. Measured over eight
#: live calls to the ideation role: three returned every idea under a prefix
#: other than "- ", and a fourth mixed "1." items in with "- " ones. Requiring
#: the one literal marker lost 38% of runs whole and part of others silently --
#: and invisibly, because "generated 0 candidate ideas" reads like a model with
#: nothing to say rather than a parser discarding 4,206 characters of
#: well-formed output.
#: ``\*(?!\*)`` so a single asterisk is a bullet and a double one is bold: the
#: naive class matched "**Reference:** Recent updates to EAR..." and turned a
#: citation note into a candidate idea.
_LIST_MARKER = re.compile(r"^\s*(?:[-•‣–—]|\*(?!\*)|\d+[.)]|#{1,6}|>)\s*")
_BOLD = re.compile(r"\*\*")
#: The model sometimes echoes the format's own placeholder as the topic
#: ("TOPIC || The publishing of open-source models is protected"), or emits a
#: bare "|| CLAIM: ..." with no topic at all. The claim is the real content in
#: both, so it stands in for the topic rather than the line being discarded.
_LABEL = re.compile(r"^\s*(?:TOPIC|CLAIM)\b\s*[:\-–]?\s*", re.IGNORECASE)
_PLACEHOLDERS = frozenset({"", "topic", "claim"})


def _unlabel(part: str) -> str:
    """Drop a leading label and any markup, keeping the sentence intact.

    Only *leading* punctuation goes. A claim is a declarative sentence and the
    full stop is part of it; stripping both ends turned "Weights are
    expression." into a fragment before it ever reached the assertability
    check.
    """

    return _LABEL.sub("", _BOLD.sub("", part)).lstrip(" .:-–—|").rstrip()


def recover_claims(client: Any, topics: list[str]) -> dict[int, str]:
    """Ask once for the claim each topic asserts. Returns ``{index: claim}``.

    A topic with no claim is not a small loss. The Writer cites a paragraph for
    the point it was drafted from, and with no claim that point is the topic —
    so the retriever is asked to support "Legal Challenges of Publishing
    Open-Source AI Models Under the EAR", and entailment against a title scores
    about zero by construction. That is REMEDIATION §22 one layer up: the fix
    there stopped titles reaching the scorer from the researcher, and this is
    the other road to the same place.

    One call for the whole list, not one per topic. Claims that come back
    unassertable are dropped rather than repaired again: a second failure is
    evidence about the model, and an unassertable claim is worse than none,
    because the empty case has a documented fallback and a bad claim silently
    becomes the thing a source is asked to support.
    """

    numbered = "\n".join(f"{n}. {topic}" for n, topic in enumerate(topics, start=1))
    raw = str(
        client.chat(
            [
                ChatMessage("system", _CLAIM_RECOVERY_SYSTEM),
                ChatMessage("user", numbered),
            ],
            DecodingPolicy.BALANCED.config,
        )
    )

    claims: dict[int, str] = {}
    for line in raw.splitlines():
        match = _NUMBERED_LINE.match(line.strip())
        if not match:
            continue
        index = int(match.group(1)) - 1
        claim = _unlabel(match.group(2))
        if 0 <= index < len(topics) and claim and is_assertable(claim):
            claims[index] = claim
    return claims


def parse_ideas(raw: str) -> list[tuple[str, str]]:
    """Every ``(topic, claim)`` pair in a reply, whatever list style it used.

    A line is an idea if it carries a list marker, or if it carries the ``||``
    separator the prompt asked for. Prose that does neither -- a preamble, a
    "Note:", a "**Reference:**" line -- is not an idea and is skipped.
    """

    ideas: list[tuple[str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        marked = bool(_LIST_MARKER.match(stripped))
        body = _BOLD.sub("", _LIST_MARKER.sub("", stripped)).strip()
        if not body or (not marked and _CLAIM_SEPARATOR not in body):
            continue

        text, claim = _split_topic_and_claim(body)
        if claim.lower() in _PLACEHOLDERS:
            claim = ""
        if text.lower() in _PLACEHOLDERS:
            text = claim
        # Re-checked after the substitution: a bare "- TOPIC || CLAIM" is the
        # model echoing the format, and promoting its placeholder claim to the
        # topic would file the word "CLAIM" as a candidate idea.
        if not text or text.lower() in _PLACEHOLDERS:
            continue
        ideas.append((text, claim))
    return ideas

_SYSTEM = (
    "TASK: ideate\n"
    "You are an idea generator for a legal-research paper. Produce several distinct, "
    "non-obvious candidate angles as a bullet list, one per line beginning with '- '. "
    "Each should suggest a genuinely original contribution.\n"
    "Write each line as 'TOPIC || CLAIM', where CLAIM is a single declarative "
    "sentence the paper would assert and a source could support or contradict. "
    "Do not write the claim as a question, a title, or an instruction to analyze "
    "something: 'Analyze how weights are treated' is not a claim, "
    "'Model weights are published information under the EAR' is."
)

#: Separates the topic from the claim on each bullet.
_CLAIM_SEPARATOR = "||"

_CLAIM_RECOVERY_SYSTEM = (
    "TASK: ideate\n"
    "For each numbered topic, write the single declarative sentence that a paper "
    "on it would ASSERT — something a court, statute or article could support or "
    "contradict.\n"
    "Answer with one line per topic, formatted 'N. <sentence>', and nothing else.\n"
    "Not a title, not a question, not an instruction. 'The economic implications "
    "of open weights' is a topic; 'Open-weight publication lowers the marginal "
    "cost of compliance below the licensing threshold' is a claim."
)

#: "3. <claim>" from the recovery pass, matched back by number.
_NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+)$")


class Ideator(Agent):
    name = "Ideator"
    expert_role = HERMES

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        seed: str = kwargs.get("seed") or ctx.blackboard.thesis or "the doctrine at issue"
        client = ctx.pool.get(self.expert_role)
        raw = client.chat(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", seed)],
            DecodingPolicy.HOT.config,
        )

        created: list[Idea] = []
        rejected = 0
        for text, claim in parse_ideas(raw):
            # An unassertable "claim" is worse than none: it would travel all
            # the way to the Verifier and be asked of a source, which is how a
            # title came to account for every verification in a live run
            # (REMEDIATION §24). Leaving it empty makes the caller fall back to
            # the thesis, which at least is a claim.
            if claim and not is_assertable(claim):
                rejected += 1
                claim = ""

            idea = ctx.blackboard.add_idea(
                text=text,
                claim=claim,
                angle="original-contribution",
                novelty_note="Proposed as a departure from the retrieved literature.",
            )
            created.append(idea)

        recovered = self._recover_missing_claims(ctx, client, created)

        with_claims = sum(1 for i in created if i.claim)
        # "generated 0 candidate ideas" was reported for both an empty reply and
        # a reply this parser could not read, and the pipeline then silently
        # substituted its deterministic fallback ideas. Those are different
        # failures with different fixes, and only one of them is the model's.
        unread = not created and bool(raw.strip())
        summary = (
            f"generated {len(created)} candidate ideas; "
            f"{with_claims} carry an assertable claim"
            + (f" ({recovered} recovered on a second pass)" if recovered else "")
            + (f" ({rejected} rejected as unassertable)" if rejected else "")
        )
        if unread:
            summary = (
                f"PARSE FAILURE: the model returned {len(raw)} characters but no line "
                f"matched the expected list format; 0 ideas were read from it"
            )
        elif not created:
            summary = "the model returned an empty reply; 0 candidate ideas"

        return AgentResult(
            agent=self.name,
            summary=summary,
            payload={
                "idea_ids": [i.id for i in created],
                "with_claims": with_claims,
                "rejected_claims": rejected,
                "recovered_claims": recovered,
                "unparsed_reply_chars": len(raw) if unread else 0,
            },
        )

    def _recover_missing_claims(
        self, ctx: AgentContext, client: Any, created: list[Idea]
    ) -> int:
        """Second pass for ideas that arrived as a topic with no claim.

        The generation prompt asks for ``TOPIC || CLAIM`` and the model complies
        most of the time; when it does not, every idea in the run reaches the
        retriever as a title. One targeted call is cheaper than losing the run's
        citations, and it is skipped entirely when nothing is missing.

        Failures here are silent by design in only one direction: a claim that
        does not come back, or comes back unassertable, leaves the idea exactly
        as it was, and the count of what was recovered is reported.
        """

        if not ctx.settings.ideator_claim_recovery_enabled:
            return 0
        missing = [(index, idea) for index, idea in enumerate(created) if not idea.claim]
        if not missing:
            return 0

        try:
            claims = recover_claims(client, [idea.text for _, idea in missing])
        except Exception:  # noqa: BLE001 - a failed recovery must not fail ideation
            return 0

        recovered = 0
        for position, (_, idea) in enumerate(missing):
            claim = claims.get(position, "")
            if claim:
                idea.claim = claim
                recovered += 1
        return recovered
