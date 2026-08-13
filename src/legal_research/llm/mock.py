"""Deterministic mock LLM.

The mock produces stable, input-dependent text with *varied* sentence cadence so
that the full pipeline (and the anti-"AI voice" checks) can run offline with no GPU
and no network. Output is a pure function of the input messages, which keeps tests
and golden files reproducible.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterator

from .base import ChatMessage, GenerationConfig, LLMClient

_TASK_RE = re.compile(r"^TASK:\s*(?P<task>[a-z0-9_.-]+)", re.IGNORECASE | re.MULTILINE)

# Short, medium and long sentence stems produce human-like cadence variance. The
# writer draws from these so generated prose passes the uniform-length lint.
_OPENERS = [
    "Consider the doctrine at its narrowest.",
    "The question here is not academic.",
    "Courts have long wrestled with this seam between rules and their reasons.",
    "Begin with the text, then follow where it forces us.",
    "There is a quieter argument hiding beneath the obvious one.",
]
_BODIES = [
    "A careful reading of the controlling authority suggests the rule bends only where its underlying purpose runs out, and not a step further.",
    "That distinction matters.",
    "When the statutory language and the equitable instinct pull apart, the better course is to name the tension rather than paper over it.",
    "The finance mechanics are unglamorous but decisive: mispriced collateral does not merely shift risk, it relocates the incentive to monitor it.",
    "Precedent gestures in two directions at once.",
    "What follows is an attempt to hold both truths in view without collapsing either.",
]
_CLOSERS = [
    "The payoff is a rule that is administrable and honest about its limits.",
    "So framed, the doctrine earns its keep.",
    "That, at least, is the wager this Article makes.",
    "The remainder of this Part defends the claim in detail.",
]


def _seed_from(messages: list[ChatMessage], salt: str) -> int:
    digest = hashlib.sha256(salt.encode())
    for m in messages:
        digest.update(m.role.encode())
        digest.update(b"\x00")
        digest.update(m.content.encode())
        digest.update(b"\x01")
    return int.from_bytes(digest.digest()[:8], "big")


def _detect_task(messages: list[ChatMessage]) -> str:
    for m in messages:
        if m.role == "system":
            match = _TASK_RE.search(m.content)
            if match:
                return match.group("task").lower()
    return "generic"


def _last_user(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", text.lower())
    stop = {
        "this", "that", "with", "from", "into", "your", "will", "shall",
        "should", "would", "which", "there", "their", "about", "these",
        "those", "have", "been", "were", "what", "when", "where", "paper",
        "section", "please", "write", "draft",
    }
    seen: list[str] = []
    for w in words:
        if w not in stop and w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


class MockLLM(LLMClient):
    """A deterministic stand-in for a locally served expert model."""

    def __init__(self, name: str) -> None:
        self.name = name

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> str:
        task = _detect_task(messages)
        seed = _seed_from(messages, salt=f"{self.name}:{task}")
        rng = random.Random(seed)
        handler = getattr(self, f"_task_{task}", None)
        if handler is not None:
            return str(handler(messages, rng))
        return self._task_generic(messages, rng)

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> Iterator[str]:
        text = self.chat(messages, config)
        yield from re.findall(r"\S+\s*", text)

    def warm_up(self, prompt: str = ".") -> None:
        """Mock models have no external process to warm."""

        return

    # ---- task-specific responders -------------------------------------------------

    def _task_generic(self, messages: list[ChatMessage], rng: random.Random) -> str:
        kws = _keywords(_last_user(messages)) or ["the doctrine", "the record"]
        subject = kws[0]
        return (
            f"On {subject}, the analysis proceeds in three moves. "
            f"First, it fixes the governing standard. "
            f"Second, it applies that standard to the facts with attention to {kws[-1]}. "
            f"Third, it confronts the strongest objection and answers it."
        )

    def _task_route(self, messages: list[ChatMessage], rng: random.Random) -> str:
        # The rule-based router in router.py is authoritative; this is only used if
        # the router asks the model to break a tie. Return a bare label.
        text = _last_user(messages).lower()
        if any(k in text for k in ("basel", "capital", "collateral", "bank", "swap")):
            return "finance"
        if any(k in text for k in ("holding", "statute", "precedent", "circuit", "doctrine")):
            return "legal"
        if any(k in text for k in ("brainstorm", "idea", "angle", "novel")):
            return "brainstorm"
        return "synthesis"

    def _task_interview(self, messages: list[ChatMessage], rng: random.Random) -> str:
        kws = _keywords(_last_user(messages)) or ["your thesis"]
        focus = kws[0]
        prompts = [
            f"What would a skeptical judge say is the weakest link between {focus} and the remedy you want?",
            f"You gesture at {focus}; which single case, if it came out the other way, would sink the argument?",
            f"Is {focus} a claim about what the law is, or what it ought to be? The paper needs to pick.",
            f"Where does {focus} stop? Name the fact pattern that falls outside your rule.",
        ]
        return prompts[rng.randrange(len(prompts))]

    def _task_ideate(self, messages: list[ChatMessage], rng: random.Random) -> str:
        kws = _keywords(_last_user(messages)) or ["the doctrine"]
        base = kws[0]
        angles = [
            f"Reframe {base} as a monitoring-cost problem rather than a fairness problem.",
            f"Argue that the circuit split over {base} is really a disagreement about remedies, not rights.",
            f"Import a finance concept — priority of claims — to explain why {base} produces perverse incentives.",
            f"Show that the leading case on {base} rests on a factual assumption that no longer holds.",
        ]
        rng.shuffle(angles)
        return "\n".join(f"- {a}" for a in angles)

    def _task_write(self, messages: list[ChatMessage], rng: random.Random) -> str:
        # Compose prose that always contains at least one very short and one very long
        # sentence, so the output reliably clears the uniform-cadence lint.
        short = "That distinction matters."
        long_ = _BODIES[0]  # the 20+ word sentence about the controlling authority
        opener = _OPENERS[rng.randrange(len(_OPENERS))]
        closer = _CLOSERS[rng.randrange(len(_CLOSERS))]
        extras = [b for b in _BODIES if b not in (short, long_)]
        rng.shuffle(extras)
        body = [short, long_, *extras[: rng.randint(1, 2)]]
        rng.shuffle(body)
        return " ".join([opener, *body, closer])

    def _task_edit(self, messages: list[ChatMessage], rng: random.Random) -> str:
        return _last_user(messages).strip()
