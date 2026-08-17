"""The novelty gate: the wall that stops the loop restating the paper.

The graph already refuses a patch that adds no nodes. That catches literal
re-insertion; it does not catch a node that says what the paper already says in
different words. This gate does.

Three bands, because one threshold cannot separate paraphrase from contribution:

* **above the hard threshold** — a restatement. Rejected outright.
* **inside the soft band** — close enough to be suspicious. An entailment critic
  decides: does this assert anything the graph does not already entail? A node
  the graph entails is not a contribution, however new its wording.
* **below the band** — novel on its face.

Two things learned the hard way elsewhere in this repository are built in rather
than bolted on:

**A degraded critic must be visible.** Without real embeddings this gate falls
back to lexical overlap, which cannot see a paraphrase at all — precisely the
failure it exists to catch. Every verdict therefore records the method that
produced it, and :meth:`NoveltyGate.assess` will not silently pretend a lexical
run is a semantic one.

**The critic must not read text the machinery wrote.** A gate that reads its own
system's output validates itself and reports a pass forever; that happened three
times in the dialectic work (REMEDIATION §5, §11.5, §11.11a). This gate reads
node text only, which carries provenance, and never notes or annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from modules.dialectic import textnorm

from .graph import ArgumentGraph, GraphPatch, Node


class Verdict(StrEnum):
    NOVEL = "novel"
    #: Too close to an existing node to be anything but a restatement.
    RESTATEMENT = "restatement"
    #: The graph already entails it; new words, no new claim.
    ENTAILED = "entailed"


class Method(StrEnum):
    """How a verdict was reached. Recorded so a degraded run is legible."""

    SEMANTIC = "semantic"
    #: Lexical overlap only — cannot detect paraphrase. A gate running this way
    #: is weaker than it looks and says so.
    LEXICAL = "lexical"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def method(self) -> Method: ...


class LexicalEmbedder:
    """Offline fallback: a bag-of-content-words vector.

    Deterministic and dependency-free, so tests and CI can exercise the gate's
    logic. It is **not** a semantic embedder: "the rule applies" and "the
    doctrine governs" share no vocabulary and score zero, which is exactly the
    paraphrase a novelty gate must catch. Verdicts from it are labelled
    ``Method.LEXICAL`` so nobody reads them as more than they are.
    """

    @property
    def method(self) -> Method:
        return Method.LEXICAL

    def embed(self, texts: list[str]) -> list[list[float]]:
        vocab: dict[str, int] = {}
        bags = []
        for text in texts:
            bag = textnorm.content_words(text, stemmed=True)
            bags.append(bag)
            for word in bag:
                vocab.setdefault(word, len(vocab))
        return [[1.0 if w in bag else 0.0 for w in vocab] for bag in bags]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EntailmentCritic(Protocol):
    """Decides whether existing text already entails a candidate claim."""

    def relate(self, premise: str, hypothesis: str) -> tuple[str, str]: ...


@dataclass
class NoveltyVerdict:
    node_id: str
    verdict: Verdict
    similarity: float
    method: Method
    nearest_id: str = ""
    nearest_text: str = ""
    reason: str = ""
    #: Which path the entailment critic used, when one ran. A downgrade from a
    #: model to a heuristic changes what the verdict means.
    critic_source: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict is Verdict.NOVEL


@dataclass
class PatchNovelty:
    verdicts: list[NoveltyVerdict] = field(default_factory=list)

    @property
    def accepted(self) -> list[NoveltyVerdict]:
        return [v for v in self.verdicts if v.accepted]

    @property
    def passed(self) -> bool:
        """A patch survives only if something in it is genuinely new."""
        return bool(self.accepted)

    @property
    def delta(self) -> float:
        """Share of proposed nodes that were genuinely new.

        Tracked per cycle: a shrinking delta means the vein is mined out and the
        engine should move to another section rather than keep asking here.
        """
        if not self.verdicts:
            return 0.0
        return len(self.accepted) / len(self.verdicts)


class NoveltyGate:
    """Reject restatement; let contribution through."""

    #: Above this, a candidate is a restatement whatever a critic says.
    #: PROVISIONAL. The dialectic module's independence thresholds were set by
    #: scoring two real populations and putting the cut in the gap between them
    #: (REMEDIATION §10.3); these have had no such calibration yet, because no
    #: corpus of accepted-versus-rejected manuscript nodes exists. Calibrate
    #: before trusting a merge decision to them.
    HARD = 0.92

    #: Inside this band the entailment critic decides.
    SOFT_LOW = 0.70

    def __init__(
        self,
        embedder: Embedder,
        critic: EntailmentCritic | None = None,
        *,
        hard: float | None = None,
        soft_low: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.critic = critic
        self.hard = self.HARD if hard is None else hard
        self.soft_low = self.SOFT_LOW if soft_low is None else soft_low
        if not 0.0 <= self.soft_low <= self.hard <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= soft_low <= hard <= 1")

    def assess(self, candidate: Node, existing: list[Node]) -> NoveltyVerdict:
        method = self.embedder.method
        if not existing:
            return NoveltyVerdict(
                node_id=candidate.id,
                verdict=Verdict.NOVEL,
                similarity=0.0,
                method=method,
                reason="nothing to restate: the graph is empty",
            )

        texts = [candidate.text, *(n.text for n in existing)]
        vectors = self.embedder.embed(texts)
        head, rest = vectors[0], vectors[1:]
        scores = [cosine(head, v) for v in rest]
        best = max(range(len(scores)), key=lambda i: scores[i])
        similarity, nearest = scores[best], existing[best]

        if similarity >= self.hard:
            return NoveltyVerdict(
                node_id=candidate.id,
                verdict=Verdict.RESTATEMENT,
                similarity=similarity,
                method=method,
                nearest_id=nearest.id,
                nearest_text=nearest.text,
                reason=f"similarity {similarity:.2f} >= {self.hard} to an existing node",
            )

        if similarity >= self.soft_low:
            return self._ask_critic(candidate, nearest, similarity, method)

        return NoveltyVerdict(
            node_id=candidate.id,
            verdict=Verdict.NOVEL,
            similarity=similarity,
            method=method,
            nearest_id=nearest.id,
            reason=f"similarity {similarity:.2f} < {self.soft_low}",
        )

    def _ask_critic(
        self, candidate: Node, nearest: Node, similarity: float, method: Method
    ) -> NoveltyVerdict:
        """In the soft band, wording is not the question — entailment is."""
        if self.critic is None:
            # No critic: the honest answer is that this was not adjudicated.
            # Admitting it beats a confident guess in either direction.
            return NoveltyVerdict(
                node_id=candidate.id,
                verdict=Verdict.NOVEL,
                similarity=similarity,
                method=method,
                nearest_id=nearest.id,
                nearest_text=nearest.text,
                reason=(
                    f"similarity {similarity:.2f} is in the soft band but no "
                    "entailment critic is configured; admitted unadjudicated"
                ),
            )

        label, source = self.critic.relate(nearest.text, candidate.text)
        if label == "entailment":
            return NoveltyVerdict(
                node_id=candidate.id,
                verdict=Verdict.ENTAILED,
                similarity=similarity,
                method=method,
                nearest_id=nearest.id,
                nearest_text=nearest.text,
                reason="the graph already entails this; new wording, no new claim",
                critic_source=source,
            )
        return NoveltyVerdict(
            node_id=candidate.id,
            verdict=Verdict.NOVEL,
            similarity=similarity,
            method=method,
            nearest_id=nearest.id,
            reason=f"close to an existing node but not entailed by it ({label})",
            critic_source=source,
        )

    def assess_patch(self, patch: GraphPatch, graph: ArgumentGraph) -> PatchNovelty:
        """Assess every proposed node against the graph and against each other.

        Candidates are compared to their accepted siblings as well as to the
        graph, so a patch cannot smuggle a restatement past by pairing it with
        something genuinely new.
        """
        existing = list(graph.nodes.values())
        result = PatchNovelty()
        for node in patch.nodes:
            if node.id in graph.nodes:
                continue
            verdict = self.assess(node, existing)
            result.verdicts.append(verdict)
            if verdict.accepted:
                existing = [*existing, node]
        return result
