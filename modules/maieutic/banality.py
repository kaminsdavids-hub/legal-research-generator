"""The banality gate: is this contribution worth making?

Novelty asks whether a claim is new *to the manuscript*. Banality asks whether it
is new *to the field* — a proposition the sources already state is not a
contribution however absent it is from the paper. That is the whole distinction:
novelty compares a node against the graph, banality compares it against the
corpus.

Two limbs, and only one of them is allowed to stop anything.

**Vacuity blocks.** A claim whose content words are entirely hedges asserts
nothing at all. ``It may perhaps be that this could arguably be so`` is not a
weak claim, it is an absence of one, and that is decidable by counting rather
than by asking a model.

**Everything else reports.** Hedging and commonplaceness are advisory, for two
different reasons and both of them matter:

* Legal writing hedges as a professional norm. ``This may be the better view`` is
  a real scholarly claim, and a gate that blocked it would be actively harmful to
  the manuscript it is supposed to protect.
* Corpus similarity is method-dependent and uncalibrated. Without real
  embeddings this degrades to lexical overlap, which cannot see a paraphrase —
  the very thing a commonplace claim is.

**Only what claims to be the contribution is assessed.** A paper needs
commonplace material: background, setup, statements of existing doctrine. A
machine-proposed premise being unoriginal is correct, not a defect, and assessing
it would make the manuscript unwritable. The rule therefore runs over what the
author wrote and over THESIS/ORIGINAL nodes — see :func:`_is_a_contribution`,
whose second limb exists because a type-only rule made this gate unreachable in
the assembled loop.

This gate is therefore weaker than the other three, and deliberately so. It is a
signal to the author and to the Socratic engine about where the paper is thin,
not a veto over what may be written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from modules.dialectic import textnorm

from .graph import ArgumentGraph, GraphPatch, Node, Provenance
from .novelty import Embedder, LexicalEmbedder, Method, cosine
from .socratic import CLAIM_TYPES

#: Words that soften a claim without adding to it. Not stopwords: they are
#: topically empty but rhetorically loaded, and the difference between a hedged
#: claim and no claim is how much survives once they are removed.
HEDGES = frozenset(
    {
        "arguably", "perhaps", "may", "maybe", "might", "could", "possibly",
        "possible", "seems", "seem", "appears", "appear", "somewhat", "rather",
        "fairly", "relatively", "generally", "often", "sometimes", "tends",
        "tend", "suggests", "suggest", "presumably", "conceivably", "likely",
        "probably", "arguable", "potentially", "largely", "broadly", "roughly",
        "apparently", "seemingly", "ostensibly", "plausibly",
    }
)


def _is_a_contribution(node: Node) -> bool:
    """Whether this node claims to be the paper's contribution.

    Two ways to qualify, and the second was learned by wiring the loop up. A
    THESIS or ORIGINAL node claims to be a contribution by its type. But the
    adapter never produces either — an author's answer becomes a REPLY, an
    OBJECTION or a PREMISE depending on the gap it answers — so a type-only rule
    made this gate unreachable in the running system: a vacuous answer merged
    because nothing looked at it. What the author wrote is the contribution
    whatever structural role it plays, and a vacuous reply is as empty as a
    vacuous thesis.

    Machine-proposed premises, authorities and objections stay exempt. They are
    background and pressure, and both are supposed to be unoriginal.
    """
    return node.provenance is Provenance.HUMAN or node.type in CLAIM_TYPES


class Verdict(StrEnum):
    #: Says something, and says it in its own right.
    SUBSTANTIVE = "substantive"
    #: Hedged past the point of asserting anything. The only blocking verdict.
    VACUOUS = "vacuous"
    #: Carries content, but most of it is hedge.
    HEDGED = "hedged"
    #: The sources already say this.
    COMMONPLACE = "commonplace"


#: The blocking verdict, as a set rather than an `is` check, so adding one is a
#: deliberate edit to a table that the tests iterate.
BLOCKING = frozenset({Verdict.VACUOUS})


@dataclass(frozen=True)
class Source:
    """A passage from the literature, to compare a claim against."""

    label: str
    text: str


@dataclass
class Assessment:
    node_id: str
    verdict: Verdict
    detail: str = ""
    #: Share of the node's content words that are hedges.
    hedge_ratio: float = 0.0
    #: Similarity to the nearest corpus passage, when the corpus limb ran.
    similarity: float = 0.0
    nearest_source: str = ""
    #: How the corpus comparison was made, or empty when it did not run. A
    #: lexical run cannot see a paraphrase and must not read as a semantic one.
    method: Method | None = None
    section: str = ""

    @property
    def blocking(self) -> bool:
        return self.verdict in BLOCKING


@dataclass
class PatchBanality:
    assessments: list[Assessment] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Only vacuity stops a merge. Everything else is a signal."""
        return not any(a.blocking for a in self.assessments)

    @property
    def advisory(self) -> list[Assessment]:
        return [
            a for a in self.assessments
            if a.verdict is not Verdict.SUBSTANTIVE and not a.blocking
        ]

    @property
    def substantive(self) -> list[Assessment]:
        return [a for a in self.assessments if a.verdict is Verdict.SUBSTANTIVE]

    def barren_sections(self) -> frozenset[str]:
        """Sections where this patch landed no substantive claim.

        Feeds `SocraticEngine.ask(avoid_sections=...)`: somewhere the author is
        producing only hedge and received wisdom is somewhere to stop asking.
        Per-patch and therefore noisy — a caller should accumulate across cycles
        rather than act on one.
        """
        touched = {a.section for a in self.assessments if a.section}
        alive = {a.section for a in self.substantive if a.section}
        return frozenset(touched - alive)


class BanalityGate:
    #: Above this similarity to a source passage, the field already says it.
    #: PROVISIONAL and uncalibrated, exactly as the novelty thresholds are: no
    #: corpus of accepted-versus-rejected manuscript claims exists to set it
    #: against. See OBSERVABLES D6.
    COMMONPLACE_AT = 0.88

    #: Above this share of hedge among a claim's content words, it is mostly
    #: throat-clearing. Also PROVISIONAL.
    HEDGED_AT = 0.34

    def __init__(
        self,
        sources: list[Source] | None = None,
        embedder: Embedder | None = None,
        *,
        commonplace_at: float | None = None,
        hedged_at: float | None = None,
    ) -> None:
        self.sources = sources or []
        self.embedder = embedder or LexicalEmbedder()
        self.commonplace_at = (
            self.COMMONPLACE_AT if commonplace_at is None else commonplace_at
        )
        self.hedged_at = self.HEDGED_AT if hedged_at is None else hedged_at

    def assess(self, node: Node) -> Assessment:
        if not _is_a_contribution(node):
            # Background, setup and statements of existing doctrine are supposed
            # to be unoriginal. Judging them would make the paper unwritable.
            return Assessment(
                node_id=node.id,
                verdict=Verdict.SUBSTANTIVE,
                detail="no banality rule applies to this node",
                section=node.section,
            )

        content = textnorm.content_words(node.text)
        substance = content - HEDGES
        ratio = (len(content) - len(substance)) / len(content) if content else 0.0

        if not substance:
            return Assessment(
                node_id=node.id,
                verdict=Verdict.VACUOUS,
                detail="every content word is a hedge; this asserts nothing",
                hedge_ratio=ratio,
                section=node.section,
            )

        commonplace = self._against_the_corpus(node)
        if commonplace is not None:
            return commonplace

        if ratio >= self.hedged_at:
            return Assessment(
                node_id=node.id,
                verdict=Verdict.HEDGED,
                detail=f"{ratio:.0%} of the content words are hedges",
                hedge_ratio=ratio,
                section=node.section,
            )

        return Assessment(
            node_id=node.id,
            verdict=Verdict.SUBSTANTIVE,
            detail="asserts something the sources do not already state",
            hedge_ratio=ratio,
            section=node.section,
        )

    def _against_the_corpus(self, node: Node) -> Assessment | None:
        """Compare a claim to the literature, if there is any to compare it to."""
        if not self.sources:
            # No corpus means the question was not asked. Saying so beats
            # reporting a claim as original because nothing was checked.
            return None

        vectors = self.embedder.embed([node.text, *(s.text for s in self.sources)])
        head, rest = vectors[0], vectors[1:]
        scores = [cosine(head, v) for v in rest]
        best = max(range(len(scores)), key=lambda i: scores[i])
        similarity, nearest = scores[best], self.sources[best]

        if similarity < self.commonplace_at:
            return None
        return Assessment(
            node_id=node.id,
            verdict=Verdict.COMMONPLACE,
            detail=f"the sources already state this ({nearest.label})",
            similarity=similarity,
            nearest_source=nearest.label,
            method=self.embedder.method,
            section=node.section,
        )

    def assess_patch(self, patch: GraphPatch, graph: ArgumentGraph) -> PatchBanality:
        result = PatchBanality()
        for node in patch.nodes:
            if node.id in graph.nodes:
                continue
            result.assessments.append(self.assess(node))
        return result
