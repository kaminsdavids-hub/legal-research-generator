"""The Socratic engine: find the holes in the argument, and ask about them.

The loop's premise is that the machine does not supply content. It supplies
*pressure* — questions the author has to answer — and the answers are the
manuscript's original material. So this module produces questions and nothing
else. It never proposes a node.

**Gap analysis reads structure, not prose.** Every gap here is a property of the
graph's shape: an objection with no reply, a claim nothing attacks, a node
connected to nothing, a `DEPENDS_ON` cycle. This is deliberate and it is the
module's main defence. A critic that reads text to judge whether an argument is
complete can be satisfied by confident writing, which is how three self-
satisfying gates got into the dialectic work (REMEDIATION §5, §11.5, §11.11a). A
missing reply edge cannot be written around.

The engine does read node text, to quote a claim back to the author when phrasing
a question. That is not the same failure: a question is not a verdict, it asserts
nothing, and the author answers it. What would be that failure is a *gate* whose
pass depends on machine-authored prose, and there is none here.

**Asking is not answering.** A gap that has been asked about is skipped when
choosing the next question, so the engine moves on rather than nagging — but it
stays in the analysis and in :meth:`AskedLog.outstanding`. A hole does not stop
being a hole because someone mentioned it once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .graph import ArgumentGraph, EdgeType, Node, NodeType

#: Node types that make a claim, as opposed to carrying the argument's plumbing.
#: These are what it means for the paper to be contestable.
CLAIM_TYPES = (NodeType.THESIS, NodeType.ORIGINAL)

#: Edge types that count as holding a node up. Mirrors the grounding gate; an
#: ATTACKS edge is engagement, not support.
SUPPORTING = (EdgeType.SUPPORTS, EdgeType.IMPLIES, EdgeType.QUALIFIES)


class GapKind(StrEnum):
    """A structural hole in the argument.

    Ordered here as they are prioritised: a self-grounding argument is a defect
    no amount of further material fixes, an objection left hanging is the thing
    that most damages a paper, and an unverified citation is last because it is
    the one gap the machine can sometimes close without asking anyone.
    """

    #: The argument grounds itself: a DEPENDS_ON cycle.
    SELF_GROUNDING = "self_grounding"
    #: An objection was stated and never replied to.
    UNANSWERED_ATTACK = "unanswered_attack"
    #: An ORIGINAL position with nothing holding it up.
    UNSUPPORTED_CLAIM = "unsupported_claim"
    #: A claim nothing in the paper attacks. Untested, not necessarily wrong.
    UNCONTESTED_CLAIM = "uncontested_claim"
    #: A node with no edges at all: present in the manuscript, part of no argument.
    ORPHANED_NODE = "orphaned_node"
    #: An AUTHORITY node whose citation has not resolved.
    UNVERIFIED_AUTHORITY = "unverified_authority"


#: Priority order. Declared once, so the policy and its test read the same list.
PRIORITY: tuple[GapKind, ...] = (
    GapKind.SELF_GROUNDING,
    GapKind.UNANSWERED_ATTACK,
    GapKind.UNSUPPORTED_CLAIM,
    GapKind.UNCONTESTED_CLAIM,
    GapKind.ORPHANED_NODE,
    GapKind.UNVERIFIED_AUTHORITY,
)


@dataclass(frozen=True)
class Gap:
    kind: GapKind
    #: The nodes the gap concerns. Ordered as the gap describes them.
    node_ids: tuple[str, ...]
    detail: str = ""
    section: str = ""

    @property
    def key(self) -> str:
        """Stable identity across cycles, so 'already asked' means something.

        Node ids are stable for a node's lifetime, so the same hole in the same
        place produces the same key on every pass.
        """
        return f"{self.kind.value}:{','.join(sorted(self.node_ids))}"


def analyse(graph: ArgumentGraph) -> list[Gap]:
    """Every structural gap in the graph, in priority order."""
    gaps: list[Gap] = []
    gaps.extend(_self_grounding(graph))
    gaps.extend(_unanswered_attacks(graph))
    gaps.extend(_unsupported_claims(graph))
    gaps.extend(_uncontested_claims(graph))
    gaps.extend(_orphans(graph))
    gaps.extend(_unverified_authorities(graph))
    return gaps


def _self_grounding(graph: ArgumentGraph) -> list[Gap]:
    cycle = graph.depends_on_cycle()
    if not cycle:
        return []
    return [
        Gap(
            kind=GapKind.SELF_GROUNDING,
            node_ids=tuple(cycle),
            detail="these nodes depend on each other in a cycle",
            section=_section_of(graph, cycle[0]),
        )
    ]


def _unanswered_attacks(graph: ArgumentGraph) -> list[Gap]:
    gaps = []
    for edge in graph.unanswered_attacks():
        gaps.append(
            Gap(
                kind=GapKind.UNANSWERED_ATTACK,
                node_ids=(edge.source, edge.target),
                detail="this objection has no reply",
                section=_section_of(graph, edge.target),
            )
        )
    return gaps


def _unsupported_claims(graph: ArgumentGraph) -> list[Gap]:
    """ORIGINAL nodes with nothing under them.

    The grounding gate refuses to merge one, so a well-behaved loop never
    produces this. It is reachable through the paths that bypass the gates: a
    graph loaded from disk, or a direct ``apply``. Analysis that only covers the
    happy path is not analysis.
    """
    gaps = []
    for node in graph.of_type(NodeType.ORIGINAL):
        if not graph.incoming(node.id, *SUPPORTING):
            gaps.append(
                Gap(
                    kind=GapKind.UNSUPPORTED_CLAIM,
                    node_ids=(node.id,),
                    detail="this position is asserted with nothing holding it up",
                    section=node.section,
                )
            )
    return gaps


def _uncontested_claims(graph: ArgumentGraph) -> list[Gap]:
    gaps = []
    for node in graph.of_type(*CLAIM_TYPES):
        if not graph.incoming(node.id, EdgeType.ATTACKS):
            gaps.append(
                Gap(
                    kind=GapKind.UNCONTESTED_CLAIM,
                    node_ids=(node.id,),
                    detail="nothing in the paper puts pressure on this claim",
                    section=node.section,
                )
            )
    return gaps


def _orphans(graph: ArgumentGraph) -> list[Gap]:
    """Nodes attached to nothing.

    A single-node graph is not orphaned, it is a paper that has just started;
    there is nothing yet for its thesis to connect to.
    """
    if len(graph.nodes) < 2:
        return []
    return [
        Gap(
            kind=GapKind.ORPHANED_NODE,
            node_ids=(node.id,),
            detail="this node is in the manuscript but part of no argument",
            section=node.section,
        )
        for node in graph.nodes.values()
        if not graph.is_connected(node.id)
    ]


def _unverified_authorities(graph: ArgumentGraph) -> list[Gap]:
    return [
        Gap(
            kind=GapKind.UNVERIFIED_AUTHORITY,
            node_ids=(node.id,),
            detail="this authority's citation has not resolved",
            section=node.section,
        )
        for node in graph.of_type(NodeType.AUTHORITY)
        if not node.verified
    ]


def _section_of(graph: ArgumentGraph, node_id: str) -> str:
    node = graph.nodes.get(node_id)
    return node.section if node else ""


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
class Phrasing(StrEnum):
    """How a question was worded. Recorded so a degraded run is legible."""

    TEMPLATE = "template"
    MODEL = "model"


@dataclass(frozen=True)
class Question:
    gap: Gap
    text: str
    phrasing: Phrasing = Phrasing.TEMPLATE

    @property
    def key(self) -> str:
        return self.gap.key


class QuestionWriter(Protocol):
    """Optionally rewords a templated question. It may not change the subject."""

    def phrase(self, gap: Gap, default: str, quotes: list[str]) -> str: ...


_TEMPLATES: dict[GapKind, str] = {
    GapKind.SELF_GROUNDING: (
        "Your argument currently grounds itself: {quote} depends on a chain that "
        "leads back to it. Which of these is prior — what does the reader have to "
        "accept first?"
    ),
    GapKind.UNANSWERED_ATTACK: (
        "You raise this objection and leave it standing: {quote} What is your "
        "answer to it — and if you do not have one, is the claim it attacks still "
        "one you want to make?"
    ),
    GapKind.UNSUPPORTED_CLAIM: (
        "You assert this without anything under it: {quote} Why should a sceptical "
        "reader accept it?"
    ),
    GapKind.UNCONTESTED_CLAIM: (
        "Nothing in the paper pushes back on this: {quote} What is the strongest "
        "objection someone who disagrees with you would make?"
    ),
    GapKind.ORPHANED_NODE: (
        "This sits in the manuscript attached to nothing: {quote} What work is it "
        "doing — what does it support, or what supports it?"
    ),
    GapKind.UNVERIFIED_AUTHORITY: (
        "This authority has not been verified: {quote} Is the citation right, and "
        "does the source actually say this?"
    ),
}


def _excerpt(node: Node | None, limit: int = 160) -> str:
    if node is None:
        return "(missing node)"
    text = " ".join(node.text.split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


@dataclass
class AskedLog:
    """What has already been asked.

    Keyed on :attr:`Gap.key`, not on question text, so rewording a question does
    not make it a new one.
    """

    keys: set[str] = field(default_factory=set)

    def record(self, question: Question) -> None:
        self.keys.add(question.key)

    def has_asked(self, gap: Gap) -> bool:
        return gap.key in self.keys

    def outstanding(self, graph: ArgumentGraph) -> list[Gap]:
        """Gaps that were asked about and are still open.

        The policy stops offering these so the loop moves on. They must stay
        visible: a paper's unanswered objections are exactly what an honest one
        declares, and a hole that vanishes from the report because it was
        mentioned once is the report lying.
        """
        return [g for g in analyse(graph) if self.has_asked(g)]


class SocraticEngine:
    """Chooses what to ask next, and words it."""

    def __init__(
        self,
        writer: QuestionWriter | None = None,
        rank: Callable[[GapKind], float] | None = None,
    ) -> None:
        self.writer = writer
        # How gap kinds are ordered. Defaults to the declared PRIORITY; a
        # learned policy supplies its own (see modules.maieutic.learn). Injected
        # rather than imported so this module stays free of the learning code.
        self.rank = rank or (lambda kind: float(PRIORITY.index(kind)))

    def ask(
        self,
        graph: ArgumentGraph,
        asked: AskedLog | None = None,
        *,
        limit: int = 1,
        avoid_sections: frozenset[str] = frozenset(),
    ) -> list[Question]:
        """The next *limit* questions, highest-priority gap first.

        Returns fewer than *limit*, or none at all, when the graph has no
        unasked gaps left. Returning nothing is a real answer — the alternative
        is manufacturing a question to fill a quota, which spends the author's
        attention on whatever the engine could think of rather than on a hole.
        """
        asked = asked or AskedLog()
        candidates = [g for g in analyse(graph) if not asked.has_asked(g)]

        # A section whose novelty delta has collapsed is mined out: further
        # questions there return restatement. Deprioritised rather than dropped,
        # since a self-grounding cycle in a mined-out section still matters more
        # than an unverified citation somewhere fresh.
        candidates.sort(
            key=lambda g: (g.section in avoid_sections, self.rank(g.kind))
        )
        return [self.phrase(g, graph) for g in candidates[:limit]]

    def phrase(self, gap: Gap, graph: ArgumentGraph) -> Question:
        quotes = [_excerpt(graph.nodes.get(nid)) for nid in gap.node_ids]
        default = _TEMPLATES[gap.kind].format(quote=f"“{quotes[0]}”")
        if self.writer is None:
            return Question(gap=gap, text=default, phrasing=Phrasing.TEMPLATE)

        try:
            worded = self.writer.phrase(gap, default, quotes).strip()
        except Exception:  # noqa: BLE001 - a broken writer must not cost the question
            return Question(gap=gap, text=default, phrasing=Phrasing.TEMPLATE)

        if not self._is_a_question(worded):
            # A "question" that asserts something is the machine putting content
            # into the author's channel, which is the one thing this module must
            # not do. Fall back rather than pass it on.
            return Question(gap=gap, text=default, phrasing=Phrasing.TEMPLATE)
        return Question(gap=gap, text=worded, phrasing=Phrasing.MODEL)

    @staticmethod
    def _is_a_question(text: str) -> bool:
        return bool(text) and text.rstrip().endswith("?")
