"""Turning an author's answer plus a dialectic exchange into a graph patch.

This is where the loop closes: a Socratic question produced a hole, the author
answered it, the dialectic module argued both sides of the answer, and the result
has to become nodes and edges.

Three rules govern the translation, and each of them is a refusal.

**The author's answer is captured verbatim, and it is the only HUMAN node.**
Everything the machine contributed carries ``Provenance.DIALECTIC``, so the
scholarly-integrity record survives rendering. ``Node.propose`` refuses HUMAN
provenance outright, so this is a guarantee rather than a convention.

**The synthesis does not enter the manuscript.** Thesis and antithesis material
is *pressure* — support the author can accept and objections they must answer —
but the synthesis is the machine writing the paper's conclusion, which is the one
thing this system exists not to do. It is returned alongside the patch as
material for the next question, never merged.

**Authority is not created here.** Only a slot the dialectic module actually
verified becomes an ``AUTHORITY`` node, and even then it arrives with
``verified=False``: the grounding gate is the only thing that may mark a node
verified, and it asks a question CourtListener does not — whether the source
supports *this claim*, not merely whether the citation resolves. A retrieved but
unconfirmed candidate becomes a plain premise carrying no citation, because a
candidate is not authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.dialectic.channel import CitationChannel
from modules.dialectic.models import (
    NOT_OPERATIVE,
    CitationSlot,
    DialecticTurn,
    SlotStatus,
)

from .graph import Edge, EdgeType, GraphPatch, Node, NodeType
from .socratic import GapKind, Question


class AnswerRefused(Exception):
    """Raised when there is no answer to build a patch from."""


@dataclass(frozen=True)
class _Role:
    """What the author's answer becomes, given the gap that prompted it."""

    node_type: NodeType
    edge_type: EdgeType


#: How an answer attaches, by the gap it answers. Complete over ``GapKind`` by
#: construction — a gap the engine can ask about but cannot absorb an answer to
#: would be a question that goes nowhere.
_ROLES: dict[GapKind, _Role] = {
    # The reply attaches to the objection, not to what the objection attacked.
    # `unanswered_attacks` looks for a REPLY pointing at the objection, so
    # anything else leaves the attack reading as open forever.
    GapKind.UNANSWERED_ATTACK: _Role(NodeType.REPLY, EdgeType.SUPPORTS),
    # The author was asked for the strongest objection to their own claim.
    GapKind.UNCONTESTED_CLAIM: _Role(NodeType.OBJECTION, EdgeType.ATTACKS),
    GapKind.UNSUPPORTED_CLAIM: _Role(NodeType.PREMISE, EdgeType.SUPPORTS),
    GapKind.ORPHANED_NODE: _Role(NodeType.PREMISE, EdgeType.SUPPORTS),
    # These two are answered with reasons, but the structural fix — removing a
    # DEPENDS_ON edge, re-running retrieval — is an edit, and a patch only adds.
    # The gap therefore stays open after the answer, which is correct: it is
    # still there. See `Adaptation.unresolved`.
    GapKind.SELF_GROUNDING: _Role(NodeType.PREMISE, EdgeType.SUPPORTS),
    GapKind.UNVERIFIED_AUTHORITY: _Role(NodeType.PREMISE, EdgeType.SUPPORTS),
}

#: Gaps an added patch cannot close, whatever the answer says.
_NEEDS_AN_EDIT = frozenset({GapKind.SELF_GROUNDING, GapKind.UNVERIFIED_AUTHORITY})


@dataclass
class Adaptation:
    patch: GraphPatch
    #: The answer node's id, so the caller can follow what the author added.
    answer_id: str
    #: Propositions refused at the boundary, each with the reason. Never silently
    #: dropped: a proposition that vanished without a record is indistinguishable
    #: from one the model never produced.
    refused: list[str] = field(default_factory=list)
    #: The machine's synthesis. Deliberately not merged — see the module
    #: docstring. Carried so the next cycle can use it to choose a question.
    synthesis: str = ""
    #: True when the answer cannot close the gap it responded to, because doing
    #: so needs an edit rather than an insertion.
    unresolved: bool = False


def adapt(
    question: Question,
    answer: str,
    turn: DialecticTurn | None = None,
    *,
    section: str = "",
) -> Adaptation:
    """Build a patch from an author's answer and the exchange it provoked.

    *turn* is optional: an answer with no dialectic run behind it still enters
    the manuscript. The machine's contribution is pressure, and a patch carrying
    only the author's own words is a smaller patch, not an invalid one.
    """
    if not answer.strip():
        raise AnswerRefused(
            "an empty answer is the author declining the question, not an "
            "insertion; record it as unanswered rather than merging nothing"
        )

    gap = question.gap
    role = _ROLES[gap.kind]
    node = Node.from_human(
        role.node_type, answer.strip(), section=section or gap.section
    )

    nodes = [node]
    edges = []
    if gap.node_ids:
        edges.append(Edge(node.id, gap.node_ids[0], role.edge_type))

    result = Adaptation(
        patch=GraphPatch(nodes=nodes, edges=edges, rationale=question.text),
        answer_id=node.id,
        unresolved=gap.kind in _NEEDS_AN_EDIT,
    )
    if turn is None:
        return result

    result.synthesis = turn.synthesis
    _absorb(turn, node, result)
    return result


def _absorb(turn: DialecticTurn, answer: Node, result: Adaptation) -> None:
    """Add the exchange's material around the answer node."""
    for slot in turn.thesis.propositions:
        _add(slot, answer, EdgeType.SUPPORTS, result)
    for slot in turn.antithesis.propositions:
        _add(slot, answer, EdgeType.ATTACKS, result)


def _add(
    slot: CitationSlot, answer: Node, relation: EdgeType, result: Adaptation
) -> None:
    text = slot.proposition.strip()
    if not text:
        return

    # Defence in depth. The dialectic engine voids a turn whose propositions
    # carry citation strings, so this should never fire -- but the invariant it
    # protects is the one that keeps fabricated authority out of the manuscript,
    # and re-checking at the boundary costs a regex.
    hits = CitationChannel().find_hits(text)
    if hits:
        result.refused.append(f"proposition carries a citation string {hits}: {text[:80]}")
        return

    node_type, citation = _classify(slot, result, text)
    if node_type is None:
        return

    node = Node.propose(node_type, text, section=answer.section, citation=citation)
    result.patch.nodes.append(node)
    result.patch.edges.append(Edge(node.id, answer.id, relation))


def _classify(
    slot: CitationSlot, result: Adaptation, text: str
) -> tuple[NodeType | None, str]:
    """Decide what a proposition becomes, and whether it may carry its citation."""
    if slot.status is not SlotStatus.VERIFIED or not slot.normalized_cite:
        # A retrieved candidate is not authority: it moves a slot to `proposed`,
        # and only a lookup returning a cluster moves it to `verified`. Anything
        # short of that enters as a plain premise carrying no citation, so an
        # unconfirmed cite cannot be read off the manuscript as a real one.
        return NodeType.PREMISE, ""

    if NOT_OPERATIVE in slot.note:
        # A rescinded rule that verified is still rescinded. `Node` has no note
        # field, so the warning cannot travel with the node, and merging it
        # would let a repealed authority render as clean -- exactly what the
        # NOT_OPERATIVE marker exists to prevent. Refused, with the reason kept.
        result.refused.append(
            f"authority is {NOT_OPERATIVE} and a node cannot carry the warning: "
            f"{slot.normalized_cite}"
        )
        return None, ""

    return NodeType.AUTHORITY, slot.normalized_cite
