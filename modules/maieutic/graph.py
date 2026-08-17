"""The argument graph: the manuscript's actual representation.

The manuscript is an argument graph, not a document; prose is a rendering of the
graph. That is not a metaphor, it is what makes non-repetition enforceable in
code rather than by exhortation: every insertion is new nodes plus edges, and a
patch adding zero nodes is rejected by construction rather than by a critic that
can be talked around.

Two invariants are enforced here rather than in prompts, because a prompt is a
request and this needs to be a guarantee:

* ``Provenance.HUMAN`` is assignable only through :meth:`Node.from_human` and
  :meth:`Node.edited_by_human`. Every other construction path refuses it, so
  machine-generated content cannot acquire the author's authority by being
  labelled with it. Originality enters the system through the author's answers;
  the machine's job is to test, sharpen and connect them.
* A :class:`GraphPatch` with no new nodes cannot be applied. Restatement is not
  an insertion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    """What kind of claim a node makes."""

    THESIS = "thesis"
    PREMISE = "premise"
    DISTINCTION = "distinction"
    OBJECTION = "objection"
    REPLY = "reply"
    IMPLICATION = "implication"
    #: A cited, source-backed proposition. Requires a resolving citation.
    AUTHORITY = "authority"
    #: An author-asserted normative or analytical position. Requires no
    #: citation, but cannot stand unsupported: see `GroundingRule`.
    ORIGINAL = "original"


class EdgeType(StrEnum):
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    QUALIFIES = "qualifies"
    IMPLIES = "implies"
    DEPENDS_ON = "depends_on"
    DISTINGUISHES = "distinguishes"


class Provenance(StrEnum):
    """Who authored a node. The scholarly-integrity record turns on this."""

    #: Captured verbatim from the author, or edited by them. Only the
    #: answer-capture and edit paths may assign it.
    HUMAN = "human"
    #: Proposed by the dialectic module and accepted by the author.
    DIALECTIC = "dialectic"
    #: Structural material the system added (e.g. a background premise).
    SYSTEM = "system"


class ProvenanceViolation(Exception):
    """Raised when human provenance is claimed outside the paths that may grant it."""


class EmptyPatch(Exception):
    """Raised when a patch would add no new nodes.

    A patch that only adds edges asserts nothing new; a patch that adds nothing
    at all is a restatement wearing the shape of an insertion.
    """


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    text: str
    provenance: Provenance
    #: Section of the manuscript this node renders into.
    section: str = ""
    #: Normalized citation for an AUTHORITY node. Never set by a generator:
    #: the dialectic module is forbidden from emitting citations at all, so an
    #: authority reaches the graph only through retrieval and verification.
    citation: str = ""
    #: Set by the grounding gate once a citation has resolved against the
    #: corpus. A generator cannot self-certify an authority.
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a node must assert something")

    @classmethod
    def propose(
        cls,
        type: NodeType,
        text: str,
        *,
        provenance: Provenance = Provenance.DIALECTIC,
        section: str = "",
        citation: str = "",
        node_id: str | None = None,
    ) -> Node:
        """Construct a machine-proposed node.

        Refuses ``Provenance.HUMAN``. This is the path the dialectic adapter
        uses, and it must not be able to mint content carrying the author's
        authority — that is the whole basis of the provenance record.
        """
        if provenance is Provenance.HUMAN:
            raise ProvenanceViolation(
                "human provenance is assignable only by Node.from_human or "
                "Node.edited_by_human; a proposed node cannot claim it"
            )
        return cls(
            id=node_id or _new_id(),
            type=type,
            text=text,
            provenance=provenance,
            section=section,
            citation=citation,
        )

    @classmethod
    def from_human(
        cls, type: NodeType, text: str, *, section: str = "", node_id: str | None = None
    ) -> Node:
        """Capture an author's answer verbatim. One of two paths granting HUMAN."""
        return cls(
            id=node_id or _new_id(),
            type=type,
            text=text,
            provenance=Provenance.HUMAN,
            section=section,
        )

    def edited_by_human(self, text: str) -> Node:
        """Author edit of a proposed node. The other path granting HUMAN.

        Editing is authorship: a node the author rewrote is theirs, even if a
        machine drafted it.
        """
        return Node(
            id=self.id,
            type=self.type,
            text=text,
            provenance=Provenance.HUMAN,
            section=self.section,
            citation=self.citation,
            verified=self.verified,
        )

    def as_verified(self, citation: str) -> Node:
        """Mark an AUTHORITY node as resolved. Only the grounding gate calls this."""
        if self.type is not NodeType.AUTHORITY:
            raise ValueError("only an AUTHORITY node carries verification")
        return Node(
            id=self.id,
            type=self.type,
            text=self.text,
            provenance=self.provenance,
            section=self.section,
            citation=citation,
            verified=True,
        )


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: EdgeType

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("a node cannot stand in a relation to itself")


@dataclass
class GraphPatch:
    """A proposed insertion: new nodes and the edges connecting them."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    #: Why this patch exists — the Socratic question that produced it.
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.nodes:
            raise EmptyPatch(
                "a patch must add at least one node; an insertion that asserts "
                "nothing new is a restatement"
            )


@dataclass
class ArgumentGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Queries the Socratic engine and the gates run over
    # ------------------------------------------------------------------ #
    def of_type(self, *types: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.type in types]

    def incoming(self, node_id: str, *types: EdgeType) -> list[Edge]:
        return [
            e for e in self.edges
            if e.target == node_id and (not types or e.type in types)
        ]

    def outgoing(self, node_id: str, *types: EdgeType) -> list[Edge]:
        return [
            e for e in self.edges
            if e.source == node_id and (not types or e.type in types)
        ]

    def is_connected(self, node_id: str) -> bool:
        return bool(self.incoming(node_id) or self.outgoing(node_id))

    def depends_on_cycle(self) -> list[str] | None:
        """Return a DEPENDS_ON cycle if one exists.

        A cycle means the argument grounds itself, which no rendering order can
        fix, so the coherence gate rejects it.
        """
        adjacency: dict[str, list[str]] = {}
        for e in self.edges:
            if e.type is EdgeType.DEPENDS_ON:
                adjacency.setdefault(e.source, []).append(e.target)

        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {}

        def walk(node: str, path: list[str]) -> list[str] | None:
            colour[node] = GREY
            for nxt in adjacency.get(node, []):
                state = colour.get(nxt, WHITE)
                if state == GREY:
                    return [*path, node, nxt]
                if state == WHITE:
                    found = walk(nxt, [*path, node])
                    if found:
                        return found
            colour[node] = BLACK
            return None

        for node in list(adjacency):
            if colour.get(node, WHITE) == WHITE:
                found = walk(node, [])
                if found:
                    return found
        return None

    def unanswered_attacks(self) -> list[Edge]:
        """ATTACKS edges whose target has no REPLY.

        Allowed, but only as an explicitly flagged open problem — a paper that
        states its strongest objections and answers them is the point, and one
        that states them and quietly moves on is not.
        """
        # An attack is answered when the OBJECTION doing the attacking has a
        # reply pointing at it — not when the thing being attacked does. Reading
        # the target here reports every attack as open, since the target is the
        # thesis and replies attach to the objection.
        answered = {
            e.target
            for e in self.edges
            if (source := self.nodes.get(e.source)) is not None
            and source.type is NodeType.REPLY
        }
        return [
            e for e in self.edges
            if e.type is EdgeType.ATTACKS and e.source not in answered
        ]

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def apply(self, patch: GraphPatch) -> list[str]:
        """Merge a patch. Callers must run the gates first; this enforces shape only.

        Returns the ids of the nodes added.
        """
        fresh = [n for n in patch.nodes if n.id not in self.nodes]
        if not fresh:
            raise EmptyPatch(
                "every node in this patch already exists in the graph; nothing "
                "is being asserted"
            )
        known = set(self.nodes) | {n.id for n in fresh}
        for edge in patch.edges:
            missing = {edge.source, edge.target} - known
            if missing:
                raise ValueError(f"edge references unknown node(s): {sorted(missing)}")

        for node in fresh:
            self.nodes[node.id] = node
        self.edges.extend(patch.edges)
        return [n.id for n in fresh]

    def replace(self, node: Node) -> None:
        """Swap a node in place, preserving its edges. Used by the edit path."""
        if node.id not in self.nodes:
            raise KeyError(f"unknown node {node.id}")
        self.nodes[node.id] = node

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "text": n.text,
                    "provenance": n.provenance.value,
                    "section": n.section,
                    "citation": n.citation,
                    "verified": n.verified,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.type.value}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentGraph:
        graph = cls()
        for raw in data.get("nodes", []):
            node = Node(
                id=raw["id"],
                type=NodeType(raw["type"]),
                text=raw["text"],
                provenance=Provenance(raw["provenance"]),
                section=raw.get("section", ""),
                citation=raw.get("citation", ""),
                verified=bool(raw.get("verified", False)),
            )
            graph.nodes[node.id] = node
        for raw in data.get("edges", []):
            graph.edges.append(
                Edge(source=raw["source"], target=raw["target"], type=EdgeType(raw["type"]))
            )
        return graph
