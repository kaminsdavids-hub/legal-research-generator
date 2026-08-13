"""Argument-graph invariants. Offline: no models, no network."""

from __future__ import annotations

import pytest

from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    EmptyPatch,
    GraphPatch,
    Node,
    NodeType,
    Provenance,
    ProvenanceViolation,
)


def _thesis(text: str = "Weights are protected expression.") -> Node:
    return Node.from_human(NodeType.THESIS, text, section="II")


# --------------------------------------------------------------------------- #
# Provenance is a guarantee, not a label
# --------------------------------------------------------------------------- #
def test_a_proposed_node_cannot_claim_human_provenance() -> None:
    """Originality enters through the author's answers.

    If a generator could mint HUMAN nodes, the provenance record would be
    decoration rather than a scholarly-integrity artefact.
    """
    with pytest.raises(ProvenanceViolation):
        Node.propose(NodeType.ORIGINAL, "A position.", provenance=Provenance.HUMAN)


def test_the_two_paths_that_may_grant_human_provenance() -> None:
    captured = Node.from_human(NodeType.ORIGINAL, "The author's own view.")
    assert captured.provenance is Provenance.HUMAN

    proposed = Node.propose(NodeType.OBJECTION, "A machine-drafted objection.")
    assert proposed.provenance is Provenance.DIALECTIC

    # Editing is authorship.
    edited = proposed.edited_by_human("The author's rewrite of that objection.")
    assert edited.provenance is Provenance.HUMAN
    assert edited.id == proposed.id, "an edit revises a node, it does not create one"


def test_default_proposal_provenance_is_dialectic() -> None:
    assert Node.propose(NodeType.REPLY, "A reply.").provenance is Provenance.DIALECTIC
    assert (
        Node.propose(NodeType.PREMISE, "Background.", provenance=Provenance.SYSTEM).provenance
        is Provenance.SYSTEM
    )


# --------------------------------------------------------------------------- #
# A patch that asserts nothing cannot be applied
# --------------------------------------------------------------------------- #
def test_a_patch_with_no_nodes_cannot_be_constructed() -> None:
    """Non-repetition is enforced by construction, not by exhortation."""
    with pytest.raises(EmptyPatch):
        GraphPatch(nodes=[], edges=[])


def test_a_patch_of_only_existing_nodes_is_rejected() -> None:
    graph = ArgumentGraph()
    node = _thesis()
    graph.apply(GraphPatch(nodes=[node]))

    with pytest.raises(EmptyPatch, match="already exists"):
        graph.apply(GraphPatch(nodes=[node]))


def test_applying_a_patch_returns_only_the_new_nodes() -> None:
    graph = ArgumentGraph()
    first = _thesis()
    graph.apply(GraphPatch(nodes=[first]))

    second = Node.propose(NodeType.OBJECTION, "But consider the export-control analogy.")
    added = graph.apply(
        GraphPatch(
            nodes=[first, second],
            edges=[Edge(second.id, first.id, EdgeType.ATTACKS)],
        )
    )
    assert added == [second.id]
    assert len(graph.nodes) == 2


def test_an_edge_to_an_unknown_node_is_rejected() -> None:
    graph = ArgumentGraph()
    node = Node.propose(NodeType.PREMISE, "A premise.")
    with pytest.raises(ValueError, match="unknown node"):
        graph.apply(GraphPatch(nodes=[node], edges=[Edge(node.id, "nope", EdgeType.SUPPORTS)]))


def test_a_node_cannot_relate_to_itself() -> None:
    with pytest.raises(ValueError):
        Edge("a", "a", EdgeType.SUPPORTS)


def test_an_empty_node_asserts_nothing() -> None:
    with pytest.raises(ValueError):
        Node.propose(NodeType.PREMISE, "   ")


# --------------------------------------------------------------------------- #
# Authority cannot self-certify
# --------------------------------------------------------------------------- #
def test_only_the_grounding_gate_marks_an_authority_verified() -> None:
    authority = Node.propose(NodeType.AUTHORITY, "Encryption source code is protected speech.")
    assert not authority.verified
    assert authority.citation == ""

    resolved = authority.as_verified("176 F.3d 1132")
    assert resolved.verified
    assert resolved.citation == "176 F.3d 1132"


def test_a_non_authority_node_cannot_be_verified() -> None:
    with pytest.raises(ValueError, match="only an AUTHORITY"):
        Node.propose(NodeType.ORIGINAL, "A position.").as_verified("176 F.3d 1132")


# --------------------------------------------------------------------------- #
# Queries the gates and the Socratic engine depend on
# --------------------------------------------------------------------------- #
def test_undefended_flank_is_a_thesis_with_no_incoming_attack() -> None:
    graph = ArgumentGraph()
    defended, exposed = _thesis("Defended."), _thesis("Nobody has tried to kill this.")
    objection = Node.propose(NodeType.OBJECTION, "An attack.")
    graph.apply(
        GraphPatch(
            nodes=[defended, exposed, objection],
            edges=[Edge(objection.id, defended.id, EdgeType.ATTACKS)],
        )
    )
    flanks = [
        n for n in graph.of_type(NodeType.THESIS, NodeType.ORIGINAL)
        if not graph.incoming(n.id, EdgeType.ATTACKS)
    ]
    assert [n.id for n in flanks] == [exposed.id]


def test_a_depends_on_cycle_is_detected() -> None:
    graph = ArgumentGraph()
    a, b, c = (Node.propose(NodeType.PREMISE, t) for t in ("A.", "B.", "C."))
    graph.apply(
        GraphPatch(
            nodes=[a, b, c],
            edges=[
                Edge(a.id, b.id, EdgeType.DEPENDS_ON),
                Edge(b.id, c.id, EdgeType.DEPENDS_ON),
                Edge(c.id, a.id, EdgeType.DEPENDS_ON),
            ],
        )
    )
    assert graph.depends_on_cycle() is not None


def test_a_dependency_chain_without_a_cycle_is_fine() -> None:
    graph = ArgumentGraph()
    a, b = Node.propose(NodeType.PREMISE, "A."), Node.propose(NodeType.PREMISE, "B.")
    graph.apply(GraphPatch(nodes=[a, b], edges=[Edge(a.id, b.id, EdgeType.DEPENDS_ON)]))
    assert graph.depends_on_cycle() is None


def test_an_attack_with_no_reply_is_reported_as_an_open_problem() -> None:
    graph = ArgumentGraph()
    thesis = _thesis()
    answered = Node.propose(NodeType.OBJECTION, "Answered objection.")
    unanswered = Node.propose(NodeType.OBJECTION, "Unanswered objection.")
    reply = Node.propose(NodeType.REPLY, "The reply.")
    graph.apply(
        GraphPatch(
            nodes=[thesis, answered, unanswered, reply],
            edges=[
                Edge(answered.id, thesis.id, EdgeType.ATTACKS),
                Edge(reply.id, answered.id, EdgeType.SUPPORTS),
                Edge(unanswered.id, thesis.id, EdgeType.ATTACKS),
            ],
        )
    )
    open_problems = graph.unanswered_attacks()
    assert [e.source for e in open_problems] == [unanswered.id]


def test_a_floating_node_is_detectable() -> None:
    graph = ArgumentGraph()
    connected, island = _thesis(), Node.propose(NodeType.PREMISE, "Unconnected.")
    support = Node.propose(NodeType.PREMISE, "A support.")
    graph.apply(
        GraphPatch(
            nodes=[connected, support, island],
            edges=[Edge(support.id, connected.id, EdgeType.SUPPORTS)],
        )
    )
    assert not graph.is_connected(island.id)
    assert graph.is_connected(connected.id)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def test_a_graph_survives_serialisation() -> None:
    graph = ArgumentGraph()
    thesis = _thesis()
    authority = Node.propose(NodeType.AUTHORITY, "Code is speech.").as_verified("176 F.3d 1132")
    graph.apply(
        GraphPatch(
            nodes=[thesis, authority],
            edges=[Edge(authority.id, thesis.id, EdgeType.SUPPORTS)],
        )
    )
    restored = ArgumentGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()
    assert restored.nodes[thesis.id].provenance is Provenance.HUMAN
    assert restored.nodes[authority.id].verified
