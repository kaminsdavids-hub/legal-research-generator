"""Grounding gate. Offline: no models, no network."""

from __future__ import annotations

from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    GraphPatch,
    Node,
    NodeType,
)
from modules.maieutic.grounding import GroundingFailure, GroundingGate


class _Verifier:
    """Programmable stand-in for retrieval + support checking."""

    def __init__(self, resolves: bool = True, supports: bool = True) -> None:
        self._resolves = resolves
        self._supports = supports
        self.resolve_calls: list[str] = []
        self.support_calls: list[tuple[str, str]] = []

    def resolve(self, citation: str) -> tuple[bool, str]:
        self.resolve_calls.append(citation)
        return self._resolves, "resolved" if self._resolves else "no such record"

    def supports(self, citation: str, claim: str) -> tuple[bool, str]:
        self.support_calls.append((citation, claim))
        return self._supports, "on point" if self._supports else "says nothing of the kind"


def _authority(text: str = "Encryption source code is protected speech.", cite: str = "176 F.3d 1132") -> Node:
    return Node.propose(NodeType.AUTHORITY, text, citation=cite)


def _original(text: str = "Weights should be treated the same way.") -> Node:
    return Node.from_human(NodeType.ORIGINAL, text)


# --------------------------------------------------------------------------- #
# AUTHORITY: the fabrication wall
# --------------------------------------------------------------------------- #
def test_a_resolving_supporting_citation_grounds_an_authority() -> None:
    gate = GroundingGate(_Verifier())
    node = _authority()
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert r.passed
    assert r.resolved_citation == "176 F.3d 1132"


def test_an_authority_without_a_citation_cannot_merge() -> None:
    gate = GroundingGate(_Verifier())
    node = Node.propose(NodeType.AUTHORITY, "Some court said something.")
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNCITED_AUTHORITY


def test_a_citation_that_does_not_resolve_is_a_fabrication() -> None:
    gate = GroundingGate(_Verifier(resolves=False))
    node = _authority(cite="999 U.S. 999")
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNRESOLVED_CITATION


def test_a_real_case_cited_for_something_it_does_not_say_fails() -> None:
    """The subtler fabrication: resolving is necessary and not sufficient."""
    verifier = _Verifier(resolves=True, supports=False)
    gate = GroundingGate(verifier)
    node = _authority("This case abolished the state secrets privilege.")
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNSUPPORTED_BY_SOURCE
    assert verifier.support_calls, "the claim must actually be checked against the source"


def test_without_a_verifier_an_authority_fails_closed() -> None:
    """Failing closed is the point: this is the fabrication wall."""
    gate = GroundingGate(verifier=None)
    node = _authority()
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNRESOLVED_CITATION


def test_a_generator_cannot_hand_over_a_pre_verified_authority() -> None:
    """Self-certification would make the gate decorative."""
    gate = GroundingGate(_Verifier())
    node = _authority().as_verified("176 F.3d 1132")
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.SELF_CERTIFIED


# --------------------------------------------------------------------------- #
# ORIGINAL: needs an argument, not a citation
# --------------------------------------------------------------------------- #
def test_an_original_node_asserted_alone_cannot_merge() -> None:
    gate = GroundingGate(_Verifier())
    node = _original()
    r = gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNARGUED_ORIGINAL


def test_an_original_node_supported_within_its_own_patch_merges() -> None:
    gate = GroundingGate(_Verifier())
    position = _original()
    premise = Node.propose(NodeType.PREMISE, "The expressive rationale does not depend on legibility.")
    patch = GraphPatch(
        nodes=[position, premise],
        edges=[Edge(premise.id, position.id, EdgeType.SUPPORTS)],
    )
    r = gate.assess(position, patch, ArgumentGraph())
    assert r.passed
    assert "argued by 1" in r.detail


def test_an_original_node_supported_by_the_existing_graph_merges() -> None:
    graph = ArgumentGraph()
    position = _original()
    graph.apply(GraphPatch(nodes=[position]))
    premise = Node.propose(NodeType.PREMISE, "An earlier premise.")
    graph.apply(
        GraphPatch(nodes=[premise], edges=[Edge(premise.id, position.id, EdgeType.SUPPORTS)])
    )
    r = GroundingGate(_Verifier()).assess(position, GraphPatch(nodes=[position]), graph)
    assert r.passed


def test_an_original_node_needs_no_citation() -> None:
    """Requiring one would push the author toward saying only what others said."""
    gate = GroundingGate(verifier=None)  # no verifier at all
    position = _original()
    premise = Node.propose(NodeType.PREMISE, "A supporting premise.")
    patch = GraphPatch(
        nodes=[position, premise], edges=[Edge(premise.id, position.id, EdgeType.SUPPORTS)]
    )
    assert gate.assess(position, patch, ArgumentGraph()).passed


def test_an_attack_does_not_count_as_support() -> None:
    gate = GroundingGate(_Verifier())
    position = _original()
    objection = Node.propose(NodeType.OBJECTION, "An objection to it.")
    patch = GraphPatch(
        nodes=[position, objection],
        edges=[Edge(objection.id, position.id, EdgeType.ATTACKS)],
    )
    r = gate.assess(position, patch, ArgumentGraph())
    assert not r.passed
    assert r.failure is GroundingFailure.UNARGUED_ORIGINAL


# --------------------------------------------------------------------------- #
# Structural nodes, and patch-level behaviour
# --------------------------------------------------------------------------- #
def test_structural_nodes_carry_no_grounding_rule() -> None:
    gate = GroundingGate(_Verifier())
    for kind in (NodeType.PREMISE, NodeType.OBJECTION, NodeType.REPLY, NodeType.DISTINCTION):
        node = Node.propose(kind, f"A {kind.value}.")
        assert gate.assess(node, GraphPatch(nodes=[node]), ArgumentGraph()).passed


def test_grounding_is_all_or_nothing_across_a_patch() -> None:
    """Unlike novelty, one ungrounded claim fails the merge.

    Letting the rest through would place a fabricated claim in the manuscript
    beside verified material, which is where it does the most damage.
    """
    gate = GroundingGate(_Verifier(resolves=False))
    good = Node.propose(NodeType.PREMISE, "A structural premise.")
    bad = _authority(cite="999 U.S. 999")
    result = gate.assess_patch(GraphPatch(nodes=[good, bad]), ArgumentGraph())
    assert not result.passed
    assert [f.node_id for f in result.failures] == [bad.id]


def test_a_wholly_grounded_patch_passes() -> None:
    gate = GroundingGate(_Verifier())
    authority = _authority()
    position = _original()
    patch = GraphPatch(
        nodes=[authority, position],
        edges=[Edge(authority.id, position.id, EdgeType.SUPPORTS)],
    )
    assert gate.assess_patch(patch, ArgumentGraph()).passed


def test_a_patch_that_grounds_nothing_does_not_pass_by_vacuous_truth() -> None:
    """``all([])`` is True. That exact reading once reported crashed eval
    questions as passing both gates (REMEDIATION §11.9).

    ``GraphPatch`` refuses to be constructed empty, so the reachable route to
    zero results is a patch whose nodes are all already in the graph. Nothing
    was assessed, so nothing may read as assessed.
    """
    graph = ArgumentGraph()
    existing = Node.propose(NodeType.PREMISE, "Already merged.")
    graph.apply(GraphPatch(nodes=[existing]))
    result = GroundingGate(_Verifier()).assess_patch(GraphPatch(nodes=[existing]), graph)
    assert result.results == []
    assert not result.passed


def test_nodes_already_in_the_graph_are_not_reassessed() -> None:
    graph = ArgumentGraph()
    existing = Node.propose(NodeType.PREMISE, "Already merged.")
    graph.apply(GraphPatch(nodes=[existing]))
    fresh = Node.propose(NodeType.PREMISE, "New.")
    result = GroundingGate(_Verifier()).assess_patch(
        GraphPatch(nodes=[existing, fresh]), graph
    )
    assert [r.node_id for r in result.results] == [fresh.id]
