"""Coherence gate. Offline: no models, no network."""

from __future__ import annotations

from modules.maieutic.coherence import (
    CoherenceGate,
    Incoherence,
    Severity,
)
from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    GraphPatch,
    Node,
    NodeType,
)


class _Critic:
    def __init__(self, label: str, source: str = "model") -> None:
        self.label = label
        self.source = source
        self.calls: list[tuple[str, str]] = []

    def relate(self, premise: str, hypothesis: str) -> tuple[str, str]:
        self.calls.append((premise, hypothesis))
        return self.label, self.source


def _node(text: str = "A claim.", kind: NodeType = NodeType.PREMISE) -> Node:
    return Node.propose(kind, text)


def _seeded() -> tuple[ArgumentGraph, Node]:
    graph = ArgumentGraph()
    thesis = Node.from_human(NodeType.THESIS, "The thesis.")
    graph.apply(GraphPatch(nodes=[thesis]))
    return graph, thesis


def _kinds(result: object) -> set[Incoherence]:
    return {f.kind for f in result.findings}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Structural findings block
# --------------------------------------------------------------------------- #
def test_a_patch_that_would_make_the_argument_ground_itself_is_blocked() -> None:
    graph, thesis = _seeded()
    a = _node("A.")
    graph.apply(GraphPatch(nodes=[a], edges=[Edge(a.id, thesis.id, EdgeType.DEPENDS_ON)]))

    closing = GraphPatch(
        nodes=[_node("Filler.")], edges=[Edge(thesis.id, a.id, EdgeType.DEPENDS_ON)]
    )
    result = CoherenceGate().assess(closing, graph)
    assert Incoherence.SELF_GROUNDING in _kinds(result)
    assert not result.passed


def test_a_chain_that_does_not_close_is_fine() -> None:
    graph, thesis = _seeded()
    a = _node("A.")
    patch = GraphPatch(nodes=[a], edges=[Edge(a.id, thesis.id, EdgeType.DEPENDS_ON)])
    assert CoherenceGate().assess(patch, graph).passed


def test_the_gate_does_not_merge_what_it_is_judging() -> None:
    """A gate with a side effect is a gate that has already merged."""
    graph, thesis = _seeded()
    before_nodes, before_edges = dict(graph.nodes), list(graph.edges)
    node = _node()
    CoherenceGate().assess(
        GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)]),
        graph,
    )
    assert graph.nodes == before_nodes
    assert graph.edges == before_edges


def test_an_edge_pointing_at_nothing_is_blocked() -> None:
    graph, _ = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, "ghost", EdgeType.SUPPORTS)])
    result = CoherenceGate().assess(patch, graph)
    assert Incoherence.DANGLING_EDGE in _kinds(result)
    assert not result.passed


def test_a_dangling_edge_does_not_crash_the_checks_downstream_of_it() -> None:
    """The later checks reason over a merged view and must not see it."""
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(
        nodes=[node],
        edges=[
            Edge(node.id, "ghost", EdgeType.SUPPORTS),
            Edge(node.id, thesis.id, EdgeType.SUPPORTS),
        ],
    )
    result = CoherenceGate().assess(patch, graph)
    assert Incoherence.DISCONNECTED not in _kinds(result)


def test_a_node_cannot_both_support_and_attack_the_same_claim() -> None:
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(
        nodes=[node],
        edges=[
            Edge(node.id, thesis.id, EdgeType.SUPPORTS),
            Edge(node.id, thesis.id, EdgeType.ATTACKS),
        ],
    )
    result = CoherenceGate().assess(patch, graph)
    assert Incoherence.CONTRADICTORY_RELATION in _kinds(result)
    assert not result.passed


def test_supporting_one_claim_and_attacking_another_is_ordinary() -> None:
    graph, thesis = _seeded()
    other = _node("Another claim.")
    graph.apply(GraphPatch(nodes=[other], edges=[Edge(other.id, thesis.id, EdgeType.SUPPORTS)]))
    node = _node()
    patch = GraphPatch(
        nodes=[node],
        edges=[
            Edge(node.id, thesis.id, EdgeType.SUPPORTS),
            Edge(node.id, other.id, EdgeType.ATTACKS),
        ],
    )
    assert CoherenceGate().assess(patch, graph).passed


def test_a_contradiction_already_in_the_graph_is_not_this_merges_fault() -> None:
    """Blocking on it would make the graph unmergeable forever."""
    graph, thesis = _seeded()
    old = _node("Pre-existing.")
    graph.apply(
        GraphPatch(
            nodes=[old],
            edges=[
                Edge(old.id, thesis.id, EdgeType.SUPPORTS),
                Edge(old.id, thesis.id, EdgeType.ATTACKS),
            ],
        )
    )
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)])
    assert CoherenceGate().assess(patch, graph).passed


def test_a_node_joining_the_manuscript_as_part_of_no_argument_is_blocked() -> None:
    graph, thesis = _seeded()
    attached, stray = _node("Attached."), _node("Stray.")
    patch = GraphPatch(
        nodes=[attached, stray],
        edges=[Edge(attached.id, thesis.id, EdgeType.SUPPORTS)],
    )
    result = CoherenceGate().assess(patch, graph)
    disconnected = [f for f in result.findings if f.kind is Incoherence.DISCONNECTED]
    assert [f.node_ids for f in disconnected] == [(stray.id,)]


def test_the_opening_node_of_an_empty_graph_is_exempt() -> None:
    """There is nothing yet to connect to; refusing it makes the first patch
    unmergeable.
    """
    assert CoherenceGate().assess(GraphPatch(nodes=[_node()]), ArgumentGraph()).passed


def test_a_first_patch_of_several_nodes_must_still_connect_them() -> None:
    patch = GraphPatch(nodes=[_node("One."), _node("Two.")])
    assert not CoherenceGate().assess(patch, ArgumentGraph()).passed


def test_a_reply_that_replies_to_nothing_is_blocked() -> None:
    graph, thesis = _seeded()
    reply = _node("A reply.", NodeType.REPLY)
    other = _node("Filler.")
    patch = GraphPatch(
        nodes=[reply, other],
        edges=[
            Edge(other.id, thesis.id, EdgeType.SUPPORTS),
            # The reply is attached, but only as something else's target.
            Edge(other.id, reply.id, EdgeType.SUPPORTS),
        ],
    )
    result = CoherenceGate().assess(patch, graph)
    assert Incoherence.REPLY_TO_NOTHING in _kinds(result)


def test_a_reply_pointing_at_an_objection_is_fine() -> None:
    graph, thesis = _seeded()
    objection = _node("An objection.", NodeType.OBJECTION)
    graph.apply(
        GraphPatch(nodes=[objection], edges=[Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    )
    reply = _node("A reply.", NodeType.REPLY)
    patch = GraphPatch(
        nodes=[reply], edges=[Edge(reply.id, objection.id, EdgeType.SUPPORTS)]
    )
    assert CoherenceGate().assess(patch, graph).passed


# --------------------------------------------------------------------------- #
# Semantic findings are advisory
# --------------------------------------------------------------------------- #
def test_support_that_contradicts_is_reported_but_does_not_block() -> None:
    """An uncalibrated critic must not be able to veto the author's work.

    A heuristic in this repository once scored `constitutional` against
    `unconstitutional` as entailment; a judge rated failure placeholders 8/10.
    """
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)])
    result = CoherenceGate(_Critic("contradiction")).assess(patch, graph)
    assert Incoherence.SUPPORT_THAT_CONTRADICTS in _kinds(result)
    assert result.passed, "a model's opinion does not block a merge"
    assert result.advisory and not result.blocking


def test_an_attack_that_agrees_with_its_target_is_reported() -> None:
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.ATTACKS)])
    result = CoherenceGate(_Critic("entailment")).assess(patch, graph)
    assert Incoherence.ATTACK_THAT_AGREES in _kinds(result)
    assert result.passed


def test_the_expected_relations_produce_no_finding() -> None:
    graph, thesis = _seeded()
    supporter, attacker = _node("Supports."), _node("Attacks.")
    patch = GraphPatch(
        nodes=[supporter, attacker],
        edges=[
            Edge(supporter.id, thesis.id, EdgeType.SUPPORTS),
            Edge(attacker.id, thesis.id, EdgeType.ATTACKS),
        ],
    )
    entails = CoherenceGate(_Critic("entailment")).assess(patch, graph)
    assert Incoherence.SUPPORT_THAT_CONTRADICTS not in _kinds(entails)
    contradicts = CoherenceGate(_Critic("contradiction")).assess(patch, graph)
    assert Incoherence.ATTACK_THAT_AGREES not in _kinds(contradicts)


def test_every_semantic_finding_records_which_critic_path_answered() -> None:
    """A silent downgrade from a model to a heuristic changes what it means."""
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)])
    result = CoherenceGate(_Critic("contradiction", "heuristic")).assess(patch, graph)
    assert [f.critic_source for f in result.advisory] == ["heuristic"]


def test_without_a_critic_no_semantic_claim_is_made() -> None:
    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)])
    assert CoherenceGate().assess(patch, graph).advisory == []


def test_a_broken_critic_costs_the_check_not_the_merge() -> None:
    class _Broken:
        def relate(self, premise: str, hypothesis: str) -> tuple[str, str]:
            raise RuntimeError("model unavailable")

    graph, thesis = _seeded()
    node = _node()
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.SUPPORTS)])
    result = CoherenceGate(_Broken()).assess(patch, graph)
    assert result.passed
    assert result.advisory == []


def test_the_critic_is_not_asked_about_relations_it_cannot_judge() -> None:
    graph, thesis = _seeded()
    node = _node()
    critic = _Critic("contradiction")
    patch = GraphPatch(nodes=[node], edges=[Edge(node.id, thesis.id, EdgeType.DEPENDS_ON)])
    CoherenceGate(critic).assess(patch, graph)
    assert not critic.calls


def test_a_semantic_finding_never_flips_a_patch_to_failing() -> None:
    """The severity table is the invariant, not the individual checks."""
    for kind, severity in [
        (Incoherence.SUPPORT_THAT_CONTRADICTS, Severity.ADVISORY),
        (Incoherence.ATTACK_THAT_AGREES, Severity.ADVISORY),
        (Incoherence.SELF_GROUNDING, Severity.BLOCKING),
        (Incoherence.DANGLING_EDGE, Severity.BLOCKING),
        (Incoherence.CONTRADICTORY_RELATION, Severity.BLOCKING),
        (Incoherence.DISCONNECTED, Severity.BLOCKING),
        (Incoherence.REPLY_TO_NOTHING, Severity.BLOCKING),
    ]:
        from modules.maieutic.coherence import Finding

        assert Finding(kind=kind, node_ids=("n",)).severity is severity


def test_every_incoherence_has_a_severity() -> None:
    """A kind with no entry would raise at report time, not at review time."""
    from modules.maieutic.coherence import Finding

    for kind in Incoherence:
        assert Finding(kind=kind, node_ids=("n",)).severity in Severity
