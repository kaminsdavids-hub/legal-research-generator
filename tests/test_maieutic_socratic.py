"""Socratic gap analysis and question policy. Offline: no models, no network."""

from __future__ import annotations

from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    GraphPatch,
    Node,
    NodeType,
)
from modules.maieutic.socratic import (
    PRIORITY,
    AskedLog,
    Gap,
    GapKind,
    Phrasing,
    SocraticEngine,
    analyse,
)


def _graph(nodes: list[Node], edges: list[Edge] | None = None) -> ArgumentGraph:
    graph = ArgumentGraph()
    graph.apply(GraphPatch(nodes=nodes, edges=edges or []))
    return graph


def _kinds(graph: ArgumentGraph) -> list[GapKind]:
    return [g.kind for g in analyse(graph)]


def _thesis(text: str = "Weights are expressive.", section: str = "") -> Node:
    return Node.from_human(NodeType.THESIS, text, section=section)


# --------------------------------------------------------------------------- #
# Gap analysis: structure, not prose
# --------------------------------------------------------------------------- #
def test_an_objection_with_no_reply_is_a_gap() -> None:
    thesis = _thesis()
    objection = Node.propose(NodeType.OBJECTION, "But weights are functional.")
    graph = _graph([thesis, objection], [Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    gaps = [g for g in analyse(graph) if g.kind is GapKind.UNANSWERED_ATTACK]
    assert [g.node_ids for g in gaps] == [(objection.id, thesis.id)]


def test_an_answered_objection_is_not_a_gap() -> None:
    thesis = _thesis()
    objection = Node.propose(NodeType.OBJECTION, "But weights are functional.")
    reply = Node.from_human(NodeType.REPLY, "Functionality does not defeat expression.")
    graph = _graph(
        [thesis, objection, reply],
        [
            Edge(objection.id, thesis.id, EdgeType.ATTACKS),
            Edge(reply.id, objection.id, EdgeType.SUPPORTS),
        ],
    )
    assert GapKind.UNANSWERED_ATTACK not in _kinds(graph)


def test_a_claim_nothing_attacks_is_untested() -> None:
    graph = _graph([_thesis()])
    assert GapKind.UNCONTESTED_CLAIM in _kinds(graph)


def test_a_contested_claim_is_not_reported_as_untested() -> None:
    thesis = _thesis()
    objection = Node.propose(NodeType.OBJECTION, "An objection.")
    graph = _graph([thesis, objection], [Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    assert GapKind.UNCONTESTED_CLAIM not in _kinds(graph)


def test_only_claim_bearing_nodes_need_contesting() -> None:
    """A premise is plumbing. Demanding an objection to every one of them would
    bury the questions that matter under questions that do not.
    """
    graph = _graph([Node.propose(NodeType.PREMISE, "A background premise.")])
    assert GapKind.UNCONTESTED_CLAIM not in _kinds(graph)


def test_an_original_position_with_nothing_under_it_is_a_gap() -> None:
    """Reachable despite the grounding gate, via a graph loaded from disk.

    Analysis that only covers the happy path is not analysis.
    """
    graph = ArgumentGraph.from_dict(
        {
            "nodes": [
                {
                    "id": "n1",
                    "type": "original",
                    "text": "A floating position.",
                    "provenance": "human",
                }
            ],
            "edges": [],
        }
    )
    assert GapKind.UNSUPPORTED_CLAIM in _kinds(graph)


def test_a_supported_position_is_not_reported_as_unsupported() -> None:
    position = Node.from_human(NodeType.ORIGINAL, "A position.")
    premise = Node.propose(NodeType.PREMISE, "A premise.")
    graph = _graph([position, premise], [Edge(premise.id, position.id, EdgeType.SUPPORTS)])
    assert GapKind.UNSUPPORTED_CLAIM not in _kinds(graph)


def test_an_attack_does_not_count_as_holding_a_claim_up() -> None:
    position = Node.from_human(NodeType.ORIGINAL, "A position.")
    objection = Node.propose(NodeType.OBJECTION, "An objection.")
    graph = _graph([position, objection], [Edge(objection.id, position.id, EdgeType.ATTACKS)])
    assert GapKind.UNSUPPORTED_CLAIM in _kinds(graph)


def test_a_node_attached_to_nothing_is_a_gap() -> None:
    thesis = _thesis()
    premise = Node.propose(NodeType.PREMISE, "A premise.")
    stray = Node.propose(NodeType.PREMISE, "Attached to nothing.")
    graph = _graph(
        [thesis, premise, stray], [Edge(premise.id, thesis.id, EdgeType.SUPPORTS)]
    )
    orphans = [g for g in analyse(graph) if g.kind is GapKind.ORPHANED_NODE]
    assert [g.node_ids for g in orphans] == [(stray.id,)]


def test_the_opening_thesis_is_not_an_orphan() -> None:
    """A paper that has just started has nothing for its thesis to connect to."""
    assert GapKind.ORPHANED_NODE not in _kinds(_graph([_thesis()]))


def test_an_unverified_authority_is_a_gap_and_a_verified_one_is_not() -> None:
    unverified = Node.propose(NodeType.AUTHORITY, "A holding.", citation="176 F.3d 1132")
    graph = _graph([unverified])
    assert GapKind.UNVERIFIED_AUTHORITY in _kinds(graph)

    graph.replace(unverified.as_verified("176 F.3d 1132"))
    assert GapKind.UNVERIFIED_AUTHORITY not in _kinds(graph)


def test_an_argument_that_grounds_itself_is_a_gap() -> None:
    a = Node.from_human(NodeType.ORIGINAL, "A.")
    b = Node.from_human(NodeType.ORIGINAL, "B.")
    graph = _graph(
        [a, b],
        [Edge(a.id, b.id, EdgeType.DEPENDS_ON), Edge(b.id, a.id, EdgeType.DEPENDS_ON)],
    )
    assert GapKind.SELF_GROUNDING in _kinds(graph)


def test_a_gaps_key_is_stable_across_passes() -> None:
    """'Already asked' means nothing if the key changes between cycles."""
    graph = _graph([_thesis()])
    assert [g.key for g in analyse(graph)] == [g.key for g in analyse(graph)]


def test_a_key_distinguishes_the_kind_of_gap_from_the_node() -> None:
    same = ("n1",)
    assert Gap(GapKind.ORPHANED_NODE, same).key != Gap(GapKind.UNCONTESTED_CLAIM, same).key


# --------------------------------------------------------------------------- #
# Question policy
# --------------------------------------------------------------------------- #
def test_the_worst_hole_is_asked_about_first() -> None:
    """A cycle beats a hanging objection beats an unverified citation."""
    a = Node.from_human(NodeType.ORIGINAL, "A.")
    b = Node.from_human(NodeType.ORIGINAL, "B.")
    authority = Node.propose(NodeType.AUTHORITY, "A holding.", citation="1 U.S. 1")
    graph = _graph(
        [a, b, authority],
        [Edge(a.id, b.id, EdgeType.DEPENDS_ON), Edge(b.id, a.id, EdgeType.DEPENDS_ON)],
    )
    asked = SocraticEngine().ask(graph, limit=3)
    assert asked[0].gap.kind is GapKind.SELF_GROUNDING
    kinds = [q.gap.kind for q in asked]
    assert kinds == sorted(kinds, key=PRIORITY.index)


def test_a_gap_already_asked_about_is_not_asked_again() -> None:
    graph = _graph([_thesis()])
    engine, log = SocraticEngine(), AskedLog()
    first = engine.ask(graph, log)
    log.record(first[0])
    assert first[0].gap.key not in {q.gap.key for q in engine.ask(graph, log, limit=5)}


def test_rewording_a_question_does_not_make_it_new() -> None:
    """The log keys on the gap, not the text."""

    class _Reworder:
        def phrase(self, gap: Gap, default: str, quotes: list[str]) -> str:
            return "An entirely different wording of the same question?"

    graph = _graph([_thesis()])
    log = AskedLog()
    log.record(SocraticEngine().ask(graph, log)[0])
    assert not SocraticEngine(_Reworder()).ask(graph, log, limit=5)


def test_an_asked_but_unanswered_gap_stays_visible() -> None:
    """It stops being offered so the loop moves on. It does not stop existing.

    A hole that vanishes from the report because it was mentioned once is the
    report lying.
    """
    graph = _graph([_thesis()])
    log = AskedLog()
    log.record(SocraticEngine().ask(graph, log)[0])
    assert [g.kind for g in log.outstanding(graph)] == [GapKind.UNCONTESTED_CLAIM]


def test_an_answered_gap_drops_out_of_outstanding() -> None:
    thesis = _thesis()
    graph = _graph([thesis])
    log = AskedLog()
    log.record(SocraticEngine().ask(graph, log)[0])

    objection = Node.propose(NodeType.OBJECTION, "The author's own strongest objection.")
    graph.apply(
        GraphPatch(nodes=[objection], edges=[Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    )
    assert GapKind.UNCONTESTED_CLAIM not in {g.kind for g in log.outstanding(graph)}


def test_no_gaps_means_no_question_rather_than_an_invented_one() -> None:
    """Manufacturing a question to fill a quota spends the author's attention on
    whatever the engine could think of instead of on a hole.
    """
    graph = _graph([_thesis()])
    log = AskedLog()
    for question in SocraticEngine().ask(graph, log, limit=10):
        log.record(question)
    assert SocraticEngine().ask(graph, log, limit=10) == []


def test_limit_is_respected() -> None:
    a, b, c = (_thesis(f"Claim {i}.") for i in range(3))
    graph = _graph([a, b, c])
    assert len(SocraticEngine().ask(graph, limit=2)) == 2


def test_a_mined_out_section_is_deprioritised_not_abandoned() -> None:
    """A cycle in a mined-out section still matters more than a stray citation
    somewhere fresh.
    """
    a = Node.from_human(NodeType.ORIGINAL, "A.", section="mined")
    b = Node.from_human(NodeType.ORIGINAL, "B.", section="mined")
    authority = Node.propose(
        NodeType.AUTHORITY, "A holding.", citation="1 U.S. 1", section="fresh"
    )
    graph = _graph(
        [a, b, authority],
        [Edge(a.id, b.id, EdgeType.DEPENDS_ON), Edge(b.id, a.id, EdgeType.DEPENDS_ON)],
    )
    questions = SocraticEngine().ask(
        graph, limit=10, avoid_sections=frozenset({"mined"})
    )
    kinds = [q.gap.kind for q in questions]
    assert GapKind.SELF_GROUNDING in kinds, "a mined-out section is not abandoned"
    assert kinds.index(GapKind.UNVERIFIED_AUTHORITY) < kinds.index(GapKind.SELF_GROUNDING)


# --------------------------------------------------------------------------- #
# Phrasing
# --------------------------------------------------------------------------- #
def test_a_question_quotes_the_claim_it_is_about() -> None:
    graph = _graph([_thesis("Weights are expressive.")])
    text = SocraticEngine().ask(graph)[0].text
    assert "Weights are expressive." in text
    assert text.rstrip().endswith("?")


def test_every_gap_kind_has_a_question() -> None:
    """A gap the engine can find but cannot ask about is a silent dead end."""
    graph = _graph([_thesis()])
    gap_holder = analyse(graph)[0]
    engine = SocraticEngine()
    for kind in GapKind:
        gap = Gap(kind=kind, node_ids=gap_holder.node_ids)
        assert engine.phrase(gap, graph).text.rstrip().endswith("?")


def test_phrasing_records_whether_a_model_wrote_it() -> None:
    class _Writer:
        def phrase(self, gap: Gap, default: str, quotes: list[str]) -> str:
            return "What is the strongest version of the opposing view?"

    graph = _graph([_thesis()])
    assert SocraticEngine().ask(graph)[0].phrasing is Phrasing.TEMPLATE
    assert SocraticEngine(_Writer()).ask(graph)[0].phrasing is Phrasing.MODEL


def test_a_writer_that_asserts_instead_of_asking_is_refused() -> None:
    """The machine supplies pressure, not content. A "question" that makes a
    claim is content entering through the author's channel.
    """

    class _Asserter:
        def phrase(self, gap: Gap, default: str, quotes: list[str]) -> str:
            return "The answer is that weights are plainly expressive."

    graph = _graph([_thesis()])
    question = SocraticEngine(_Asserter()).ask(graph)[0]
    assert question.phrasing is Phrasing.TEMPLATE
    assert "plainly expressive" not in question.text


def test_a_broken_writer_costs_wording_not_the_question() -> None:
    class _Broken:
        def phrase(self, gap: Gap, default: str, quotes: list[str]) -> str:
            raise RuntimeError("model unavailable")

    graph = _graph([_thesis()])
    question = SocraticEngine(_Broken()).ask(graph)[0]
    assert question.phrasing is Phrasing.TEMPLATE
    assert question.text.rstrip().endswith("?")


def test_a_long_claim_is_excerpted_not_dumped() -> None:
    graph = _graph([_thesis("word " * 200)])
    assert len(SocraticEngine().ask(graph)[0].text) < 400
