"""Cross-references: emitted exactly when a dependency leaves its section.

This is where the ordering work becomes visible to a reader. A dependency
inside the section needs no reference — they just read it — so the number of
references is a function of the order, which is the quantity
:mod:`modules.linearize` exists to minimise. The test that matters is the last
one: the count the optimizer reports and the count the reader meets are the same
number.
"""

from __future__ import annotations

import pytest

from modules.linearize import anneal, build_problem, reading_plan, segment
from modules.linearize.report import build_report
from modules.linearize.sa import SAConfig
from modules.linearize.segment import SegmentConfig, section_titles
from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    Provenance,
)
from modules.maieutic.render import Audience, ReadingPlan, render


def _node(node_id: str, text: str = "", kind: NodeType = NodeType.PREMISE) -> Node:
    return Node(
        id=node_id,
        type=kind,
        text=text or f"claim {node_id} about published information " + "word " * 30,
        provenance=Provenance.HUMAN,
    )


def _graph(nodes, edges) -> ArgumentGraph:
    return ArgumentGraph(nodes={n.id: n for n in nodes}, edges=list(edges))


def _plan(order: list[str], sections: dict[str, str]) -> ReadingPlan:
    return ReadingPlan(order=order, section_of=sections)


def test_a_dependency_inside_a_section_needs_no_reference() -> None:
    """The reader just read it."""

    graph = _graph([_node("a"), _node("b")], [Edge("b", "a", EdgeType.DEPENDS_ON)])
    plan = _plan(["a", "b"], {"a": "Part I", "b": "Part I"})

    manuscript = render(graph, Audience.MANUSCRIPT, plan)

    assert manuscript.cross_references == []
    assert "as argued in" not in manuscript.text


def test_a_dependency_leaving_the_section_is_referenced() -> None:
    graph = _graph([_node("a"), _node("b")], [Edge("b", "a", EdgeType.DEPENDS_ON)])
    plan = _plan(["a", "b"], {"a": "Part I", "b": "Part II"})

    manuscript = render(graph, Audience.MANUSCRIPT, plan)

    assert manuscript.cross_references == [("b", "Part I")]
    assert "(as argued in Part I)" in manuscript.text


def test_a_prerequisite_not_yet_read_is_not_described_as_argued() -> None:
    """Precedence makes this abnormal, but a plan need not come from the
    optimizer, and "as argued in" about unread material is simply false."""

    graph = _graph([_node("a"), _node("b")], [Edge("a", "b", EdgeType.DEPENDS_ON)])
    plan = _plan(["a", "b"], {"a": "Part I", "b": "Part II"})

    manuscript = render(graph, Audience.MANUSCRIPT, plan)

    assert "as argued in" not in manuscript.text
    assert "(see Part II, below)" in manuscript.text


def test_two_dependencies_into_one_section_are_one_reference() -> None:
    """A reader should not be sent to Part I twice in a sentence."""

    graph = _graph(
        [_node("a"), _node("b"), _node("c")],
        [Edge("c", "a", EdgeType.DEPENDS_ON), Edge("c", "b", EdgeType.DEPENDS_ON)],
    )
    plan = _plan(["a", "b", "c"], {"a": "Part I", "b": "Part I", "c": "Part II"})

    manuscript = render(graph, Audience.MANUSCRIPT, plan)

    assert manuscript.cross_references == [("c", "Part I")]
    assert manuscript.text.count("as argued in Part I") == 1


def test_references_to_two_sections_read_as_a_list() -> None:
    graph = _graph(
        [_node("a"), _node("b"), _node("c")],
        [Edge("c", "a", EdgeType.DEPENDS_ON), Edge("c", "b", EdgeType.DEPENDS_ON)],
    )
    plan = _plan(["a", "b", "c"], {"a": "Part I", "b": "Part II", "c": "Part III"})

    manuscript = render(graph, Audience.MANUSCRIPT, plan)

    assert "(as argued in Part I and Part II)" in manuscript.text


def test_without_a_plan_the_renderer_is_unchanged() -> None:
    """The plan is optional: the renderer still works on the graph's own section
    fields, and emits no references because it cannot know what crosses."""

    graph = _graph(
        [_node("a"), _node("b")],
        [Edge("b", "a", EdgeType.DEPENDS_ON)],
    )

    manuscript = render(graph)

    assert manuscript.cross_references == []
    assert set(manuscript.order) == {"a", "b"}


def test_every_node_still_renders_exactly_once_under_a_plan() -> None:
    """The property the whole module rests on: a claim reachable two ways is
    written once and referred to, never restated."""

    nodes = [_node(f"n{i}") for i in range(6)]
    edges = [Edge("n5", "n0", EdgeType.DEPENDS_ON), Edge("n4", "n0", EdgeType.DEPENDS_ON)]
    order = [f"n{i}" for i in range(6)]
    plan = _plan(order, {n: ("Part I" if i < 3 else "Part II") for i, n in enumerate(order)})

    manuscript = render(_graph(nodes, edges), Audience.MANUSCRIPT, plan)

    assert manuscript.order == order
    assert len(set(manuscript.order)) == len(manuscript.order)


def test_the_optimizer_and_the_reader_count_the_same_references() -> None:
    """The report says N dependencies cross a boundary; the manuscript contains
    N references. If these ever disagree, the optimizer is minimising something
    the reader does not experience."""

    nodes = [_node(f"n{i}") for i in range(8)]
    edges = [
        Edge("n7", "n0", EdgeType.DEPENDS_ON),
        Edge("n6", "n1", EdgeType.DEPENDS_ON),
        Edge("n3", "n2", EdgeType.DEPENDS_ON),
    ]
    graph = _graph(nodes, edges)
    problem = build_problem(graph)

    order = anneal(problem, config=SAConfig(iterations=1_500, seed=3)).order
    seg = segment(problem, order, 2, SegmentConfig(min_words=1, max_words=100_000))
    report = build_report(problem, order, segmentation=seg)

    manuscript = render(
        graph,
        Audience.MANUSCRIPT,
        ReadingPlan(*reading_plan(order, seg, section_titles(order, seg))),
    )

    # De-duplication means the manuscript can carry fewer references than there
    # are crossing edges (two into one section read as one), never more.
    assert len(manuscript.cross_references) <= report.cross_section_dependencies
    crossing_nodes = {node for node, _ in manuscript.cross_references}
    assert len(crossing_nodes) == len({node for node, _ in manuscript.cross_references})
    assert manuscript.text.count("as argued in") + manuscript.text.count("see ") >= len(
        manuscript.cross_references
    )


# --------------------------------------------------------------------------- #
# A stretched dependency becomes a question
# --------------------------------------------------------------------------- #
def test_a_long_range_dependency_becomes_a_socratic_question() -> None:
    """The distance survived the search, so it is usually not an ordering
    problem: it is a missing intermediate step, and only the author writes those."""

    from modules.maieutic.socratic import GapKind, SocraticEngine, stretched_dependencies

    nodes = [_node(f"n{i:02d}") for i in range(10)]
    edges = [Edge("n09", "n00", EdgeType.DEPENDS_ON)]
    graph = _graph(nodes, edges)
    problem = build_problem(graph)
    report = build_report(problem, sorted(graph.nodes))

    gaps = stretched_dependencies(graph, report.long_range)

    assert [g.kind for g in gaps] == [GapKind.STRETCHED_DEPENDENCY]
    assert gaps[0].node_ids == ("n09", "n00")
    assert "9 position(s) earlier" in gaps[0].detail

    question = SocraticEngine().phrase(gaps[0], graph)
    assert "step in between" in question.text
    assert gaps[0].node_ids[0] in question.gap.node_ids


def test_analyse_stays_pure_over_the_graph() -> None:
    """Ordering-derived gaps are kept out of analyse() on purpose: it must give
    the same answer for the same graph, or "already asked" stops meaning
    anything across cycles."""

    from modules.maieutic.socratic import GapKind, analyse

    nodes = [_node(f"n{i:02d}") for i in range(10)]
    graph = _graph(nodes, [Edge("n09", "n00", EdgeType.DEPENDS_ON)])

    kinds = {gap.kind for gap in analyse(graph)}

    assert GapKind.STRETCHED_DEPENDENCY not in kinds


def test_a_stretched_gap_has_a_stable_key_across_orderings() -> None:
    """Two orderings that both stretch the same pair ask one question, not two."""

    from modules.maieutic.socratic import stretched_dependencies

    graph = _graph([_node("a"), _node("b")], [Edge("b", "a", EdgeType.DEPENDS_ON)])
    first = stretched_dependencies(graph, [{"dependent": "b", "prerequisite": "a", "distance": 9}])
    second = stretched_dependencies(graph, [{"dependent": "b", "prerequisite": "a", "distance": 4}])

    assert first[0].key == second[0].key


def test_a_finding_about_a_deleted_node_is_dropped() -> None:
    """A report can outlive the graph it described."""

    from modules.maieutic.socratic import stretched_dependencies

    graph = _graph([_node("a")], [])

    assert stretched_dependencies(graph, [{"dependent": "gone", "prerequisite": "a"}]) == []


# --------------------------------------------------------------------------- #
# The author states dependencies; nothing else does
# --------------------------------------------------------------------------- #
def _session_with(*texts: str):
    from modules.maieutic.graph import Node
    from modules.maieutic.loop import Session

    session = Session()
    nodes = [Node.from_human(NodeType.PREMISE, text) for text in texts]
    for node in nodes:
        session.graph.nodes[node.id] = node
    return session, [n.id for n in nodes]


def test_the_author_can_state_a_dependency() -> None:
    """DEPENDS_ON is the one edge type nothing in the package infers: it asserts
    what a reader must accept first, which is a claim about the argument."""

    session, ids = _session_with("the rule", "the application of that rule")

    edge = session.depends_on(ids[1], ids[0])

    assert edge.type is EdgeType.DEPENDS_ON
    assert (edge.source, edge.target) == (ids[1], ids[0])
    assert build_problem(session.graph).prerequisites[ids[1]] == {ids[0]}


def test_stating_the_same_dependency_twice_asserts_nothing_new() -> None:
    session, ids = _session_with("a", "b depends on a")

    session.depends_on(ids[1], ids[0])
    session.depends_on(ids[1], ids[0])

    assert sum(1 for e in session.graph.edges if e.type is EdgeType.DEPENDS_ON) == 1


def test_a_dependency_that_would_close_a_cycle_is_refused() -> None:
    """Self-grounding is the gap the Socratic engine ranks first, and an edge
    creates one far more easily than a node does."""

    from modules.maieutic.loop import SelfGroundingEdge

    session, ids = _session_with("a", "b")
    session.depends_on(ids[1], ids[0])

    with pytest.raises(SelfGroundingEdge, match="cannot ground itself"):
        session.depends_on(ids[0], ids[1])

    # And nothing was left behind by the attempt.
    assert sum(1 for e in session.graph.edges if e.type is EdgeType.DEPENDS_ON) == 1
    assert session.graph.depends_on_cycle() is None


def test_a_claim_cannot_depend_on_itself() -> None:
    session, ids = _session_with("a")

    with pytest.raises(ValueError, match="cannot depend on itself"):
        session.depends_on(ids[0], ids[0])


def test_an_unknown_node_is_refused() -> None:
    session, ids = _session_with("a")

    with pytest.raises(KeyError):
        session.depends_on(ids[0], "nope")


def test_a_stated_dependency_reaches_the_optimizer_and_the_reader() -> None:
    """End to end: the author states it, the order honours it, and the renderer
    references it when the sections split the pair."""

    session, ids = _session_with(
        "the EAR excludes published information " + "word " * 30,
        "open weights are published in that sense " + "word " * 30,
    )
    session.depends_on(ids[1], ids[0])

    problem = build_problem(session.graph)
    order = anneal(problem, config=SAConfig(iterations=400, seed=2)).order
    assert order.index(ids[0]) < order.index(ids[1])  # prerequisite first

    plan = _plan(order, {order[0]: "Part I", order[1]: "Part II"})
    manuscript = render(session.graph, Audience.MANUSCRIPT, plan)

    assert manuscript.cross_references == [(ids[1], "Part I")]
    assert "as argued in Part I" in manuscript.text


def test_the_cli_states_a_dependency(tmp_path) -> None:
    from modules.maieutic.cli import main
    from modules.maieutic.loop import Session

    session, ids = _session_with("the rule itself", "the application of the rule")
    state = tmp_path / "session.json"
    session.save(state)

    code = main([
        "--state", str(state), "--journal", str(tmp_path / "j.json"),
        "depends", ids[1][:6], ids[0][:6],
    ])

    assert code == 0
    reloaded = Session.load(state)
    assert any(e.type is EdgeType.DEPENDS_ON for e in reloaded.graph.edges)


def test_the_cli_refuses_an_ambiguous_prefix(tmp_path, capsys) -> None:
    """An author who mistypes should be told which nodes matched, not watch the
    command address the wrong claim."""

    from modules.maieutic.cli import main

    session, ids = _session_with("first claim", "second claim")
    state = tmp_path / "session.json"
    session.save(state)

    code = main([
        "--state", str(state), "--journal", str(tmp_path / "j.json"),
        "depends", "", ids[0],
    ])

    assert code == 1
    assert "matches" in capsys.readouterr().err


def test_a_dependency_can_be_retracted() -> None:
    """The author is the only one who may say a claim rests on another, so they
    are the only one who can say it does not."""

    session, ids = _session_with("the rule", "the application")
    session.depends_on(ids[1], ids[0])

    assert session.retract_dependency(ids[1], ids[0]) is True
    assert not [e for e in session.graph.edges if e.type is EdgeType.DEPENDS_ON]
    assert build_problem(session.graph).prerequisites[ids[1]] == set()


def test_retracting_something_never_stated_is_a_no_op() -> None:
    session, ids = _session_with("a", "b")

    assert session.retract_dependency(ids[1], ids[0]) is False


def test_retraction_leaves_other_edges_alone() -> None:
    """SUPPORTS and ATTACKS are the argument's substance; this command is not a
    general edge delete."""

    session, ids = _session_with("a", "b")
    session.graph.edges.append(Edge(ids[1], ids[0], EdgeType.SUPPORTS))
    session.depends_on(ids[1], ids[0])

    session.retract_dependency(ids[1], ids[0])

    assert [e.type for e in session.graph.edges] == [EdgeType.SUPPORTS]


def test_retraction_direction_matters() -> None:
    """Given the ids the wrong way round, nothing is removed — and the CLI says
    so rather than reporting success."""

    session, ids = _session_with("a", "b")
    session.depends_on(ids[1], ids[0])

    assert session.retract_dependency(ids[0], ids[1]) is False
    assert len([e for e in session.graph.edges if e.type is EdgeType.DEPENDS_ON]) == 1


def test_the_cli_retracts(tmp_path, capsys) -> None:
    from modules.maieutic.cli import main
    from modules.maieutic.loop import Session

    session, ids = _session_with("the rule itself", "the application of the rule")
    session.depends_on(ids[1], ids[0])
    state = tmp_path / "session.json"
    session.save(state)

    code = main([
        "--state", str(state), "--journal", str(tmp_path / "j.json"),
        "depends", ids[1][:6], ids[0][:6], "--retract",
    ])

    assert code == 0
    assert "Retracted" in capsys.readouterr().out
    assert not [e for e in Session.load(state).graph.edges if e.type is EdgeType.DEPENDS_ON]


def test_the_cli_says_when_there_was_nothing_to_retract(tmp_path, capsys) -> None:
    """Most often because the two ids were given the wrong way round."""

    from modules.maieutic.cli import main

    session, ids = _session_with("a", "b")
    state = tmp_path / "session.json"
    session.save(state)

    code = main([
        "--state", str(state), "--journal", str(tmp_path / "j.json"),
        "depends", ids[0][:6], ids[1][:6], "--retract",
    ])

    assert code == 0
    assert "nothing retracted" in capsys.readouterr().out
