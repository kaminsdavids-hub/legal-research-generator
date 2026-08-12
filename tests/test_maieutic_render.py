"""Rendering the graph as prose. Offline: no models, no network."""

from __future__ import annotations

from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    GraphPatch,
    Node,
    NodeType,
)
from modules.maieutic.render import Audience, order_nodes, render, sections


def _graph(nodes: list[Node], edges: list[Edge] | None = None) -> ArgumentGraph:
    graph = ArgumentGraph()
    graph.apply(GraphPatch(nodes=nodes, edges=edges or []))
    return graph


# --------------------------------------------------------------------------- #
# Non-repetition, by construction
# --------------------------------------------------------------------------- #
def test_a_node_reachable_by_two_paths_is_written_once() -> None:
    """The payoff the graph representation exists for."""
    a = Node.from_human(NodeType.ORIGINAL, "First claim.")
    b = Node.from_human(NodeType.ORIGINAL, "Second claim.")
    shared = Node.propose(NodeType.PREMISE, "The shared premise.")
    graph = _graph(
        [a, b, shared],
        [
            Edge(shared.id, a.id, EdgeType.SUPPORTS),
            Edge(shared.id, b.id, EdgeType.SUPPORTS),
        ],
    )
    result = render(graph)
    assert result.order.count(shared.id) == 1
    assert result.text.count("The shared premise.") == 1


def test_every_node_is_rendered_exactly_once() -> None:
    nodes = [Node.from_human(NodeType.ORIGINAL, f"Claim {i}.") for i in range(5)]
    result = render(_graph(nodes))
    assert sorted(result.order) == sorted(n.id for n in nodes)
    assert len(set(result.order)) == len(result.order)


def test_the_authors_words_are_not_rewritten() -> None:
    text = "Weights encode human choices, and choosing among expressions is expressive."
    assert text in render(_graph([Node.from_human(NodeType.ORIGINAL, text)])).text


# --------------------------------------------------------------------------- #
# Order is deterministic
# --------------------------------------------------------------------------- #
def test_the_same_graph_always_renders_the_same_way() -> None:
    """A renderer that shuffles equivalent nodes makes every draft diff
    unreadable.
    """
    nodes = [Node.from_human(NodeType.ORIGINAL, f"Claim {i}.") for i in range(6)]
    graph = _graph(nodes)
    assert render(graph).order == render(graph).order

    reloaded = ArgumentGraph.from_dict(graph.to_dict())
    assert render(reloaded).order == render(graph).order


def test_what_the_reader_must_accept_first_comes_first() -> None:
    ground = Node.from_human(NodeType.ORIGINAL, "The prior claim.")
    built = Node.from_human(NodeType.ORIGINAL, "The dependent claim.")
    graph = _graph([built, ground], [Edge(built.id, ground.id, EdgeType.DEPENDS_ON)])
    order = [n.id for n in order_nodes(graph, "Argument")]
    assert order.index(ground.id) < order.index(built.id)


def test_a_reply_is_written_directly_after_the_objection_it_answers() -> None:
    """An objection three paragraphs from its answer is how a paper looks like
    it has no answer.
    """
    thesis = Node.from_human(NodeType.THESIS, "The thesis.")
    objection = Node.propose(NodeType.OBJECTION, "The objection.")
    reply = Node.from_human(NodeType.REPLY, "The reply.")
    filler = Node.propose(NodeType.PREMISE, "Some other premise.")
    graph = _graph(
        [thesis, objection, reply, filler],
        [
            Edge(objection.id, thesis.id, EdgeType.ATTACKS),
            Edge(reply.id, objection.id, EdgeType.SUPPORTS),
            Edge(filler.id, thesis.id, EdgeType.SUPPORTS),
        ],
    )
    order = render(graph).order
    assert order.index(reply.id) == order.index(objection.id) + 1


def test_a_self_grounding_graph_renders_rather_than_hanging() -> None:
    """The coherence gate blocks patches that create a cycle, but a graph
    loaded from disk can carry one. Dropping the nodes silently would be worse.
    """
    a = Node.from_human(NodeType.ORIGINAL, "A.")
    b = Node.from_human(NodeType.ORIGINAL, "B.")
    graph = _graph(
        [a, b],
        [Edge(a.id, b.id, EdgeType.DEPENDS_ON), Edge(b.id, a.id, EdgeType.DEPENDS_ON)],
    )
    result = render(graph)
    assert sorted(result.order) == sorted([a.id, b.id])
    assert any("depends on itself" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Disclosures no audience can suppress
# --------------------------------------------------------------------------- #
def test_an_unverified_citation_never_renders_as_clean_authority() -> None:
    """A reader must not have to know which view they are reading to know what
    has been confirmed.
    """
    node = Node.propose(NodeType.AUTHORITY, "A holding.", citation="176 F.3d 1132")
    for audience in Audience:
        text = render(_graph([node]), audience).text
        assert "UNVERIFIED" in text


def test_a_verified_citation_renders_clean() -> None:
    node = Node.propose(
        NodeType.AUTHORITY, "A holding.", citation="176 F.3d 1132"
    ).as_verified("176 F.3d 1132")
    text = render(_graph([node])).text
    assert "176 F.3d 1132" in text
    assert "UNVERIFIED" not in text


def test_unanswered_objections_are_declared_not_buried() -> None:
    thesis = Node.from_human(NodeType.THESIS, "The thesis.")
    objection = Node.propose(NodeType.OBJECTION, "The unanswered objection.")
    graph = _graph([thesis, objection], [Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    result = render(graph)
    assert result.open_problems == ["The unanswered objection."]
    assert "Open problems" in result.text


def test_an_answered_objection_is_not_an_open_problem() -> None:
    thesis = Node.from_human(NodeType.THESIS, "The thesis.")
    objection = Node.propose(NodeType.OBJECTION, "The objection.")
    reply = Node.from_human(NodeType.REPLY, "The reply.")
    graph = _graph(
        [thesis, objection, reply],
        [
            Edge(objection.id, thesis.id, EdgeType.ATTACKS),
            Edge(reply.id, objection.id, EdgeType.SUPPORTS),
        ],
    )
    result = render(graph)
    assert result.open_problems == []
    assert "Open problems" not in result.text


def test_a_manuscript_with_no_authored_content_says_so() -> None:
    """A paper written entirely by the machine is a fact about the paper."""
    graph = _graph([Node.propose(NodeType.ORIGINAL, "Machine-proposed claim.")])
    assert any("written by the author" in w for w in render(graph).warnings)


def test_one_authored_node_is_enough_to_clear_that_warning() -> None:
    graph = _graph(
        [
            Node.propose(NodeType.ORIGINAL, "Machine-proposed."),
            Node.from_human(NodeType.ORIGINAL, "The author's own."),
        ]
    )
    assert not any("written by the author" in w for w in render(graph).warnings)


def test_unconfirmed_citations_are_counted_in_the_warnings() -> None:
    graph = _graph(
        [
            Node.propose(NodeType.AUTHORITY, "One.", citation="1 U.S. 1"),
            Node.propose(NodeType.AUTHORITY, "Two.", citation="2 U.S. 2"),
        ]
    )
    assert any("2 authority node(s)" in w for w in render(graph).warnings)


def test_a_clean_graph_carries_no_warnings_section() -> None:
    node = Node.from_human(NodeType.ORIGINAL, "A clean claim.")
    result = render(_graph([node]))
    assert result.warnings == []
    assert "Warnings" not in result.text


# --------------------------------------------------------------------------- #
# Provenance and sections
# --------------------------------------------------------------------------- #
def test_the_review_view_shows_what_the_author_wrote_and_what_they_accepted() -> None:
    authored = Node.from_human(NodeType.ORIGINAL, "The author's claim.")
    accepted = Node.propose(NodeType.PREMISE, "A proposed premise.")
    text = render(_graph([authored, accepted]), Audience.REVIEW).text
    assert "`[human]`" in text
    assert "`[dialectic]`" in text


def test_the_manuscript_view_does_not_annotate_provenance() -> None:
    node = Node.from_human(NodeType.ORIGINAL, "A claim.")
    assert "`[human]`" not in render(_graph([node])).text


def test_structural_roles_are_labelled_without_asserting_anything() -> None:
    thesis = Node.from_human(NodeType.THESIS, "The thesis.")
    objection = Node.propose(NodeType.OBJECTION, "The objection.")
    graph = _graph([thesis, objection], [Edge(objection.id, thesis.id, EdgeType.ATTACKS)])
    assert "*One objection.* The objection." in render(graph).text


def test_sections_are_rendered_as_headings_with_unsectioned_material_first() -> None:
    graph = _graph(
        [
            Node.from_human(NodeType.ORIGINAL, "Sectioned.", section="Doctrine"),
            Node.from_human(NodeType.ORIGINAL, "Unsectioned."),
        ]
    )
    assert sections(graph) == ["Argument", "Doctrine"]
    text = render(graph).text
    assert text.index("## Argument") < text.index("## Doctrine")


def test_a_node_belongs_to_exactly_one_section() -> None:
    graph = _graph(
        [
            Node.from_human(NodeType.ORIGINAL, "A.", section="One"),
            Node.from_human(NodeType.ORIGINAL, "B.", section="Two"),
        ]
    )
    result = render(graph)
    assert len(result.order) == 2
    assert result.text.count("A.") == 1


def test_an_empty_graph_renders_to_nothing_rather_than_failing() -> None:
    result = render(ArgumentGraph())
    assert result.order == []
    assert result.text.strip() == ""
