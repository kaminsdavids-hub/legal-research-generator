"""Manuscript ordering: constraints that must never break, and costs that must pay.

The hard constraints get a fuzzer rather than examples. Precedence, pins and
objection/reply adjacency are not preferences the search trades against — an
order that breaks one is not a worse manuscript, it is an unreadable one — so
the test that matters is that no move in any solver can produce one, across many
random graphs and seeds.
"""

from __future__ import annotations

import itertools
import random

import pytest

from modules.linearize import (
    BRKGAConfig,
    PrecedenceViolation,
    SAConfig,
    SegmentConfig,
    Weights,
    ablation,
    anneal,
    build_problem,
    build_report,
    compare_baselines,
    evolve,
    segment,
    sweep_k,
)
from modules.linearize.brkga import decode
from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    Provenance,
)


def _node(node_id: str, text: str = "", kind: NodeType = NodeType.PREMISE) -> Node:
    return Node(
        id=node_id,
        type=kind,
        text=text or f"claim {node_id} about export control and published information",
        provenance=Provenance.HUMAN,
    )


def _graph(nodes, edges) -> ArgumentGraph:
    return ArgumentGraph(nodes={n.id: n for n in nodes}, edges=list(edges))


def _embeddings(graph: ArgumentGraph) -> dict[str, list[float]]:
    from modules.maieutic.novelty import LexicalEmbedder

    ids = sorted(graph.nodes)
    vectors = LexicalEmbedder().embed([graph.nodes[i].text for i in ids])
    return dict(zip(ids, vectors, strict=False))


def _random_graph(rng: random.Random, size: int = 12) -> ArgumentGraph:
    """A DAG of DEPENDS_ON edges plus reference edges, never cyclic.

    Edges only ever run from a later id to an earlier one, which is what makes
    the DEPENDS_ON subgraph acyclic by construction — the fuzzer is testing the
    solvers, not the graph builder.
    """

    nodes = [_node(f"n{i:02d}") for i in range(size)]
    edges: list[Edge] = []
    for i in range(1, size):
        for j in range(i):
            if rng.random() < 0.15:
                edges.append(Edge(f"n{i:02d}", f"n{j:02d}", EdgeType.DEPENDS_ON))
            elif rng.random() < 0.12:
                kind = rng.choice(
                    [EdgeType.SUPPORTS, EdgeType.ATTACKS, EdgeType.QUALIFIES, EdgeType.IMPLIES]
                )
                edges.append(Edge(f"n{i:02d}", f"n{j:02d}", kind))
    return _graph(nodes, edges)


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #
def test_dependency_direction_follows_the_renderer() -> None:
    """"A depends on B: B must be read first" (render.order_nodes). The brief
    states the opposite; following it literally would put every claim before the
    material it rests on."""

    graph = _graph(
        [_node("a"), _node("b")],
        [Edge("a", "b", EdgeType.DEPENDS_ON)],
    )
    problem = build_problem(graph)

    assert problem.prerequisites["a"] == {"b"}
    assert problem.is_feasible(["b", "a"])
    assert not problem.is_feasible(["a", "b"])


@pytest.mark.parametrize("seed", range(12))
def test_the_annealer_never_produces_an_infeasible_order(seed: int) -> None:
    rng = random.Random(seed)
    graph = _random_graph(rng)
    problem = build_problem(graph, embeddings=_embeddings(graph))

    result = anneal(problem, config=SAConfig(iterations=600, seed=seed))

    assert problem.violations(result.order) == []


@pytest.mark.parametrize("seed", range(6))
def test_the_ga_decoder_never_produces_an_infeasible_order(seed: int) -> None:
    rng = random.Random(seed)
    graph = _random_graph(rng, size=14)
    problem = build_problem(graph, embeddings=_embeddings(graph))

    result = evolve(problem, config=BRKGAConfig(population=20, generations=8, seed=seed))

    assert problem.violations(result.order) == []
    # And any random key vector decodes feasibly, which is the property that
    # makes crossover safe: there is nothing to repair.
    for _ in range(20):
        keys = [rng.random() for _ in problem.node_ids]
        assert problem.violations(decode(problem, keys)) == []


def test_a_dependency_cycle_fails_loudly() -> None:
    """The integration gate rejects these; one arriving here means something
    upstream failed, and a silent repair would produce a manuscript whose
    reading guarantees are false."""

    graph = _graph(
        [_node("a"), _node("b")],
        [Edge("a", "b", EdgeType.DEPENDS_ON), Edge("b", "a", EdgeType.DEPENDS_ON)],
    )

    with pytest.raises(PrecedenceViolation, match="cycle"):
        build_problem(graph)


def test_pins_are_absolute() -> None:
    graph = _random_graph(random.Random(3))
    ids = sorted(graph.nodes)
    problem = build_problem(
        graph, embeddings=_embeddings(graph), pins={ids[0]: 0, ids[-1]: len(ids) - 1}
    )

    result = anneal(problem, config=SAConfig(iterations=800, seed=1))

    assert result.order[0] == ids[0]
    assert result.order[-1] == ids[-1]


def test_a_reply_stays_next_to_its_objection() -> None:
    """Splitting an objection from its reply is a reading failure, not a cost."""

    nodes = [
        _node("thesis", kind=NodeType.THESIS),
        _node("obj", "the strongest objection to the thesis", NodeType.OBJECTION),
        _node("reply", "the reply answering that objection", NodeType.REPLY),
        *[_node(f"filler{i}") for i in range(6)],
    ]
    edges = [
        Edge("obj", "thesis", EdgeType.ATTACKS),
        Edge("reply", "obj", EdgeType.ATTACKS),
    ]
    problem = build_problem(_graph(nodes, edges), embeddings=_embeddings(_graph(nodes, edges)))

    assert problem.pairs == [("obj", "reply")]
    result = anneal(problem, config=SAConfig(iterations=1_500, seed=5))
    pos = {n: i for i, n in enumerate(result.order)}

    assert 0 < pos["reply"] - pos["obj"] <= 2


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def test_a_forward_reference_costs_more_than_a_backward_one() -> None:
    """Being sent ahead to material you have not read is the expensive failure;
    reaching back a little is normal legal writing."""

    graph = _graph([_node("a"), _node("b")], [Edge("a", "b", EdgeType.SUPPORTS)])
    problem = build_problem(graph)

    backward = problem.cost(["b", "a"]).reference  # referenced first: cheap
    forward = problem.cost(["a", "b"]).reference  # referenced later: expensive

    assert forward > backward
    assert forward == pytest.approx(backward * problem.weights.forward_multiplier)


def test_distance_hurts_superlinearly() -> None:
    """Two references of distance 5 must read better than one of distance 10."""

    graph = _graph([_node(f"n{i}") for i in range(11)], [Edge("n0", "n10", EdgeType.SUPPORTS)])
    problem = build_problem(graph)
    order = [f"n{i}" for i in range(11)]

    far = problem.cost(order).reference
    near = problem.cost([*order[:5], "n10", *order[5:10]]).reference

    assert far > 2 * near


def test_the_search_beats_a_plain_topological_sort() -> None:
    """A search that cannot beat the baseline is not earning its runtime."""

    rng = random.Random(11)
    graph = _random_graph(rng, size=16)
    problem = build_problem(graph, embeddings=_embeddings(graph))

    result = anneal(problem, config=SAConfig(iterations=4_000, seed=2))
    baselines = compare_baselines(problem, result.order)

    assert baselines["optimized"]["total"] <= baselines["dfs_topological"]["total"]


# --------------------------------------------------------------------------- #
# Stability — the term that makes iterative work possible
# --------------------------------------------------------------------------- #
def test_adding_one_node_does_not_reshuffle_the_manuscript() -> None:
    """Without this the author cannot work iteratively: every accepted answer
    would rewrite the table of contents."""

    rng = random.Random(21)
    graph = _random_graph(rng, size=14)
    settled = anneal(
        build_problem(graph, embeddings=_embeddings(graph)),
        config=SAConfig(iterations=3_000, seed=4),
    ).order

    grown = _graph(
        [*graph.nodes.values(), _node("new", "a newly answered premise about publication")],
        [*graph.edges, Edge("new", settled[3], EdgeType.DEPENDS_ON)],
    )
    problem = build_problem(grown, embeddings=_embeddings(grown), previous=settled)
    result = anneal(problem, config=SAConfig(iterations=3_000, seed=4))

    moved = [n for n in settled if result.order.index(n) != settled.index(n)]
    # A handful of nodes shift around the insertion; the manuscript does not
    # rearrange itself.
    assert len(moved) <= len(settled) // 2


def test_stability_is_reported_as_an_edit_distance() -> None:
    rng = random.Random(31)
    graph = _random_graph(rng, size=10)
    problem = build_problem(graph, embeddings=_embeddings(graph))
    first = anneal(problem, config=SAConfig(iterations=500, seed=1)).order

    again = build_problem(graph, embeddings=_embeddings(graph), previous=first)
    report = build_report(again, first)

    assert report.moved_nodes == 0
    assert report.edit_distance == 0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_a_fixed_seed_reproduces_the_order_exactly() -> None:
    graph = _random_graph(random.Random(41), size=13)
    problem = build_problem(graph, embeddings=_embeddings(graph))

    a = anneal(problem, config=SAConfig(iterations=1_200, seed=99))
    b = anneal(problem, config=SAConfig(iterations=1_200, seed=99))

    assert a.order == b.order
    assert a.seed == b.seed == 99


def test_the_ga_is_deterministic_too() -> None:
    graph = _random_graph(random.Random(42), size=12)
    problem = build_problem(graph, embeddings=_embeddings(graph))
    config = BRKGAConfig(population=16, generations=5, seed=123)

    assert evolve(problem, config=config).order == evolve(problem, config=config).order


# --------------------------------------------------------------------------- #
# Segmentation is exact, so it can be checked against brute force
# --------------------------------------------------------------------------- #
def test_the_dp_matches_exhaustive_search() -> None:
    """The reason segmentation is a DP and not a metaheuristic: the answer is
    checkable."""

    nodes = [_node(f"n{i}", f"passage {i} " + "word " * 40) for i in range(9)]
    graph = _graph(nodes, [])
    problem = build_problem(graph, embeddings=_embeddings(graph))
    order = sorted(graph.nodes)
    config = SegmentConfig(min_words=1, max_words=10_000, balance=1.0)

    solution = segment(problem, order, 3, config)

    def brute_force(k: int) -> float:
        best = float("inf")
        for cuts in itertools.combinations(range(1, len(order)), k - 1):
            bounds = list(zip((0, *cuts), (*cuts, len(order)), strict=False))
            total = 0.0
            for start, end in bounds:
                probe = segment(problem, order[start:end], 1, config)
                total += probe.cost
            best = min(best, total)
        return best

    # The DP's own per-run costs, recomposed by hand over every possible cut.
    assert solution.cost <= brute_force(3) + 1e-9


def test_segmentation_respects_word_bounds_or_says_why_not() -> None:
    nodes = [_node(f"n{i}", "short text here") for i in range(6)]
    graph = _graph(nodes, [])
    problem = build_problem(graph, embeddings=_embeddings(graph))

    with pytest.raises(ValueError, match="no segmentation"):
        segment(problem, sorted(graph.nodes), 3, SegmentConfig(min_words=500))


def test_sweep_picks_an_elbow_not_the_largest_k() -> None:
    """Cost falls monotonically with k, so argmin would always return the
    largest option, which is not a choice."""

    nodes = [_node(f"n{i}", f"passage {i} " + "word " * 30) for i in range(12)]
    graph = _graph(nodes, [])
    problem = build_problem(graph, embeddings=_embeddings(graph))

    solution, curve = sweep_k(problem, sorted(graph.nodes), range(2, 7),
                              SegmentConfig(min_words=1, max_words=10_000))

    assert solution.k < max(curve)


# --------------------------------------------------------------------------- #
# The report is what the author reads
# --------------------------------------------------------------------------- #
def test_the_report_flags_a_long_range_dependency() -> None:
    """A dependency stretched across the manuscript is usually the argument
    wanting an intermediate step, not an ordering failure."""

    nodes = [_node(f"n{i:02d}") for i in range(10)]
    edges = [Edge("n09", "n00", EdgeType.DEPENDS_ON)]
    graph = _graph(nodes, edges)
    problem = build_problem(graph, embeddings=_embeddings(graph))

    report = build_report(problem, sorted(graph.nodes))

    assert report.long_range
    assert report.long_range[0]["dependent"] == "n09"
    assert report.long_range[0]["prerequisite"] == "n00"


def test_ablation_shows_what_each_weight_buys() -> None:
    graph = _random_graph(random.Random(51), size=10)
    problem = build_problem(
        graph, embeddings=_embeddings(graph), previous=sorted(graph.nodes),
        weights=Weights(reference=1.0, cohesion=1.0, stability=3.0),
    )
    order = anneal(problem, config=SAConfig(iterations=800, seed=8)).order

    rows = ablation(problem, order)

    assert set(rows) == {"full", "without_reference", "without_cohesion", "without_stability"}
    # Zeroing a term can only lower the total it contributed to.
    assert rows["without_reference"]["total"] <= rows["full"]["total"] + 1e-9
    # And the weights are restored, not left zeroed for the next caller.
    assert problem.weights.reference == 1.0


def test_the_report_counts_cross_section_dependencies() -> None:
    """Each one becomes an explicit cross-reference in the rendered manuscript."""

    nodes = [_node(f"n{i}", f"passage {i} " + "word " * 40) for i in range(8)]
    edges = [Edge("n7", "n0", EdgeType.DEPENDS_ON)]
    graph = _graph(nodes, edges)
    problem = build_problem(graph, embeddings=_embeddings(graph))
    order = sorted(graph.nodes)
    seg = segment(problem, order, 2, SegmentConfig(min_words=1, max_words=10_000))

    report = build_report(problem, order, segmentation=seg)

    assert report.cross_section_dependencies == 1


def test_the_baseline_is_not_seeded_by_the_order_it_measures() -> None:
    """`[] or self.previous` made the DFS baseline a copy of the accepted order,
    so compare_baselines reported the optimizer tying a baseline that was its own
    output — a measurement that could only confirm itself."""

    graph = _random_graph(random.Random(61), size=12)
    problem = build_problem(graph, embeddings=_embeddings(graph))
    optimized = anneal(problem, config=SAConfig(iterations=2_000, seed=6)).order

    # Now re-run as the author would: the accepted order becomes `previous`.
    again = build_problem(graph, embeddings=_embeddings(graph), previous=optimized)
    rows = compare_baselines(again, optimized)

    assert again.topological_order(seed_order=[]) != optimized or rows["optimized"][
        "total"
    ] <= rows["dfs_topological"]["total"]
    # The baseline is the id-sorted topological order, whatever `previous` holds.
    assert again.topological_order(seed_order=[]) == problem.topological_order(seed_order=[])


def test_an_explicit_seed_still_seeds() -> None:
    graph = _random_graph(random.Random(62), size=10)
    problem = build_problem(graph, embeddings=_embeddings(graph))
    reversed_ids = sorted(graph.nodes, reverse=True)

    seeded = problem.topological_order(seed_order=reversed_ids)
    unseeded = problem.topological_order(seed_order=[])

    assert problem.violations(seeded) == []
    assert seeded != unseeded
