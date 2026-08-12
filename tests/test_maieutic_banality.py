"""Banality gate. Offline: no models, no network."""

from __future__ import annotations

from modules.maieutic.banality import (
    BLOCKING,
    BanalityGate,
    Source,
    Verdict,
)
from modules.maieutic.graph import ArgumentGraph, GraphPatch, Node, NodeType
from modules.maieutic.novelty import Method


class _StubEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    @property
    def method(self) -> Method:
        return Method.SEMANTIC

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def _claim(text: str, section: str = "") -> Node:
    return Node.from_human(NodeType.ORIGINAL, text, section=section)


# --------------------------------------------------------------------------- #
# Vacuity: the only thing that blocks
# --------------------------------------------------------------------------- #
def test_a_claim_that_is_all_hedge_asserts_nothing_and_is_blocked() -> None:
    node = _claim("It may perhaps be that this could arguably seem possible.")
    result = BanalityGate().assess(node)
    assert result.verdict is Verdict.VACUOUS
    assert result.blocking


def test_a_hedged_claim_that_still_says_something_is_not_vacuous() -> None:
    node = _claim("Weights may arguably be expressive under the First Amendment.")
    assert BanalityGate().assess(node).verdict is not Verdict.VACUOUS


def test_vacuity_is_the_only_blocking_verdict() -> None:
    """The table is the invariant. Adding to it should be a deliberate edit."""
    assert set(BLOCKING) == {Verdict.VACUOUS}
    for verdict in Verdict:
        assert (verdict in BLOCKING) is (verdict is Verdict.VACUOUS)


# --------------------------------------------------------------------------- #
# Hedging reports, because legal writing hedges legitimately
# --------------------------------------------------------------------------- #
def test_a_mostly_hedged_claim_is_reported_but_merges() -> None:
    """"This may be the better view" is a real scholarly claim. A gate that
    blocked it would harm the manuscript it exists to protect.
    """
    node = _claim("This may perhaps arguably be the generally better view.")
    gate = BanalityGate()
    assessment = gate.assess(node)
    assert assessment.verdict is Verdict.HEDGED
    assert not assessment.blocking

    patch = GraphPatch(nodes=[node])
    assert gate.assess_patch(patch, ArgumentGraph()).passed


def test_a_direct_claim_is_substantive() -> None:
    node = _claim("Publishing model weights is not a deemed export.")
    assessment = BanalityGate().assess(node)
    assert assessment.verdict is Verdict.SUBSTANTIVE
    assert assessment.hedge_ratio == 0.0


# --------------------------------------------------------------------------- #
# Commonplace: new to the paper is not the same as new to the field
# --------------------------------------------------------------------------- #
def test_a_claim_the_sources_already_state_is_commonplace() -> None:
    """Novelty compares against the manuscript; this compares against the field."""
    node = _claim("Source code is protected expression.")
    source = Source("Bernstein", "Protected expression includes source code.")
    gate = BanalityGate(
        [source],
        _StubEmbedder({node.text: [1.0, 0.0], source.text: [0.99, 0.14]}),
    )
    assessment = gate.assess(node)
    assert assessment.verdict is Verdict.COMMONPLACE
    assert assessment.nearest_source == "Bernstein"
    assert not assessment.blocking


def test_a_claim_the_sources_do_not_state_survives_the_corpus_limb() -> None:
    node = _claim("Weights are expressive for the same reason.")
    source = Source("Unrelated", "Preemption doctrine governs conflicting statutes.")
    gate = BanalityGate(
        [source], _StubEmbedder({node.text: [1.0, 0.0], source.text: [0.0, 1.0]})
    )
    assert gate.assess(node).verdict is Verdict.SUBSTANTIVE


def test_without_a_corpus_no_claim_about_the_field_is_made() -> None:
    """Reporting a claim as original because nothing was checked would be a
    stronger statement than the gate is entitled to.
    """
    assessment = BanalityGate().assess(_claim("An entirely ordinary observation."))
    assert assessment.method is None
    assert assessment.similarity == 0.0


def test_a_commonplace_verdict_records_how_it_was_reached() -> None:
    """Lexical overlap cannot see a paraphrase, which is what a commonplace
    claim usually is. A degraded run must not read as a semantic one.
    """
    text = "Source code is protected expression."
    source = Source("Bernstein", text)
    lexical = BanalityGate([source]).assess(_claim(text))
    assert lexical.verdict is Verdict.COMMONPLACE
    assert lexical.method is Method.LEXICAL

    node = _claim("Expression protections extend to written software.")
    semantic = BanalityGate(
        [source], _StubEmbedder({node.text: [1.0, 0.0], source.text: [0.98, 0.2]})
    ).assess(node)
    assert semantic.method is Method.SEMANTIC


# --------------------------------------------------------------------------- #
# Only claim-bearing nodes are judged
# --------------------------------------------------------------------------- #
def test_background_material_is_supposed_to_be_unoriginal() -> None:
    """A paper needs commonplace premises. Judging them makes it unwritable."""
    text = "Source code is protected expression."
    gate = BanalityGate([Source("Bernstein", text)])
    for kind in (NodeType.PREMISE, NodeType.AUTHORITY, NodeType.OBJECTION, NodeType.REPLY):
        assessment = gate.assess(Node.propose(kind, text))
        assert assessment.verdict is Verdict.SUBSTANTIVE
        assert "no banality rule applies" in assessment.detail


def test_a_vacuous_premise_does_not_block_a_merge() -> None:
    node = Node.propose(NodeType.PREMISE, "It may perhaps possibly be so.")
    assert BanalityGate().assess_patch(GraphPatch(nodes=[node]), ArgumentGraph()).passed


def test_what_the_author_wrote_is_judged_whatever_role_it_plays() -> None:
    """The adapter never produces a THESIS or an ORIGINAL — an answer becomes a
    reply, an objection or a premise. A type-only rule made this gate
    unreachable in the assembled loop, so a vacuous answer merged unlooked-at.
    """
    for kind in (NodeType.REPLY, NodeType.OBJECTION, NodeType.PREMISE):
        node = Node.from_human(kind, "It may perhaps arguably seem possible.")
        assert BanalityGate().assess(node).verdict is Verdict.VACUOUS


def test_the_same_words_from_the_machine_are_not_judged() -> None:
    """Machine-proposed premises and objections are background and pressure.
    Both are supposed to be unoriginal.
    """
    node = Node.propose(NodeType.REPLY, "It may perhaps arguably seem possible.")
    assert BanalityGate().assess(node).verdict is Verdict.SUBSTANTIVE


def test_a_thesis_is_judged_like_an_original() -> None:
    node = Node.from_human(NodeType.THESIS, "It may perhaps arguably seem possible.")
    assert BanalityGate().assess(node).verdict is Verdict.VACUOUS


# --------------------------------------------------------------------------- #
# Patch level, and the signal it feeds
# --------------------------------------------------------------------------- #
def test_one_vacuous_claim_fails_the_patch() -> None:
    good = _claim("Publishing weights is not a deemed export.")
    empty = _claim("It could perhaps arguably seem so.")
    result = BanalityGate().assess_patch(GraphPatch(nodes=[good, empty]), ArgumentGraph())
    assert not result.passed
    assert [a.node_id for a in result.assessments if a.blocking] == [empty.id]


def test_advisory_verdicts_are_separated_from_substantive_ones() -> None:
    direct = _claim("Publishing weights is not a deemed export.")
    hedged = _claim("This may perhaps arguably be the generally better view.")
    result = BanalityGate().assess_patch(
        GraphPatch(nodes=[direct, hedged]), ArgumentGraph()
    )
    assert result.passed
    assert [a.node_id for a in result.advisory] == [hedged.id]
    assert [a.node_id for a in result.substantive] == [direct.id]


def test_a_section_producing_only_hedge_is_reported_as_barren() -> None:
    """Somewhere the author is producing only throat-clearing is somewhere to
    stop asking. Feeds `SocraticEngine.ask(avoid_sections=...)`.
    """
    thin = _claim("This may perhaps arguably be the generally better view.", "worked")
    alive = _claim("Publishing weights is not a deemed export.", "fresh")
    result = BanalityGate().assess_patch(GraphPatch(nodes=[thin, alive]), ArgumentGraph())
    assert result.barren_sections() == frozenset({"worked"})


def test_a_section_with_one_real_claim_is_not_barren() -> None:
    thin = _claim("This may perhaps arguably be the generally better view.", "s")
    alive = _claim("Publishing weights is not a deemed export.", "s")
    result = BanalityGate().assess_patch(GraphPatch(nodes=[thin, alive]), ArgumentGraph())
    assert result.barren_sections() == frozenset()


def test_nodes_already_in_the_graph_are_not_reassessed() -> None:
    graph = ArgumentGraph()
    existing = _claim("Already merged.")
    graph.apply(GraphPatch(nodes=[existing]))
    fresh = _claim("Publishing weights is not a deemed export.")
    result = BanalityGate().assess_patch(GraphPatch(nodes=[existing, fresh]), graph)
    assert [a.node_id for a in result.assessments] == [fresh.id]
