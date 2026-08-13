"""Novelty gate. Offline: no models, no network."""

from __future__ import annotations

import pytest

from modules.maieutic.graph import ArgumentGraph, GraphPatch, Node, NodeType
from modules.maieutic.novelty import (
    LexicalEmbedder,
    Method,
    NoveltyGate,
    Verdict,
    cosine,
)


class _StubEmbedder:
    """Returns vectors from an explicit map, so similarity is exact in tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    @property
    def method(self) -> Method:
        return Method.SEMANTIC

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


class _Critic:
    def __init__(self, label: str, source: str = "model") -> None:
        self.label = label
        self.source = source
        self.calls: list[tuple[str, str]] = []

    def relate(self, premise: str, hypothesis: str) -> tuple[str, str]:
        self.calls.append((premise, hypothesis))
        return self.label, self.source


def _node(text: str) -> Node:
    return Node.propose(NodeType.ORIGINAL, text)


# --------------------------------------------------------------------------- #
# The three bands
# --------------------------------------------------------------------------- #
def test_a_near_identical_node_is_rejected_as_restatement() -> None:
    existing = _node("The published exclusion does constitutional-avoidance work.")
    candidate = _node("Constitutional avoidance is the work the published exclusion does.")
    gate = NoveltyGate(
        _StubEmbedder({existing.text: [1.0, 0.0], candidate.text: [0.99, 0.14]})
    )
    v = gate.assess(candidate, [existing])
    assert v.verdict is Verdict.RESTATEMENT
    assert not v.accepted
    assert v.nearest_id == existing.id
    assert "similarity" in v.reason


def test_a_distant_node_is_novel_without_troubling_the_critic() -> None:
    existing = _node("The published exclusion does constitutional-avoidance work.")
    candidate = _node("Congress has occupied the field through express preemption.")
    critic = _Critic("entailment")  # would reject, but must not be consulted
    gate = NoveltyGate(
        _StubEmbedder({existing.text: [1.0, 0.0], candidate.text: [0.0, 1.0]}), critic
    )
    v = gate.assess(candidate, [existing])
    assert v.verdict is Verdict.NOVEL
    assert not critic.calls, "a distant node needs no adjudication"


def test_in_the_soft_band_entailment_decides_not_wording() -> None:
    """Different words are not a contribution if the graph already entails them."""
    existing = _node("Weights are expressive because they encode human choices.")
    candidate = _node("Because they encode human choices, weights are expressive.")
    vectors = {existing.text: [1.0, 0.0], candidate.text: [0.8, 0.6]}

    entailed = NoveltyGate(_StubEmbedder(vectors), _Critic("entailment"))
    v = entailed.assess(candidate, [existing])
    assert v.verdict is Verdict.ENTAILED
    assert v.critic_source == "model"

    contradicts = NoveltyGate(_StubEmbedder(vectors), _Critic("contradiction"))
    assert contradicts.assess(candidate, [existing]).verdict is Verdict.NOVEL


def test_the_soft_band_without_a_critic_admits_but_says_so() -> None:
    """Admitting the question was not adjudicated beats a confident guess."""
    existing = _node("A premise.")
    candidate = _node("A closely related premise.")
    gate = NoveltyGate(
        _StubEmbedder({existing.text: [1.0, 0.0], candidate.text: [0.8, 0.6]})
    )
    v = gate.assess(candidate, [existing])
    assert v.verdict is Verdict.NOVEL
    assert "unadjudicated" in v.reason


def test_the_first_node_cannot_restate_anything() -> None:
    gate = NoveltyGate(LexicalEmbedder())
    v = gate.assess(_node("The opening thesis."), [])
    assert v.verdict is Verdict.NOVEL
    assert v.similarity == 0.0


# --------------------------------------------------------------------------- #
# A degraded gate must be legible
# --------------------------------------------------------------------------- #
def test_every_verdict_records_the_method_that_produced_it() -> None:
    """Lexical overlap cannot see a paraphrase -- the very thing this catches.

    A run that degrades to it is weaker than it looks, so the verdict says which
    it was rather than leaving the reader to assume the stronger one.
    """
    existing = _node("The rule applies to public officials.")
    candidate = _node("The doctrine governs government employees.")

    lexical = NoveltyGate(LexicalEmbedder()).assess(candidate, [existing])
    assert lexical.method is Method.LEXICAL
    # It scores these as unrelated despite being near-paraphrases; that is the
    # limitation, stated rather than hidden.
    assert lexical.verdict is Verdict.NOVEL

    semantic = NoveltyGate(
        _StubEmbedder({existing.text: [1.0, 0.0], candidate.text: [0.98, 0.2]})
    ).assess(candidate, [existing])
    assert semantic.method is Method.SEMANTIC
    assert semantic.verdict is Verdict.RESTATEMENT


def test_lexical_embedder_still_catches_a_literal_repeat() -> None:
    text = "Publication of weights is not a deemed export under the regulation."
    gate = NoveltyGate(LexicalEmbedder())
    assert gate.assess(_node(text), [_node(text)]).verdict is Verdict.RESTATEMENT


# --------------------------------------------------------------------------- #
# Patch-level behaviour
# --------------------------------------------------------------------------- #
def test_a_patch_cannot_smuggle_a_restatement_beside_something_new() -> None:
    """Candidates are compared to accepted siblings, not only to the graph."""
    graph = ArgumentGraph()
    seed = _node("Seed claim.")
    graph.apply(GraphPatch(nodes=[seed]))

    fresh = _node("A genuinely different line of argument.")
    echo = _node("A genuinely different line of argument, restated.")
    gate = NoveltyGate(
        _StubEmbedder({
            seed.text: [1.0, 0.0, 0.0],
            fresh.text: [0.0, 1.0, 0.0],
            echo.text: [0.0, 0.99, 0.14],
        })
    )
    result = gate.assess_patch(GraphPatch(nodes=[fresh, echo]), graph)
    verdicts = {v.node_id: v.verdict for v in result.verdicts}
    assert verdicts[fresh.id] is Verdict.NOVEL
    assert verdicts[echo.id] is Verdict.RESTATEMENT
    assert result.passed, "the patch still contributes something"
    assert result.delta == pytest.approx(0.5)


def test_a_patch_of_pure_restatement_fails() -> None:
    graph = ArgumentGraph()
    seed = _node("Seed claim.")
    graph.apply(GraphPatch(nodes=[seed]))
    echo = _node("Seed claim.")
    gate = NoveltyGate(LexicalEmbedder())
    result = gate.assess_patch(GraphPatch(nodes=[echo]), graph)
    assert not result.passed
    assert result.delta == 0.0


def test_delta_tracks_a_vein_being_mined_out() -> None:
    """A shrinking delta is the signal to ask elsewhere."""
    graph = ArgumentGraph()
    seed = _node("Seed.")
    graph.apply(GraphPatch(nodes=[seed]))
    gate = NoveltyGate(LexicalEmbedder())

    rich = gate.assess_patch(
        GraphPatch(nodes=[_node("Wholly distinct doctrinal ground here."),
                          _node("Another separate preemption argument entirely.")]),
        graph,
    )
    barren = gate.assess_patch(GraphPatch(nodes=[_node("Seed."), _node("Seed.")]), graph)
    assert rich.delta > barren.delta
    assert barren.delta == 0.0


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_cosine_handles_degenerate_input() -> None:
    assert cosine([], []) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        NoveltyGate(LexicalEmbedder(), hard=0.5, soft_low=0.9)


def test_nodes_already_in_the_graph_are_not_reassessed() -> None:
    graph = ArgumentGraph()
    seed = _node("Seed.")
    graph.apply(GraphPatch(nodes=[seed]))
    result = NoveltyGate(LexicalEmbedder()).assess_patch(
        GraphPatch(nodes=[seed, _node("Something new and quite unrelated.")]), graph
    )
    assert [v.node_id for v in result.verdicts] != [seed.id]
    assert len(result.verdicts) == 1
