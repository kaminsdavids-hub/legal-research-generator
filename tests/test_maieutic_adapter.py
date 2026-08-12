"""The dialectic adapter. Offline: no models, no network."""

from __future__ import annotations

import pytest

from modules.dialectic.models import (
    NOT_OPERATIVE,
    CitationSlot,
    DialecticTurn,
    Position,
    SlotStatus,
)
from modules.maieutic.dialectic_adapter import AnswerRefused, adapt
from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    GraphPatch,
    Node,
    NodeType,
    Provenance,
)
from modules.maieutic.socratic import Gap, GapKind, Question


def _question(kind: GapKind = GapKind.UNCONTESTED_CLAIM, target: str = "n1") -> Question:
    return Question(gap=Gap(kind=kind, node_ids=(target,)), text="What about X?")


def _slot(text: str, **kwargs: object) -> CitationSlot:
    return CitationSlot(proposition=text, **kwargs)  # type: ignore[arg-type]


def _turn(
    thesis: list[CitationSlot] | None = None,
    antithesis: list[CitationSlot] | None = None,
    synthesis: str = "",
) -> DialecticTurn:
    return DialecticTurn(
        question="q",
        thesis=Position(
            side="thesis", model="m1", family="f1", propositions=thesis or []
        ),
        antithesis=Position(
            side="antithesis", model="m2", family="f2", propositions=antithesis or []
        ),
        synthesis=synthesis,
    )


def _typed(result: object, kind: NodeType) -> list[Node]:
    return [n for n in result.patch.nodes if n.type is kind]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# The author's answer
# --------------------------------------------------------------------------- #
def test_the_answer_is_captured_verbatim_and_is_human() -> None:
    answer = "Weights encode choices, and choosing is expressive."
    result = adapt(_question(), answer)
    node = next(n for n in result.patch.nodes if n.id == result.answer_id)
    assert node.text == answer
    assert node.provenance is Provenance.HUMAN


def test_nothing_the_machine_contributed_claims_human_provenance() -> None:
    result = adapt(
        _question(),
        "The author's answer.",
        _turn(thesis=[_slot("A supporting point.")], antithesis=[_slot("An objection.")]),
    )
    machine = [n for n in result.patch.nodes if n.id != result.answer_id]
    assert machine, "the exchange contributed nothing to check"
    assert all(n.provenance is Provenance.DIALECTIC for n in machine)


def test_an_empty_answer_is_declining_the_question_not_an_insertion() -> None:
    with pytest.raises(AnswerRefused):
        adapt(_question(), "   ")


def test_an_answer_alone_is_a_smaller_patch_not_an_invalid_one() -> None:
    result = adapt(_question(), "An answer with no exchange behind it.")
    assert len(result.patch.nodes) == 1
    assert result.patch.edges[0].target == "n1"


def test_the_question_travels_with_the_patch() -> None:
    """A patch whose rationale is lost cannot be audited later."""
    question = _question()
    assert adapt(question, "An answer.").patch.rationale == question.text


# --------------------------------------------------------------------------- #
# What the answer becomes depends on the gap it answers
# --------------------------------------------------------------------------- #
def test_a_reply_attaches_to_the_objection_not_to_what_it_attacked() -> None:
    """`unanswered_attacks` looks for a REPLY pointing at the OBJECTION.

    Attaching anywhere else leaves the attack reading as open forever.
    """
    thesis = Node.from_human(NodeType.THESIS, "The claim.")
    objection = Node.propose(NodeType.OBJECTION, "The objection.")
    graph = ArgumentGraph()
    graph.apply(
        GraphPatch(
            nodes=[thesis, objection],
            edges=[Edge(objection.id, thesis.id, EdgeType.ATTACKS)],
        )
    )
    question = Question(
        gap=Gap(GapKind.UNANSWERED_ATTACK, (objection.id, thesis.id)), text="Answer it?"
    )
    graph.apply(adapt(question, "Here is my reply.").patch)
    assert graph.unanswered_attacks() == []


def test_an_answer_to_an_uncontested_claim_attacks_it() -> None:
    """The author was asked for the strongest objection to their own claim."""
    result = adapt(_question(GapKind.UNCONTESTED_CLAIM), "The best counter-argument.")
    node = next(n for n in result.patch.nodes if n.id == result.answer_id)
    assert node.type is NodeType.OBJECTION
    assert result.patch.edges[0].type is EdgeType.ATTACKS


def test_an_answer_to_an_unsupported_claim_supports_it() -> None:
    result = adapt(_question(GapKind.UNSUPPORTED_CLAIM), "Because of this reason.")
    node = next(n for n in result.patch.nodes if n.id == result.answer_id)
    assert node.type is NodeType.PREMISE
    assert result.patch.edges[0].type is EdgeType.SUPPORTS


def test_every_gap_kind_can_absorb_an_answer() -> None:
    """A question the loop cannot take an answer to is a question that goes
    nowhere. Iterates GapKind, so a new kind fails until it has a role.
    """
    for kind in GapKind:
        result = adapt(_question(kind), "An answer.")
        assert result.patch.nodes


def test_a_gap_needing_an_edit_says_the_answer_did_not_close_it() -> None:
    """Breaking a DEPENDS_ON cycle removes an edge, and a patch only adds.

    The gap is still there after the answer, and the honest report says so
    rather than treating the question as resolved by having been answered.
    """
    assert adapt(_question(GapKind.SELF_GROUNDING), "A is prior.").unresolved
    assert not adapt(_question(GapKind.UNSUPPORTED_CLAIM), "Because.").unresolved


# --------------------------------------------------------------------------- #
# The synthesis
# --------------------------------------------------------------------------- #
def test_the_synthesis_never_enters_the_manuscript() -> None:
    """Thesis and antithesis are pressure. The synthesis is the machine writing
    the paper's conclusion, which is the one thing this system exists not to do.
    """
    synthesis = "On balance, the better view reconciles both sides as follows."
    result = adapt(_question(), "The answer.", _turn(synthesis=synthesis))
    assert result.synthesis == synthesis
    assert all(synthesis not in n.text for n in result.patch.nodes)


# --------------------------------------------------------------------------- #
# Authority is not created here
# --------------------------------------------------------------------------- #
def test_a_verified_slot_becomes_an_authority_carrying_its_citation() -> None:
    slot = _slot(
        "Source code is protected speech.",
        status=SlotStatus.VERIFIED,
        normalized_cite="176 F.3d 1132",
    )
    authorities = _typed(adapt(_question(), "The answer.", _turn(thesis=[slot])), NodeType.AUTHORITY)
    assert [n.citation for n in authorities] == ["176 F.3d 1132"]


def test_an_authority_arrives_unverified_for_the_grounding_gate_to_judge() -> None:
    """CourtListener answered whether the citation resolves. The grounding gate
    asks whether the source supports *this claim*, which is a different question.
    """
    slot = _slot(
        "A holding.", status=SlotStatus.VERIFIED, normalized_cite="176 F.3d 1132"
    )
    authorities = _typed(adapt(_question(), "The answer.", _turn(thesis=[slot])), NodeType.AUTHORITY)
    assert authorities and not any(n.verified for n in authorities)


def test_a_retrieved_candidate_is_not_authority() -> None:
    """`proposed` means a candidate awaiting lookup. Entering it as authority
    would let an unconfirmed cite be read off the manuscript as a real one.
    """
    slot = _slot(
        "A proposition.", status=SlotStatus.PROPOSED, normalized_cite="176 F.3d 1132"
    )
    result = adapt(_question(), "The answer.", _turn(thesis=[slot]))
    assert not _typed(result, NodeType.AUTHORITY)
    premises = _typed(result, NodeType.PREMISE)
    assert premises and all(not n.citation for n in premises)


def test_a_rescinded_authority_is_refused_rather_than_rendered_clean() -> None:
    """A repealed rule that verified is still repealed. `Node` has no note
    field, so the warning cannot travel with it, and merging it silently is
    exactly what the NOT_OPERATIVE marker exists to prevent.
    """
    slot = _slot(
        "The published exclusion applies.",
        status=SlotStatus.VERIFIED,
        normalized_cite="15 C.F.R. 734.7(c)",
        note=f"{NOT_OPERATIVE} (superseded): replaced in 2024",
    )
    result = adapt(_question(), "The answer.", _turn(thesis=[slot]))
    assert not _typed(result, NodeType.AUTHORITY)
    assert any(NOT_OPERATIVE in r for r in result.refused)


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #
def test_a_proposition_carrying_a_citation_string_is_refused() -> None:
    """Defence in depth: the engine voids such a turn, so this should never
    fire. The invariant it protects is the one keeping fabricated authority out.
    """
    result = adapt(
        _question(),
        "The answer.",
        _turn(thesis=[_slot("As held in 176 F.3d 1132, this follows.")]),
    )
    assert len(result.patch.nodes) == 1, "only the author's answer survived"
    assert result.refused and "citation string" in result.refused[0]


def test_a_refusal_is_recorded_not_silent() -> None:
    """A proposition that vanished without a record is indistinguishable from
    one the model never produced.
    """
    result = adapt(
        _question(), "The answer.", _turn(thesis=[_slot("See 5 U.S. 137 for this.")])
    )
    assert result.refused


def test_the_two_sides_land_on_opposite_relations() -> None:
    result = adapt(
        _question(),
        "The answer.",
        _turn(thesis=[_slot("Support.")], antithesis=[_slot("Objection.")]),
    )
    relations = {
        next(n for n in result.patch.nodes if n.id == e.source).text: e.type
        for e in result.patch.edges
        if e.source != result.answer_id
    }
    assert relations == {"Support.": EdgeType.SUPPORTS, "Objection.": EdgeType.ATTACKS}


def test_an_empty_proposition_is_skipped_without_a_node() -> None:
    result = adapt(_question(), "The answer.", _turn(thesis=[_slot("   ")]))
    assert len(result.patch.nodes) == 1


def test_the_patch_applies_cleanly_to_a_graph_containing_the_target() -> None:
    """The end of the loop: what comes out of here has to merge."""
    target = Node.from_human(NodeType.THESIS, "The claim.")
    graph = ArgumentGraph()
    graph.apply(GraphPatch(nodes=[target]))
    question = Question(gap=Gap(GapKind.UNCONTESTED_CLAIM, (target.id,)), text="Q?")
    result = adapt(
        question,
        "The strongest objection.",
        _turn(thesis=[_slot("Support.")], antithesis=[_slot("Counter.")]),
    )
    added = graph.apply(result.patch)
    assert len(added) == 3
    assert graph.is_connected(result.answer_id)
