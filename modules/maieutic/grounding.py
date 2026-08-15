"""The grounding gate: every node is verified-cited or explicitly argued.

There is no third category. A node either rests on an authority that retrieval
resolved, or it is the author's own position with an argument under it. What
cannot merge is the thing in between — an assertion with the confidence of a
citation and neither the citation nor the argument.

The two rules differ because the failure modes differ:

* **AUTHORITY** nodes fabricate. The generator never certifies them: the
  citation must resolve through retrieval and verification, exactly as in the
  dialectic module, where a debater is forbidden from emitting a citation at
  all and authority reaches a slot only by being looked up.
* **ORIGINAL** nodes float. The author needs no permission to assert a position,
  but a position with nothing under it is an opinion, not scholarship. So an
  ORIGINAL node merges only with an argument subgraph — something SUPPORTS-ing
  it, or an authority it rests on.

The asymmetry is deliberate. Requiring citations of ORIGINAL nodes would push
the author toward saying only what someone else has already said, which is the
opposite of what the loop is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .graph import ArgumentGraph, EdgeType, GraphPatch, Node, NodeType


class GroundingFailure(StrEnum):
    #: An AUTHORITY node with no citation at all.
    UNCITED_AUTHORITY = "uncited_authority"
    #: A citation that retrieval could not resolve.
    UNRESOLVED_CITATION = "unresolved_citation"
    #: A citation that resolved, but whose text does not support the claim.
    UNSUPPORTED_BY_SOURCE = "unsupported_by_source"
    #: An ORIGINAL node asserted with nothing under it.
    UNARGUED_ORIGINAL = "unargued_original"
    #: A generator tried to hand us a pre-verified authority.
    SELF_CERTIFIED = "self_certified"


class CitationVerifier(Protocol):
    """Resolves a citation and checks it supports the claim made of it."""

    def resolve(self, citation: str) -> tuple[bool, str]:
        """Return ``(resolved, detail)`` for *citation*."""
        ...

    def supports(self, citation: str, claim: str) -> tuple[bool, str]:
        """Return ``(supports, detail)`` for a resolved citation and a claim."""
        ...


@dataclass
class NodeGrounding:
    node_id: str
    passed: bool
    failure: GroundingFailure | None = None
    detail: str = ""
    #: Set when an AUTHORITY node resolved, so the caller can mark it verified.
    resolved_citation: str = ""


@dataclass
class PatchGrounding:
    results: list[NodeGrounding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Grounding is all-or-nothing: one ungrounded node fails the patch.

        Unlike novelty, where a patch survives if anything in it is new, a
        single fabricated or floating claim contaminates the merge. Letting the
        rest through would put it in the manuscript beside verified material,
        which is where an unsupported claim does the most damage.
        """
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failures(self) -> list[NodeGrounding]:
        return [r for r in self.results if not r.passed]


class GroundingGate:
    def __init__(self, verifier: CitationVerifier | None = None) -> None:
        self.verifier = verifier

    def assess(
        self, node: Node, patch: GraphPatch, graph: ArgumentGraph
    ) -> NodeGrounding:
        if node.type is NodeType.AUTHORITY:
            return self._assess_authority(node)
        if node.type is NodeType.ORIGINAL:
            return self._assess_original(node, patch, graph)
        # PREMISE, OBJECTION, REPLY and the rest carry the argument's structure
        # rather than its claims to truth; coherence governs them.
        return NodeGrounding(node_id=node.id, passed=True, detail="no grounding rule applies")

    def _assess_authority(self, node: Node) -> NodeGrounding:
        if node.verified:
            # `as_verified` is the grounding gate's own call. A node arriving
            # already marked came from somewhere else, and a generator that can
            # self-certify makes this gate decorative.
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.SELF_CERTIFIED,
                detail="arrived pre-verified; only this gate may mark an authority verified",
            )
        if not node.citation.strip():
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.UNCITED_AUTHORITY,
                detail="an AUTHORITY node must carry a citation",
            )
        if self.verifier is None:
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.UNRESOLVED_CITATION,
                detail=(
                    "no verifier configured; an authority cannot merge on trust. "
                    "Failing closed here is deliberate: this is the fabrication wall."
                ),
            )

        resolved, detail = self.verifier.resolve(node.citation)
        if not resolved:
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.UNRESOLVED_CITATION,
                detail=f"{node.citation}: {detail}",
            )

        supports, why = self.verifier.supports(node.citation, node.text)
        if not supports:
            # The subtler fabrication: a real case cited for something it does
            # not say. Resolving is necessary and not sufficient.
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.UNSUPPORTED_BY_SOURCE,
                detail=f"{node.citation} resolves but does not support this claim: {why}",
            )
        return NodeGrounding(
            node_id=node.id, passed=True, detail=why, resolved_citation=node.citation
        )

    def _assess_original(
        self, node: Node, patch: GraphPatch, graph: ArgumentGraph
    ) -> NodeGrounding:
        """An ORIGINAL node needs an argument, not a citation."""
        supporting = {
            e.source
            for e in [*patch.edges, *graph.edges]
            if e.target == node.id
            and e.type in (EdgeType.SUPPORTS, EdgeType.IMPLIES, EdgeType.QUALIFIES)
        }
        if not supporting:
            return NodeGrounding(
                node_id=node.id,
                passed=False,
                failure=GroundingFailure.UNARGUED_ORIGINAL,
                detail=(
                    "an ORIGINAL node needs no citation but cannot stand alone; "
                    "give it a premise, an authority, or an implication"
                ),
            )
        return NodeGrounding(
            node_id=node.id,
            passed=True,
            detail=f"argued by {len(supporting)} node(s)",
        )

    def assess_patch(self, patch: GraphPatch, graph: ArgumentGraph) -> PatchGrounding:
        result = PatchGrounding()
        for node in patch.nodes:
            if node.id in graph.nodes:
                continue
            result.results.append(self.assess(node, patch, graph))
        return result
