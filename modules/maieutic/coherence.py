"""The coherence gate: does the graph still say something, after this merge?

Novelty asks whether a patch is new. Grounding asks whether it is supported.
Coherence asks the remaining question — whether what it adds fits together with
what is already there, or quietly breaks it.

The findings split into two severities, and the split is the point.

**Structural findings block.** A `DEPENDS_ON` cycle, an edge pointing at a node
that does not exist, a node asserting it both supports and attacks the same
claim, a reply replying to nothing. Each is decidable from the graph's shape
alone. No model is consulted and none can be talked around, which is what makes
them safe to merge on.

**Semantic findings are advisory.** Whether a proposition marked SUPPORTS in fact
contradicts what it supports is a question only an entailment critic can answer,
and this repository has already been burned by trusting one: an uncalibrated NLI
heuristic scored ``constitutional`` against ``unconstitutional`` as entailment
(REMEDIATION §9.5), and an uncalibrated judge rated pure failure placeholders
8.00/10. Letting a model's opinion block a merge would put the author's work at
the mercy of exactly that. So the critic reports, the report is surfaced, and the
merge proceeds. Promoting any of these to blocking is a calibration decision, not
a code change — see OBSERVABLES D13.

The gate never mutates the graph it is judging. Cycle detection runs against a
trial view of the merged result, because a gate with a side effect is a gate that
has already merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .graph import ArgumentGraph, Edge, EdgeType, GraphPatch, Node, NodeType
from .novelty import EntailmentCritic

#: Relations whose assertion the critic can check against the text.
_CHECKABLE = (EdgeType.SUPPORTS, EdgeType.ATTACKS)


class Severity(StrEnum):
    #: Decidable from the graph's shape. Blocks the merge.
    BLOCKING = "blocking"
    #: A model's opinion. Reported, never blocking.
    ADVISORY = "advisory"


class Incoherence(StrEnum):
    #: A DEPENDS_ON cycle: the argument grounds itself.
    SELF_GROUNDING = "self_grounding"
    #: An edge referencing a node in neither the patch nor the graph.
    DANGLING_EDGE = "dangling_edge"
    #: One node both supports and attacks the same target.
    CONTRADICTORY_RELATION = "contradictory_relation"
    #: A new node attaching to nothing in a graph that already has content.
    DISCONNECTED = "disconnected"
    #: A REPLY that replies to nothing.
    REPLY_TO_NOTHING = "reply_to_nothing"
    #: The critic says this SUPPORTS edge's source contradicts its target.
    SUPPORT_THAT_CONTRADICTS = "support_that_contradicts"
    #: The critic says this ATTACKS edge's source entails its target: it agrees
    #: with the thing it is marked as attacking.
    ATTACK_THAT_AGREES = "attack_that_agrees"


_SEVERITY: dict[Incoherence, Severity] = {
    Incoherence.SELF_GROUNDING: Severity.BLOCKING,
    Incoherence.DANGLING_EDGE: Severity.BLOCKING,
    Incoherence.CONTRADICTORY_RELATION: Severity.BLOCKING,
    Incoherence.DISCONNECTED: Severity.BLOCKING,
    Incoherence.REPLY_TO_NOTHING: Severity.BLOCKING,
    Incoherence.SUPPORT_THAT_CONTRADICTS: Severity.ADVISORY,
    Incoherence.ATTACK_THAT_AGREES: Severity.ADVISORY,
}


@dataclass(frozen=True)
class Finding:
    kind: Incoherence
    node_ids: tuple[str, ...]
    detail: str = ""
    #: Which path the entailment critic used, when one produced this. A silent
    #: downgrade from a model to a heuristic changes what the finding means.
    critic_source: str = ""

    @property
    def severity(self) -> Severity:
        return _SEVERITY[self.kind]


@dataclass
class PatchCoherence:
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKING]

    @property
    def advisory(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ADVISORY]

    @property
    def passed(self) -> bool:
        """Structural findings decide. A model's opinion does not block a merge."""
        return not self.blocking


class CoherenceGate:
    def __init__(self, critic: EntailmentCritic | None = None) -> None:
        self.critic = critic

    def assess(self, patch: GraphPatch, graph: ArgumentGraph) -> PatchCoherence:
        result = PatchCoherence()
        fresh = [n for n in patch.nodes if n.id not in graph.nodes]
        known = set(graph.nodes) | {n.id for n in patch.nodes}

        result.findings.extend(_dangling(patch, known))
        # Everything downstream reasons over a merged view, so it must not run
        # against edges pointing into nothing.
        sound = [e for e in patch.edges if {e.source, e.target} <= known]

        result.findings.extend(_cycles(patch, graph, sound))
        result.findings.extend(_contradictory_relations(sound, graph))
        result.findings.extend(_disconnected(fresh, sound, graph))
        result.findings.extend(_replies_to_nothing(fresh, sound, graph))
        result.findings.extend(self._asserted_relations(sound, patch, graph))
        return result

    # ------------------------------------------------------------------ #
    # Semantic: a model's opinion, never blocking
    # ------------------------------------------------------------------ #
    def _asserted_relations(
        self, edges: list[Edge], patch: GraphPatch, graph: ArgumentGraph
    ) -> list[Finding]:
        """Check that an edge's asserted relation is not contradicted by the text.

        An edge is a claim about two propositions. ``A SUPPORTS B`` asserts that
        A helps B stand; if A contradicts B, the edge is decoration. ``A ATTACKS
        B`` asserts opposition; if A entails B, it is not an attack at all.
        """
        if self.critic is None:
            return []

        text = {n.id: n.text for n in patch.nodes}
        text.update({n.id: n.text for n in graph.nodes.values()})

        findings = []
        for edge in edges:
            if edge.type not in _CHECKABLE:
                continue
            source, target = text.get(edge.source), text.get(edge.target)
            if not source or not target:
                continue

            try:
                label, critic_source = self.critic.relate(source, target)
            except Exception:  # noqa: BLE001 - a broken critic costs the check, not the merge
                continue

            if edge.type is EdgeType.SUPPORTS and label == "contradiction":
                findings.append(
                    Finding(
                        kind=Incoherence.SUPPORT_THAT_CONTRADICTS,
                        node_ids=(edge.source, edge.target),
                        detail="marked as support, but it contradicts what it supports",
                        critic_source=critic_source,
                    )
                )
            elif edge.type is EdgeType.ATTACKS and label == "entailment":
                findings.append(
                    Finding(
                        kind=Incoherence.ATTACK_THAT_AGREES,
                        node_ids=(edge.source, edge.target),
                        detail="marked as an attack, but it agrees with its target",
                        critic_source=critic_source,
                    )
                )
        return findings


# --------------------------------------------------------------------------- #
# Structural: decidable from the graph's shape
# --------------------------------------------------------------------------- #
def _dangling(patch: GraphPatch, known: set[str]) -> list[Finding]:
    findings = []
    for edge in patch.edges:
        missing = {edge.source, edge.target} - known
        if missing:
            findings.append(
                Finding(
                    kind=Incoherence.DANGLING_EDGE,
                    node_ids=tuple(sorted(missing)),
                    detail="edge references a node in neither the patch nor the graph",
                )
            )
    return findings


def _cycles(
    patch: GraphPatch, graph: ArgumentGraph, edges: list[Edge]
) -> list[Finding]:
    """Detect a DEPENDS_ON cycle in the *merged* result, without merging.

    A trial view rather than a mutation: a gate with a side effect is a gate that
    has already merged whatever it was asked to judge.
    """
    trial = ArgumentGraph(
        nodes={**graph.nodes, **{n.id: n for n in patch.nodes}},
        edges=[*graph.edges, *edges],
    )
    cycle = trial.depends_on_cycle()
    if not cycle:
        return []
    return [
        Finding(
            kind=Incoherence.SELF_GROUNDING,
            node_ids=tuple(cycle),
            detail="merging this would make the argument depend on itself",
        )
    ]


def _contradictory_relations(
    edges: list[Edge], graph: ArgumentGraph
) -> list[Finding]:
    """One node cannot both support and attack the same claim."""
    merged = [*graph.edges, *edges]
    supports = {(e.source, e.target) for e in merged if e.type is EdgeType.SUPPORTS}
    attacks = {(e.source, e.target) for e in merged if e.type is EdgeType.ATTACKS}
    both = supports & attacks
    # Only report pairs this patch is responsible for; a contradiction already
    # sitting in the graph is not this merge's fault and blocking on it would
    # make the graph unmergeable forever.
    introduced = {(e.source, e.target) for e in edges}
    return [
        Finding(
            kind=Incoherence.CONTRADICTORY_RELATION,
            node_ids=pair,
            detail="this node both supports and attacks the same claim",
        )
        for pair in sorted(both & introduced)
    ]


def _disconnected(
    fresh: list[Node], edges: list[Edge], graph: ArgumentGraph
) -> list[Finding]:
    """Every new node must join the argument, once there is an argument to join.

    The opening node of an empty graph is exempt: there is nothing yet for it to
    connect to, and refusing it would make the first patch unmergeable.
    """
    if not graph.nodes and len(fresh) < 2:
        return []

    attached = {e.source for e in edges} | {e.target for e in edges}
    attached |= {e.source for e in graph.edges} | {e.target for e in graph.edges}
    return [
        Finding(
            kind=Incoherence.DISCONNECTED,
            node_ids=(node.id,),
            detail="this node would join the manuscript as part of no argument",
        )
        for node in fresh
        if node.id not in attached
    ]


def _replies_to_nothing(
    fresh: list[Node], edges: list[Edge], graph: ArgumentGraph
) -> list[Finding]:
    outgoing = {e.source for e in [*graph.edges, *edges]}
    return [
        Finding(
            kind=Incoherence.REPLY_TO_NOTHING,
            node_ids=(node.id,),
            detail="a REPLY must point at what it replies to",
        )
        for node in fresh
        if node.type is NodeType.REPLY and node.id not in outgoing
    ]
