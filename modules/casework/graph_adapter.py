"""Emission into the argument graph. Where a finding stops being a search result.

A `MishandledPrecedent` in a JSON archive is a fact about a search. The same
finding as a node in the argument graph is a claim the manuscript has to answer:
gap analysis sees it, the Socratic engine asks about it, and the renderer prints
it. That difference is the whole point of this module — the two search modules
are only coupled *through the graph*, and until a finding lands there it has
changed nothing about the paper.

Three rules govern what may be written:

**Nothing here is the author's.** Every emitted node carries
``Provenance.SYSTEM``. The graph reserves ``HUMAN`` for the answer-capture and
edit paths, and a search result wearing human provenance would corrupt the
scholarly-integrity record the whole package exists to keep.

**Only verified codings become nodes.** The claim a precedent node makes is
"a court decided this and your framework gets it wrong". That rests entirely on
the coding of the real case, so an unverified coding emits nothing — the same
rule ``precedent.match_precedents`` applies one layer up, enforced again here
because this is the layer that writes.

**Emission is idempotent.** Node ids are derived from the case id, so re-running
a search re-finds the same precedent and adds nothing. A loop that appended a
node per epoch would make the manuscript grow with every run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from modules.maieutic.graph import (
    ArgumentGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    Provenance,
)

__all__ = [
    "Emission",
    "emit_precedents",
    "node_id_for",
    "propositions_for_portfolio",
]

#: Prefix on every emitted node id, so a reader of the graph can see at a glance
#: which claims arrived from a search rather than from the author.
PREFIX = "divergence-"


def node_id_for(case_id: str) -> str:
    """A stable id per coded case: the same finding is the same node, always."""

    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8]
    return f"{PREFIX}{digest}"


@dataclass
class Emission:
    added: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    #: Matched a coded case, but no reading contradicts what it held. Kept
    #: separate from ``refused`` because the reason is the opposite one: the
    #: coding was fine, and the framework got the case right.
    no_conflict: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.added)} node(s) added"]
        if self.already_present:
            parts.append(f"{len(self.already_present)} already present")
        if self.refused:
            parts.append(f"{len(self.refused)} refused (unverified coding)")
        if self.no_conflict:
            parts.append(f"{len(self.no_conflict)} skipped (no reading contradicts)")
        return ", ".join(parts)


def _objection_text(precedent: Any) -> str:
    """What the node asserts, in the author's manuscript.

    Written as an objection because that is what it is: a decided case the
    paper's framework classifies wrongly is the strongest form of "your reading
    cannot be right". Filing it as an OBJECTION also puts it under the graph's
    existing machinery -- an objection with no reply is an
    ``UNANSWERED_ATTACK``, which the Socratic engine ranks second only to a
    self-grounding cycle. The finding therefore generates pressure on the author
    rather than sitting in a report.

    The citation is included and its warrant is the *human-verified coding* in
    the case corpus, not a model. No generator in this repository may emit a
    citation; a verified coding is the author's own work, and this is the one
    path that carries one into the graph.
    """

    readings = ", ".join(precedent.contradicting)
    return (
        f"{precedent.name} ({precedent.citation}) was decided {precedent.held}, "
        f"but {readings} classify its facts the other way. Either the reading is "
        f"wrong about a decided case, or the case is distinguishable on a ground "
        f"the framework does not yet capture."
    )


def emit_precedents(
    graph: ArgumentGraph,
    precedents: Sequence[Any],
    *,
    epoch: int,
    attach_to: str | None = None,
) -> Emission:
    """Write each mishandled precedent into the graph as an objection.

    ``attach_to`` is the node the objection attacks -- normally the thesis. When
    absent the node is still added but stands unattached, which gap analysis
    reports as an ``ORPHANED_NODE``: visible, and the author decides what it
    bears on. Guessing an attachment would be the machine deciding which of the
    paper's claims a case undermines, and that is an argument, not a fact.
    """

    emission = Emission()

    for precedent in precedents:
        coded_by = getattr(precedent, "coded_by", "")
        if not coded_by:
            # Mirrors precedent.match_precedents: a collision built on a guessed
            # coding is a guess with a citation attached, and this is the layer
            # that would put it in the manuscript.
            emission.refused.append(getattr(precedent, "case_id", "?"))
            continue

        if not getattr(precedent, "contradicting", ()):
            # A match with nothing contradicting it is a case the framework
            # handles correctly, and there is nothing for the author to answer.
            # Emitting it anyway would file the sentence "X was decided <held>,
            # but the formalised readings classify its facts the other way"
            # against a case where they did not -- and for a threshold holding
            # coded with an empty outcome (Junger), that sentence is doubly
            # false. An objection nobody can reply to is noise the gap analysis
            # would then report as an UNANSWERED_ATTACK forever.
            emission.no_conflict.append(precedent.case_id)
            continue

        node_id = node_id_for(precedent.case_id)
        if node_id in graph.nodes:
            emission.already_present.append(node_id)
            continue

        graph.nodes[node_id] = Node(
            id=node_id,
            type=NodeType.OBJECTION,
            text=_objection_text(precedent),
            # Never HUMAN. A search result is structural material the system
            # added, and the graph's provenance record is what separates the
            # author's contribution from the machine's.
            provenance=Provenance.SYSTEM,
            section=f"epoch-{epoch}",
        )
        if attach_to and attach_to in graph.nodes:
            graph.edges.append(Edge(node_id, attach_to, EdgeType.ATTACKS))
        emission.added.append(node_id)

    return emission


def propositions_for_portfolio(
    graph: ArgumentGraph,
    *,
    contested_from_divergence: bool = True,
) -> list[dict[str, Any]]:
    """Graph nodes that need authority, in the shape Portfolio's loader reads.

    This is the second half of the coupling: Divergence writes objections into
    the graph, and Portfolio's coverage requirement set is read *from the graph*
    rather than handed over directly. Neither module imports the other, and the
    manuscript is the only thing they share.

    Emitted objections are marked ``contested`` by default. An objection built
    from a decided case is exactly the claim a single persuasive citation should
    not be allowed to carry -- so it inherits the two-distinct-sources rule.
    """

    propositions: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        if node.type not in (NodeType.OBJECTION, NodeType.ORIGINAL, NodeType.THESIS):
            continue
        from_search = node.id.startswith(PREFIX)
        propositions.append(
            {
                "id": node.id,
                "text": node.text,
                "contested": bool(from_search and contested_from_divergence),
                "breadth_matters": False,
                "from_divergence": from_search,
            }
        )
    return sorted(propositions, key=lambda p: p["id"])


def unanswered_emissions(graph: ArgumentGraph) -> list[str]:
    """Emitted objections nothing in the paper replies to.

    The measure that says whether the coupling did any good. A search that
    generated twelve precedent objections and left all twelve unanswered has
    produced twelve open problems, which the manuscript's own warnings will
    declare -- and that is the honest outcome, not a failure to hide.
    """

    answered: set[str] = set()
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        if source is not None and source.type is NodeType.REPLY:
            answered.add(edge.target)
    return sorted(
        node_id
        for node_id, node in graph.nodes.items()
        if node_id.startswith(PREFIX)
        and node.type is NodeType.OBJECTION
        and node_id not in answered
    )


def emitted_nodes(graph: ArgumentGraph) -> Iterable[Node]:
    return (node for nid, node in sorted(graph.nodes.items()) if nid.startswith(PREFIX))
