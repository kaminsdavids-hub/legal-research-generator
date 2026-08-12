"""Rendering the graph as prose.

The manuscript is the argument graph; prose is a view of it. That ordering
matters, because it is what makes non-repetition structural: every node renders
exactly once, so a claim reachable by two paths is written once and referred to,
rather than restated because the writer forgot it had already been said.

**Rendering is deterministic and involves no model.** A model writing the prose
would reintroduce, at the last step, every fabrication risk the gates spend their
effort preventing — and it would do so at the point where nothing downstream
checks it. The author's own sentences are the prose. What this module adds is
ordering, structural labels, and disclosure; it never composes a claim.

Two disclosures are not optional, and neither is suppressible by audience:

* **An unverified citation never renders as clean authority.** It is marked in
  the manuscript itself, not only in the review view. The same rule the dialectic
  module enforces for rescinded authority: a reader must not have to know which
  view they are reading to know what is confirmed.
* **Unanswered objections are declared.** A paper that states its strongest
  objections and answers them is the point; one that states them and quietly
  moves on is the failure the graph exists to make visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .graph import ArgumentGraph, EdgeType, Node, NodeType, Provenance

#: Reading order within a section, before DEPENDS_ON is taken into account.
#: Claims first, then what holds them up, then what attacks them.
_TYPE_ORDER: dict[NodeType, int] = {
    NodeType.THESIS: 0,
    NodeType.ORIGINAL: 1,
    NodeType.PREMISE: 2,
    NodeType.AUTHORITY: 3,
    NodeType.DISTINCTION: 4,
    NodeType.IMPLICATION: 5,
    NodeType.OBJECTION: 6,
    NodeType.REPLY: 7,
}

#: Structural labels. These assert nothing — they name the role a node plays,
#: which the graph already records. Composing a claim would be a different act.
_LABEL: dict[NodeType, str] = {
    NodeType.OBJECTION: "One objection",
    NodeType.REPLY: "Reply",
    NodeType.DISTINCTION: "A distinction",
    NodeType.IMPLICATION: "It follows",
}

_UNSECTIONED = "Argument"


class Audience(StrEnum):
    #: Clean prose. Still carries the two mandatory disclosures.
    MANUSCRIPT = "manuscript"
    #: Adds provenance, so the author can see what they wrote and what they
    #: merely accepted.
    REVIEW = "review"


@dataclass
class Manuscript:
    text: str = ""
    #: Node ids in the order they were rendered. No id appears twice.
    order: list[str] = field(default_factory=list)
    #: Objections nothing replies to, declared rather than buried.
    open_problems: list[str] = field(default_factory=list)
    #: Structural problems a reader of the prose could not otherwise see.
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


def order_nodes(graph: ArgumentGraph, section: str) -> list[Node]:
    """Deterministic reading order for one section.

    ``DEPENDS_ON`` decides what a reader must accept first; everything it does
    not constrain falls back to node type and then to id. The id tie-break is
    what makes the order *stable* — a renderer that shuffles equivalent nodes
    makes every diff between drafts unreadable.
    """
    nodes = [n for n in graph.nodes.values() if (n.section or _UNSECTIONED) == section]
    ids = {n.id for n in nodes}

    # A depends on B: B must be read first.
    prerequisites: dict[str, set[str]] = {n.id: set() for n in nodes}
    for edge in graph.edges:
        if edge.type is EdgeType.DEPENDS_ON and edge.source in ids and edge.target in ids:
            prerequisites[edge.source].add(edge.target)

    def rank(node: Node) -> tuple[int, str]:
        return (_TYPE_ORDER.get(node.type, len(_TYPE_ORDER)), node.id)

    remaining = sorted(nodes, key=rank)
    ordered: list[Node] = []
    placed: set[str] = set()
    while remaining:
        ready = [n for n in remaining if prerequisites[n.id] <= placed]
        if not ready:
            # A cycle among the survivors. The coherence gate blocks patches that
            # create one, but a graph loaded from disk can carry it. Emitting the
            # rest in a stable order beats hanging or dropping them silently; the
            # cycle itself is reported as a warning.
            ready = [remaining[0]]
        node = ready[0]
        ordered.append(node)
        placed.add(node.id)
        remaining.remove(node)

    return _replies_after_their_objections(ordered, graph)


def _replies_after_their_objections(
    ordered: list[Node], graph: ArgumentGraph
) -> list[Node]:
    """Move each reply directly after the objection it answers.

    Reading an objection three paragraphs before its answer is how a paper looks
    like it has no answer.
    """
    by_id = {n.id: n for n in ordered}
    answers: dict[str, list[Node]] = {}
    for node in ordered:
        if node.type is not NodeType.REPLY:
            continue
        for edge in graph.outgoing(node.id):
            if edge.target in by_id and by_id[edge.target].type is NodeType.OBJECTION:
                answers.setdefault(edge.target, []).append(node)
                break

    attached = {n.id for replies in answers.values() for n in replies}
    result: list[Node] = []
    for node in ordered:
        if node.id in attached:
            continue
        result.append(node)
        result.extend(answers.get(node.id, []))
    return result


def sections(graph: ArgumentGraph) -> list[str]:
    """Section names, unsectioned material first.

    The graph records which section a node belongs to but not what order the
    sections go in, so the rest is sorted. See OBSERVABLES D18: a real manuscript
    needs an authored outline, and this is not one.
    """
    named = {n.section for n in graph.nodes.values() if n.section}
    unsectioned = any(not n.section for n in graph.nodes.values())
    return ([_UNSECTIONED] if unsectioned else []) + sorted(named)


def render(graph: ArgumentGraph, audience: Audience = Audience.MANUSCRIPT) -> Manuscript:
    """Render the whole graph. Every node appears exactly once."""
    manuscript = Manuscript()
    lines: list[str] = []

    for section in sections(graph):
        lines.append(f"## {section}")
        lines.append("")
        for node in order_nodes(graph, section):
            manuscript.order.append(node.id)
            lines.append(_paragraph(node, audience))
            lines.append("")

    _declare_open_problems(graph, manuscript, lines)
    _warn(graph, manuscript, lines)

    manuscript.text = "\n".join(lines).strip() + "\n"
    return manuscript


def _paragraph(node: Node, audience: Audience) -> str:
    body = " ".join(node.text.split())
    label = _LABEL.get(node.type)
    if label:
        body = f"*{label}.* {body}"

    if node.type is NodeType.AUTHORITY and node.citation:
        # An unverified citation is disclosed in the manuscript view too. A
        # reader must not have to know which view they are reading to know what
        # has been confirmed.
        mark = node.citation if node.verified else f"{node.citation} — UNVERIFIED"
        body = f"{body} ({mark})"

    if audience is Audience.REVIEW:
        body = f"{body}  \n`[{node.provenance.value}]`"
    return body


def _declare_open_problems(
    graph: ArgumentGraph, manuscript: Manuscript, lines: list[str]
) -> None:
    unanswered = graph.unanswered_attacks()
    if not unanswered:
        return

    seen: set[str] = set()
    for edge in unanswered:
        objection = graph.nodes.get(edge.source)
        if objection is None or objection.id in seen:
            continue
        seen.add(objection.id)
        manuscript.open_problems.append(objection.text)

    lines.append("## Open problems")
    lines.append("")
    lines.append(
        "These objections are stated in this paper and not answered by it."
    )
    lines.append("")
    lines.extend(f"* {' '.join(text.split())}" for text in manuscript.open_problems)
    lines.append("")


def _warn(graph: ArgumentGraph, manuscript: Manuscript, lines: list[str]) -> None:
    cycle = graph.depends_on_cycle()
    if cycle:
        manuscript.warnings.append(
            f"the argument depends on itself: {' -> '.join(cycle)}; "
            "no reading order can fix this"
        )

    unverified = [
        n for n in graph.of_type(NodeType.AUTHORITY) if n.citation and not n.verified
    ]
    if unverified:
        manuscript.warnings.append(
            f"{len(unverified)} authority node(s) carry an unconfirmed citation"
        )

    unattributed = [
        n for n in graph.nodes.values() if n.provenance is not Provenance.HUMAN
    ]
    if unattributed and len(unattributed) == len(graph.nodes):
        manuscript.warnings.append(
            "no node in this manuscript was written by the author"
        )

    if not manuscript.warnings:
        return
    lines.append("## Warnings")
    lines.append("")
    lines.extend(f"* {w}" for w in manuscript.warnings)
    lines.append("")
