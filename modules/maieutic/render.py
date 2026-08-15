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


@dataclass(frozen=True)
class ReadingPlan:
    """An order and a section for each node, from :mod:`modules.linearize`.

    Optional. Without one the renderer groups by each node's own ``section``
    field and orders within a section, which is what it always did. With one,
    the order is the optimizer's and the sections are its segmentation — and
    only then can the renderer know which dependencies cross a boundary.
    """

    order: list[str]
    #: node id -> section title ("Part II").
    section_of: dict[str, str]

    def titles(self) -> list[str]:
        """Section titles in reading order, each appearing once."""

        seen: list[str] = []
        for node in self.order:
            title = self.section_of.get(node, _UNSECTIONED)
            if title not in seen:
                seen.append(title)
        return seen


@dataclass
class Manuscript:
    text: str = ""
    #: Node ids in the order they were rendered. No id appears twice.
    order: list[str] = field(default_factory=list)
    #: Objections nothing replies to, declared rather than buried.
    open_problems: list[str] = field(default_factory=list)
    #: Structural problems a reader of the prose could not otherwise see.
    warnings: list[str] = field(default_factory=list)
    #: (node, section) for every cross-reference emitted. The count is what the
    #: ordering optimizer is minimising, so it belongs in the artifact rather
    #: than only in the prose.
    cross_references: list[tuple[str, str]] = field(default_factory=list)

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


def render(
    graph: ArgumentGraph,
    audience: Audience = Audience.MANUSCRIPT,
    plan: ReadingPlan | None = None,
) -> Manuscript:
    """Render the whole graph. Every node appears exactly once."""
    manuscript = Manuscript()
    lines: list[str] = []

    for section, nodes in _sectioned(graph, plan):
        lines.append(f"## {section}")
        lines.append("")
        for node in nodes:
            manuscript.order.append(node.id)
            references = _cross_references(node, graph, plan, manuscript.order)
            for _, title in references:
                manuscript.cross_references.append((node.id, title))
            lines.append(_paragraph(node, audience, references, manuscript.order))
            lines.append("")

    _declare_open_problems(graph, manuscript, lines)
    _warn(graph, manuscript, lines)

    manuscript.text = "\n".join(lines).strip() + "\n"
    return manuscript


def _sectioned(
    graph: ArgumentGraph, plan: ReadingPlan | None
) -> list[tuple[str, list[Node]]]:
    """Sections and their nodes, in reading order.

    With a plan, both come from the optimizer. Without one, the previous
    behaviour: group by each node's ``section`` field and order within it.
    """

    if plan is None:
        return [(section, order_nodes(graph, section)) for section in sections(graph)]

    grouped: list[tuple[str, list[Node]]] = []
    for title in plan.titles():
        nodes = [
            graph.nodes[n]
            for n in plan.order
            if n in graph.nodes and plan.section_of.get(n, _UNSECTIONED) == title
        ]
        grouped.append((title, nodes))
    return grouped


def _cross_references(
    node: Node,
    graph: ArgumentGraph,
    plan: ReadingPlan | None,
    rendered: list[str],
) -> list[tuple[str, str]]:
    """``(prerequisite id, section title)`` for each dependency leaving this section.

    This is where the ordering work becomes visible to a reader. A dependency
    inside the section needs no reference — they just read it — so a reference
    is emitted **exactly** when a DEPENDS_ON edge crosses a boundary. That makes
    the reference count a function of the order, which is what
    :mod:`modules.linearize` minimises.

    A reference is a structural label, not a claim: it names where the material
    already is. The renderer still composes nothing, which is the rule that
    keeps the last step free of the fabrication risk the gates exist to stop.
    """

    if plan is None:
        return []

    here = plan.section_of.get(node.id, _UNSECTIONED)
    references: list[tuple[str, str]] = []
    for edge in graph.outgoing(node.id, EdgeType.DEPENDS_ON):
        if edge.target not in graph.nodes:
            continue
        there = plan.section_of.get(edge.target, _UNSECTIONED)
        if there != here:
            references.append((edge.target, there))
    # Deterministic and de-duplicated: two dependencies into one section are one
    # reference, and a reader should not be sent to Part II twice in a sentence.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for target, title in sorted(references, key=lambda r: (r[1], r[0])):
        if title not in seen:
            seen.add(title)
            unique.append((target, title))
    return unique


def _reference_clause(references: list[tuple[str, str]], rendered: list[str]) -> str:
    """"as argued in Part II" — or "below", when the material has not been read.

    Precedence makes the backward case the normal one, but a plan is not
    required to come from the optimizer, and a section grouping that puts a
    prerequisite later must not be described as already argued. Saying "as
    argued in" about something the reader has not reached is simply false.
    """

    titles = [title for _, title in references]
    if not titles:
        return ""
    behind = all(target in rendered for target, _ in references)
    joined = titles[0] if len(titles) == 1 else ", ".join(titles[:-1]) + f" and {titles[-1]}"
    return f"as argued in {joined}" if behind else f"see {joined}, below"


def _paragraph(
    node: Node,
    audience: Audience,
    references: list[tuple[str, str]] | None = None,
    rendered: list[str] | None = None,
) -> str:
    body = " ".join(node.text.split())
    label = _LABEL.get(node.type)
    if label:
        body = f"*{label}.* {body}"

    clause = _reference_clause(references or [], rendered or [])
    if clause:
        body = f"{body} ({clause})"

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
