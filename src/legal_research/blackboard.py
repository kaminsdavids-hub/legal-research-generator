"""The shared blackboard.

Every agent reads from and writes to a single :class:`Blackboard` per paper. It is
the coordination surface described in the spec: thesis, ideas, outline, sources,
drafts, citations, edits and verification results all live here.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from pydantic import BaseModel, Field, PrivateAttr

from .models import (
    Authority,
    BrainstormTurn,
    Citation,
    CiteStatus,
    EditRecord,
    FootnoteRef,
    Idea,
    IdeaStatus,
    NoveltyAssessment,
    OutlineSection,
    RetrievedPassage,
    SectionStatus,
    VerificationResult,
)


class Blackboard(BaseModel):
    """Mutable shared state for one paper project."""

    session_id: str
    title: str = "Untitled Research Paper"
    thesis: str = ""
    brainstorm: list[BrainstormTurn] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    outline: list[OutlineSection] = Field(default_factory=list)
    authorities: list[Authority] = Field(default_factory=list)
    retrieved: list[RetrievedPassage] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    footnotes: dict[str, list[FootnoteRef]] = Field(default_factory=dict)
    toa: dict[str, list[str]] = Field(default_factory=dict)
    edits: list[EditRecord] = Field(default_factory=list)
    novelty: NoveltyAssessment | None = None
    # Set when the last interview turn ran in degraded mode (interviewer model
    # unreachable or empty), cleared on the next healthy turn. Without this the
    # brainstorm endpoint returns only the Blackboard, so a dead model is
    # indistinguishable in the UI from an interviewer that repeats itself.
    brainstorm_degraded: str = ""

    _idea_seq: int = PrivateAttr(default=0)
    _section_seq: int = PrivateAttr(default=0)
    _cite_seq: int = PrivateAttr(default=0)
    _edit_seq: int = PrivateAttr(default=0)

    # ---- ideas ---------------------------------------------------------------------

    def add_idea(
        self, text: str, angle: str = "", novelty_note: str = "", claim: str = ""
    ) -> Idea:
        self._idea_seq += 1
        idea = Idea(
            id=f"idea-{self._idea_seq:03d}",
            text=text,
            claim=claim,
            angle=angle,
            novelty_note=novelty_note,
        )
        self.ideas.append(idea)
        return idea

    def set_idea_status(self, idea_id: str, status: IdeaStatus, priority: int | None = None) -> Idea:
        idea = self.get_idea(idea_id)
        idea.status = status
        if priority is not None:
            idea.priority = priority
        return idea

    def get_idea(self, idea_id: str) -> Idea:
        for idea in self.ideas:
            if idea.id == idea_id:
                return idea
        raise KeyError(f"no idea {idea_id!r}")

    def selected_ideas(self) -> list[Idea]:
        kept = [i for i in self.ideas if i.status == IdeaStatus.KEEP]
        return sorted(kept, key=lambda i: (i.priority, i.id))

    # ---- outline -------------------------------------------------------------------

    def add_section(self, title: str, idea_ids: Iterable[str] = ()) -> OutlineSection:
        self._section_seq += 1
        section = OutlineSection(
            id=f"sec-{self._section_seq:03d}", title=title, idea_ids=list(idea_ids)
        )
        self.outline.append(section)
        return section

    def get_section(self, section_id: str) -> OutlineSection:
        for section in self.outline:
            if section.id == section_id:
                return section
        raise KeyError(f"no section {section_id!r}")

    # ---- citations -----------------------------------------------------------------

    def new_citation_id(self) -> str:
        self._cite_seq += 1
        return f"cite-{self._cite_seq:03d}"

    def add_citation(self, citation: Citation) -> Citation:
        self.citations.append(citation)
        return citation

    def add_edit(self, section_id: str, before: str, after: str, note: str, author: str) -> EditRecord:
        self._edit_seq += 1
        record = EditRecord(
            id=f"edit-{self._edit_seq:03d}",
            section_id=section_id,
            before=before,
            after=after,
            note=note,
            author=author,
        )
        self.edits.append(record)
        return record

    def get_citation(self, citation_id: str) -> Citation:
        for c in self.citations:
            if c.id == citation_id:
                return c
        raise KeyError(f"no citation {citation_id!r}")

    def verified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.status == CiteStatus.VERIFIED]

    def unverified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.status != CiteStatus.VERIFIED]

    # ---- status helpers ------------------------------------------------------------

    def refresh_section_statuses(self) -> None:
        for section in self.outline:
            if not section.content:
                section.status = SectionStatus.IDEA
                continue
            cites = [self.get_citation(cid) for cid in section.citation_ids]
            active = [c for c in cites if c.status != CiteStatus.REMOVED]
            if not active:
                section.status = SectionStatus.DRAFTED
            elif all(c.status == CiteStatus.VERIFIED for c in active):
                section.status = SectionStatus.VERIFIED
            else:
                section.status = SectionStatus.CITED

    def is_shippable(self) -> bool:
        """A paper may ship only when no citation remains unverified, and only
        when it still has authority to stand on.

        A REMOVED cite has already been stripped from the manuscript, so it does not
        block shipping; only PENDING / NEEDS_REVIEW cites do.

        That rule alone is satisfied vacuously by removing everything, which is
        not a hypothetical. A full pipeline run proposed 61 citations, removed
        all 61, and reported ``shippable=True`` over a 9,653-word draft with an
        empty table of authorities. A legal paper with no surviving authority is
        not shippable by any standard the rest of this system applies -- and
        `tests/test_pipeline.py` already asserted "at least one verified
        citation" separately, because this function did not.

        So: a paper that proposed citations must retain at least one verified
        one. A paper that has not reached the citation stage at all is left
        alone, since it has not failed anything yet.
        """

        self.refresh_section_statuses()
        if any(
            c.status in (CiteStatus.PENDING, CiteStatus.NEEDS_REVIEW)
            for c in self.citations
        ):
            return False
        return not self.citations or any(
            c.status is CiteStatus.VERIFIED for c in self.citations
        )


_session_counter = itertools.count(1)


def new_session_id() -> str:
    return f"session-{next(_session_counter):04d}"
