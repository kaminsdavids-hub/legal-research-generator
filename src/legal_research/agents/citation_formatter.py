"""Citation Formatter (spec §2.8).

Formats all authorities in the configured style (Bluebook default), turning the
Writer's ``{{cite:ID}}`` tokens into numbered footnotes with correct
short-form/``id.``/``supra`` handling, and building the table of authorities.
Citations the Verifier removed are stripped from the prose here.
"""

from __future__ import annotations

import re
from typing import Any

from ..citations.bluebook import FootnoteBuilder, table_of_authorities
from ..models import CiteStatus, FootnoteRef
from .base import Agent, AgentContext, AgentResult

_TOKEN = re.compile(r"\{\{cite:([^}]+)\}\}")


class CitationFormatterAgent(Agent):
    name = "Citation Formatter"
    expert_role = "writer"

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        fb = FootnoteBuilder(ctx.formatter)
        bb.footnotes = {}

        for section in bb.outline:
            section_footnotes: list[FootnoteRef] = []
            pieces: list[str] = []
            cursor = 0
            for match in _TOKEN.finditer(section.content):
                pieces.append(section.content[cursor : match.start()])
                cursor = match.end()
                pieces.append(self._render_token(ctx, fb, match.group(1), section_footnotes))
            pieces.append(section.content[cursor:])
            section.content = _clean("".join(pieces))
            if section_footnotes:
                bb.footnotes[section.id] = section_footnotes

        verified_records = []
        seen: set[str] = set()
        for citation in bb.citations:
            if citation.status != CiteStatus.VERIFIED:
                continue
            record = ctx.corpus.get(citation.record_id)
            if record and record.id not in seen:
                seen.add(record.id)
                verified_records.append(record)
        bb.toa = table_of_authorities(verified_records, ctx.formatter)

        total_footnotes = sum(len(v) for v in bb.footnotes.values())
        return AgentResult(
            agent=self.name,
            summary=f"formatted {total_footnotes} footnotes; TOA has {len(seen)} authorities",
            payload={"footnotes": total_footnotes, "toa_authorities": len(seen)},
        )

    @staticmethod
    def _render_token(
        ctx: AgentContext,
        fb: FootnoteBuilder,
        cid: str,
        section_footnotes: list[FootnoteRef],
    ) -> str:
        bb = ctx.blackboard
        try:
            citation = bb.get_citation(cid)
        except KeyError:
            return ""
        record = ctx.corpus.get(citation.record_id)
        if record is None or citation.status == CiteStatus.REMOVED:
            return ""  # strip removed / unresolved cites from the prose
        number, text = fb.cite(record, citation.pin_cite)
        section_footnotes.append(FootnoteRef(number=number, text=text, citation_id=cid))
        return f"[^{number}]"


def _clean(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()
