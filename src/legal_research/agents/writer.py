"""Writer / Voice (spec §2.6, §4, §5).

Produces human-like, distinctive prose per section. It is the enforcement point for
the anti-hallucination rule: every legal proposition it asserts must be grounded
through the retriever via the :class:`CitationGuard`. A proposition that cannot be
grounded is dropped (never invented) and recorded as a blocked assertion.

Grounded sentences carry a ``{{cite:ID}}`` token that the Citation Formatter later
turns into a numbered footnote.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..citations.verifier import GroundingError
from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import HERMES, SAUL
from ..models import Relation, SectionStatus
from .base import Agent, AgentContext, AgentResult

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_SYSTEM = (
    "TASK: write\n"
    "You are a legal scholar writing in a distinctive, human voice: varied sentence "
    "length, no formulaic transitions, no empty summarizing filler. Write a tight "
    "paragraph developing the given point. Do not invent citations."
)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


#: Sentence boundaries, avoiding the abbreviations legal prose is full of.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])")

#: Abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS = (
    " v.", " no.", " cf.", " id.", " ch.", " art.", " sec.", " cl.", " pt.",
    " u.s.", " f.2d", " f.3d", " f. supp.", " c.f.r.", " stat.", " ed.", " al.",
)







#: A claim shorter than this is a fragment, not something a source can support.
_MIN_CLAIM_WORDS = 6

#: Long hypotheses degrade entailment scoring, and a paragraph is not one claim.
_MAX_CLAIM_WORDS = 60


def split_sentences(paragraph: str) -> list[str]:
    """Split *paragraph* into sentences, keeping legal abbreviations intact."""
    text = " ".join(paragraph.split())
    parts: list[str] = []
    for candidate in _SENTENCE_END.split(text):
        if parts and parts[-1].lower().endswith(_ABBREVIATIONS):
            parts[-1] = f"{parts[-1]} {candidate}"
        else:
            parts.append(candidate)
    return [p.strip() for p in parts if p.strip()]



_COUNTER_SYSTEM = (
    "TASK: write\n"
    "You are a skeptical appellate advocate. Draft the strongest concise "
    "counterargument to the stated thesis, focusing on the best doctrinal and policy "
    "objections. Return one tight paragraph; no bullets and no citations."
)

_REBUTTAL_SYSTEM = (
    "TASK: write\n"
    "You are the article's author. Rebut the counterargument directly using "
    "controlling principles and clear limiting logic. Return one tight paragraph; no "
    "bullets and no citations."
)


class WriterAgent(Agent):
    name = "Writer / Voice"
    expert_role = SAUL

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        bb = ctx.blackboard
        section_ids: list[str] = kwargs.get("section_ids") or [s.id for s in bb.outline]

        requested_target = kwargs.get("target_words")
        if requested_target is not None:
            target_words = max(1200, min(12000, int(requested_target)))
            min_words = int(target_words * 0.9)
            max_words = int(target_words * 1.1)
        else:
            min_words = max(500, int(ctx.settings.manuscript_target_min_words))
            max_words = max(min_words, int(ctx.settings.manuscript_target_max_words))
            target_words = (min_words + max_words) // 2

        per_section_target = max(
            int(ctx.settings.draft_paragraph_target_words) * 2,
            target_words // max(1, len(section_ids)),
        )

        drafted = 0
        scaffolded = 0
        blocked: list[str] = []
        for section_id in section_ids:
            section = bb.get_section(section_id)
            propositions = self._section_propositions(ctx, section)

            is_counter_rebuttal = self._is_counter_rebuttal_title(section.title)

            def _on_progress(paragraphs: list[str], section: Any = section) -> None:
                # Persist raw (pre-citation) prose as soon as each paragraph lands so
                # the manuscript panel shows real progress within one LLM call instead
                # of only after the whole section (up to ~15 calls) finishes, and so a
                # mid-section interruption still leaves visible partial content.
                content = "\n\n".join(p.strip() for p in paragraphs if p.strip()).strip()
                if content:
                    section.content = content
                    section.status = SectionStatus.DRAFTED

            drafted_paragraphs = self._draft_section_prose(
                ctx,
                section.title,
                propositions,
                target_words=per_section_target,
                on_progress=_on_progress,
            )
            if is_counter_rebuttal:
                scaffolded += 1

            section.citation_ids = []
            rendered: list[str] = []
            for paragraph, point in drafted_paragraphs:
                marker = ""
                # Cite the paragraph for the point it was actually drafted from.
                # That association exists at generation time; it used to be
                # discarded and then re-derived by a second, independent
                # round-robin, so a paragraph could be cited for a proposition it
                # had never been written about (REMEDIATION §23).
                if point:
                    try:
                        citation = ctx.guard.ground(point)
                        ctx.guard.assert_grounded(citation)
                    except GroundingError:
                        blocked.append(point)
                    else:
                        bb.add_citation(citation)
                        section.citation_ids.append(citation.id)
                        marker = f"{{{{cite:{citation.id}}}}}"
                rendered.append((paragraph + marker).strip())

            section.content = "\n\n".join(rendered).strip()
            section.status = SectionStatus.DRAFTED if section.content else SectionStatus.IDEA
            if section.content:
                drafted += 1

        manuscript_words = sum(_count_words(s.content) for s in bb.outline if s.content)

        return AgentResult(
            agent=self.name,
            summary=(
                f"drafted {drafted} sections; scaffolded {scaffolded} "
                f"counterargument/rebuttal sections; blocked {len(blocked)} "
                f"ungrounded assertions; manuscript words={manuscript_words} "
                f"(target {min_words}-{max_words})"
            ),
            payload={
                "blocked_assertions": blocked,
                "counter_rebuttal_sections": scaffolded,
                "manuscript_words": manuscript_words,
                "target_min_words": min_words,
                "target_max_words": max_words,
                "requested_target_words": target_words,
            },
        )

    def _draft_section_prose(
        self,
        ctx: AgentContext,
        section_title: str,
        propositions: list[str],
        target_words: int,
        on_progress: Callable[[list[str]], None] | None = None,
    ) -> list[tuple[str, str]]:
        if self._is_counter_rebuttal_title(section_title):
            thesis = ctx.blackboard.thesis or (propositions[0] if propositions else "the thesis")
            contrary_props = [
                a.proposition
                for a in ctx.blackboard.authorities
                if a.relation is Relation.CONTRARY and a.proposition
            ]
            contrary_note = (
                "\n".join(f"- {p}" for p in list(dict.fromkeys(contrary_props))[:3])
                if contrary_props
                else "- no explicit contrary authority mapped; infer the strongest doctrinal objection"
            )

            hermes = ctx.pool.get(HERMES)
            counter = hermes.chat(
                [
                    ChatMessage("system", _COUNTER_SYSTEM),
                    ChatMessage(
                        "user",
                        (
                            f"Thesis: {thesis}\n"
                            "Likely contrary authority/propositions:\n"
                            f"{contrary_note}"
                        ),
                    ),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()

            legal_writer = ctx.pool.get(SAUL)
            rebuttal = legal_writer.chat(
                [
                    ChatMessage("system", _REBUTTAL_SYSTEM),
                    ChatMessage(
                        "user",
                        (
                            f"Thesis: {thesis}\n"
                            f"Counterargument to answer: {counter}"
                        ),
                    ),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
            paragraphs = [f"Counterargument. {counter}".strip(), f"Rebuttal. {rebuttal}".strip()]
            if on_progress:
                on_progress(paragraphs)
            # The counterargument and the rebuttal are both offered for the
            # thesis: one states the objection to it, the other defends it.
            return self._expand_section(
                ctx,
                section_title=section_title,
                propositions=propositions,
                paragraphs=paragraphs,
                drafted_from=[thesis, thesis],
                target_words=target_words,
                on_progress=on_progress,
            )

        client = ctx.pool.get(self.expert_role)
        first = client.chat(
            [
                ChatMessage("system", _SYSTEM),
                ChatMessage(
                    "user",
                    (
                        f"Section: {section_title}\n"
                        f"Point: {propositions[0] if propositions else section_title}\n"
                        "Write one substantial legal-analysis paragraph around 180-260 words."
                    ),
                ),
            ],
            DecodingPolicy.BALANCED.config,
        ).strip()
        if on_progress:
            on_progress([first])
        opening_point = propositions[0] if propositions else section_title
        return self._expand_section(
            ctx,
            section_title=section_title,
            propositions=propositions,
            paragraphs=[first],
            drafted_from=[opening_point],
            target_words=target_words,
            on_progress=on_progress,
        )

    def _expand_section(
        self,
        ctx: AgentContext,
        section_title: str,
        propositions: list[str],
        paragraphs: list[str],
        drafted_from: list[str],
        target_words: int,
        on_progress: Callable[[list[str]], None] | None = None,
    ) -> list[tuple[str, str]]:
        max_paragraphs = max(2, int(ctx.settings.draft_max_paragraphs_per_section))
        # Each paragraph carries the point it was drafted from, so the citation
        # loop does not have to guess it back out of the prose.
        paragraph_target = max(120, int(ctx.settings.draft_paragraph_target_words))
        client = ctx.pool.get(self.expert_role)

        while _count_words("\n\n".join(paragraphs)) < target_words and len(paragraphs) < max_paragraphs:
            focus = propositions[len(paragraphs) % len(propositions)] if propositions else section_title
            para = client.chat(
                [
                    ChatMessage("system", _SYSTEM),
                    ChatMessage(
                        "user",
                        (
                            f"Section: {section_title}\n"
                            f"Current draft excerpt: {paragraphs[-1][-500:]}\n"
                            f"Next focus: {focus}\n"
                            "Write one additional paragraph that advances the argument with "
                            f"roughly {max(140, paragraph_target - 40)}-{paragraph_target + 60} words, "
                            "without bullets and without citations."
                        ),
                    ),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
            if para:
                paragraphs.append(para)
                drafted_from.append(focus)
                if on_progress:
                    on_progress(paragraphs)

        return [
            (p.strip(), f) for p, f in zip(paragraphs, drafted_from, strict=True) if p.strip()
        ]

    def _section_propositions(self, ctx: AgentContext, section: Any) -> list[str]:
        bb = ctx.blackboard
        if self._is_counter_rebuttal_title(getattr(section, "title", "")):
            return self._counter_rebuttal_propositions(bb)

        props: list[str] = []
        # Anchor the introduction on the thesis so the paper's central claim is cited.
        is_intro = bool(bb.outline) and bb.outline[0].id == section.id
        if is_intro and bb.thesis:
            props.append(bb.thesis)
        for idea_id in section.idea_ids:
            try:
                idea = bb.get_idea(idea_id)
                # Draft from, and cite for, the claim rather than the topic.
                text = idea.claim or idea.text
                if text not in props:
                    props.append(text)
            except KeyError:
                continue
        return props

    @staticmethod
    def _is_counter_rebuttal_title(title: str) -> bool:
        lowered = title.lower()
        return "counterargument" in lowered and "rebuttal" in lowered

    @staticmethod
    def _counter_rebuttal_propositions(bb: Any) -> list[str]:
        props: list[str] = []
        if bb.thesis:
            props.append(bb.thesis)

        contrary = [
            a.proposition
            for a in bb.authorities
            if a.relation is Relation.CONTRARY and a.proposition
        ]
        for proposition in contrary:
            if proposition not in props:
                props.append(proposition)
            if len(props) >= 3:
                break
        return props
