"""End-to-end pipeline (spec §0).

Wires the expert pool, router, retriever, corpus, formatter, guard and verifier into
an :class:`AgentContext`, then drives the nine agents through the stages: brainstorm
-> ideate -> select -> outline -> research -> draft -> voice/grammar -> verify ->
format -> novelty -> document/PDF. Operates on a caller-supplied
:class:`Blackboard` so the API can keep one per session.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import DISCLAIMER
from .agents import (
    ArgumentArchitect,
    CitationFormatterAgent,
    EditorAgent,
    FinanceAnalyst,
    Ideator,
    LegalResearcher,
    SocraticInterviewer,
    VerifierAgent,
    WriterAgent,
)
from .agents.argument_architect import noun_phrase
from .agents.base import AgentContext, AgentResult
from .agents.runtime import AgentRuntime, build_runtime
from .blackboard import Blackboard, new_session_id
from .citations.bluebook import build_formatter
from .citations.corpus import Corpus, load_corpus
from .citations.report import build_verification_report
from .citations.retriever import Retriever, build_retriever
from .citations.verifier import CitationGuard, build_verifier
from .config import Settings, get_settings
from .llm.pool import ExpertPool, build_expert_pool
from .mechanism import MechanismCritic, run_gate
from .models import IdeaStatus, SectionStatus
from .pdf.base import Footnote, PaperDocument, SectionRender, build_renderer
from .router import Router
from .voice.novelty import assess_novelty


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


@dataclass
class PipelineResult:
    blackboard: Blackboard
    steps: list[AgentResult]


class LegalResearchPipeline:
    def __init__(self, settings: Settings | None = None, runtime: AgentRuntime | None = None) -> None:
        self.settings = settings or get_settings()
        self.pool: ExpertPool = build_expert_pool(self.settings)
        self.router = Router(self.pool)
        self.corpus: Corpus = load_corpus(self.settings.corpus_path)
        self.retriever: Retriever = build_retriever(self.settings, self.corpus)
        self.formatter = build_formatter(self.settings.citation_style)
        self.verifier = build_verifier(self.settings, self.corpus)
        self.runtime = runtime or build_runtime("openhands")

        # Agent singletons (stateless; operate on the injected blackboard).
        self.interviewer = SocraticInterviewer()
        self.ideator = Ideator()
        self.legal_researcher = LegalResearcher()
        self.finance_analyst = FinanceAnalyst()
        self.architect = ArgumentArchitect()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.citation_formatter = CitationFormatterAgent()
        self.verifier_agent = VerifierAgent()

    # ---- context -------------------------------------------------------------------

    def context(self, bb: Blackboard) -> AgentContext:
        # The verifier's scorer, shared rather than rebuilt: the cross-encoder
        # is expensive to load, and a guard that chose passages with a different
        # scorer than the verifier judges them by would ground citations the
        # verifier then removes on a standard it never applied at write time.
        guard = CitationGuard(
            self.retriever, bb.new_citation_id, scorer=self.verifier.scorer
        )
        return AgentContext(
            settings=self.settings,
            pool=self.pool,
            router=self.router,
            retriever=self.retriever,
            corpus=self.corpus,
            formatter=self.formatter,
            guard=guard,
            verifier=self.verifier,
            blackboard=bb,
        )

    def new_session(self, title: str = "Untitled Research Paper") -> Blackboard:
        return Blackboard(session_id=new_session_id(), title=title)

    # ---- individual stages ---------------------------------------------------------

    def brainstorm(self, bb: Blackboard, scholar_input: str | None = None) -> AgentResult:
        return self.runtime.run(self.interviewer, self.context(bb), scholar_input=scholar_input)

    def ideate(self, bb: Blackboard, seed: str | None = None) -> AgentResult:
        return self.runtime.run(self.ideator, self.context(bb), seed=seed)

    def select_ideas(self, bb: Blackboard, selections: list[tuple[str, int]]) -> None:
        for idea_id, priority in selections:
            bb.set_idea_status(idea_id, IdeaStatus.KEEP, priority=priority)

    def build_outline(self, bb: Blackboard) -> AgentResult:
        return self.runtime.run(self.architect, self.context(bb))

    def research(self, bb: Blackboard) -> list[AgentResult]:
        ctx = self.context(bb)
        return [
            self.runtime.run(self.legal_researcher, ctx),
            self.runtime.run(self.finance_analyst, ctx),
        ]

    def draft(self, bb: Blackboard, target_words: int | None = None) -> AgentResult:
        kwargs: dict[str, int] = {}
        if target_words is not None:
            kwargs["target_words"] = int(target_words)
        return self.runtime.run(self.writer, self.context(bb), **kwargs)

    def verify(self, bb: Blackboard) -> AgentResult:
        return self.runtime.run(self.verifier_agent, self.context(bb))

    def format_citations(self, bb: Blackboard) -> AgentResult:
        return self.runtime.run(self.citation_formatter, self.context(bb))

    def edit_voice(self, bb: Blackboard) -> AgentResult:
        return self.runtime.run(self.editor, self.context(bb))

    def revise(self, bb: Blackboard, section_id: str, instruction: str) -> AgentResult:
        return self.runtime.run(
            self.editor, self.context(bb), section_id=section_id, instruction=instruction
        )

    def socratic_revise_paragraph(
        self,
        bb: Blackboard,
        section_id: str,
        paragraph_index: int,
        message: str,
        history: list[dict[str, str]] | None = None,
        apply_revision: bool = False,
        mode: str = "strengthen_doctrine",
    ) -> AgentResult:
        return self.runtime.run(
            self.editor,
            self.context(bb),
            socratic=True,
            section_id=section_id,
            paragraph_index=paragraph_index,
            message=message,
            history=history or [],
            apply_revision=apply_revision,
            mode=mode,
        )

    def mechanism_gate(self, bb: Blackboard) -> AgentResult:
        """Run the mechanism gate over the drafted manuscript.

        Not one of the nine agents: the layer that matters here is a regex scan
        with no model behind it and nothing for the router to route. It runs
        after the voice pass so it sees the prose that will ship, and before
        verification so a repaired paragraph's citations are checked in their
        final wording.
        """

        s = self.settings
        if not s.mechanism_gate_enabled:
            return AgentResult(
                agent="Mechanism Gate",
                summary="mechanism gate disabled",
                payload={"enabled": False},
            )

        critic = None
        if s.mechanism_gate_critic_enabled:
            critic = MechanismCritic(self.pool.get(s.mechanism_gate_critic_role))
        repair_client = (
            self.pool.get(self.writer.expert_role) if s.mechanism_gate_repair_enabled else None
        )
        report = run_gate(
            bb,
            critic=critic,
            repair_client=repair_client,
            critic_fail_open=bool(s.mechanism_gate_fail_open),
        )
        return AgentResult(
            agent="Mechanism Gate",
            summary=report.summary(),
            payload={
                "enabled": True,
                "findings": len(report.findings),
                "repaired": report.repaired,
                "unresolved": len(report.unresolved),
                "black_box": len(report.blocking),
                "critic_ran": report.critic_ran,
                "critic_error": report.critic_error,
            },
        )

    def assess_novelty(self, bb: Blackboard) -> None:
        bb.novelty = assess_novelty(bb.thesis or "", bb.retrieved)

    # ---- outputs -------------------------------------------------------------------

    def verification_report(self, bb: Blackboard) -> str:
        return build_verification_report(
            bb.citations, bb.verifications, self.corpus, bb.mechanism_findings
        )

    def build_document(
        self, bb: Blackboard, author: str = "Anonymous Scholar", date: str = ""
    ) -> PaperDocument:
        bb.refresh_section_statuses()
        sections: list[SectionRender] = []
        for section in bb.outline:
            if not section.content:
                continue
            footnotes = [
                Footnote(number=fn.number, text=fn.text)
                for fn in bb.footnotes.get(section.id, [])
            ]
            sections.append(
                SectionRender(
                    title=section.title,
                    paragraphs=_split_paragraphs(section.content),
                    footnotes=footnotes,
                )
            )
        return PaperDocument(
            title=bb.title,
            author=author,
            date=date,
            thesis=bb.thesis,
            sections=sections,
            table_of_authorities=bb.toa,
            verification_report_md=self.verification_report(bb),
            disclaimer=DISCLAIMER,
        )

    def render_pdf(self, bb: Blackboard, out_path: str | Path, author: str = "Anonymous Scholar", date: str = "") -> Path:
        renderer = build_renderer(self.settings.pdf_renderer)
        doc = self.build_document(bb, author=author, date=date)
        return renderer.render(doc, out_path)

    def _degraded_step(self, agent: str, summary: str, stage: str) -> AgentResult:
        return AgentResult(
            agent=agent,
            summary=summary,
            payload={"degraded": True, "stage": stage},
        )

    def _seed_fallback_ideas(self, bb: Blackboard, raw_idea: str, max_ideas: int) -> int:
        # The templates below take a noun phrase; the scholar supplies a thesis,
        # which is a sentence. Interpolating it whole produced ideas like "The
        # controlling doctrinal test for Publishing open model weights is
        # protected expression, and the Export Administration Regulations may
        # not treat that publication as a deemed export" -- ungrammatical as an
        # idea, unreadable as a heading, and, worse, put to the retriever as a
        # single proposition asserting two things at once.
        #
        # The leading capital is left alone deliberately: lowercasing it would
        # be right for "Publishing" and wrong for "Congress", "First Amendment"
        # or an acronym, and legal prose is made of those.
        anchor = noun_phrase((raw_idea or bb.thesis or "the governing legal issue").strip(), 12)
        prompts = [
            f"The controlling doctrinal test for {anchor}",
            f"The strongest textual and precedential objection to {anchor}",
            f"A practical framework courts can use to resolve {anchor}",
        ]
        cap = max(1, int(max_ideas))
        created = 0
        for text in prompts[:cap]:
            bb.add_idea(
                text=text,
                angle="fallback-framework",
                novelty_note="Deterministic fallback idea produced during run-all recovery.",
            )
            created += 1
        return created

    def _ensure_minimal_outline(self, bb: Blackboard) -> int:
        if bb.outline:
            return 0
        kept = bb.selected_ideas()
        intro = bb.add_section("Introduction")
        intro.idea_ids = [i.id for i in kept]
        if kept:
            bb.add_section("Doctrinal Analysis", idea_ids=[kept[0].id])
        else:
            bb.add_section("Doctrinal Analysis")
        bb.add_section("Conclusion")
        for section in bb.outline:
            section.status = SectionStatus.IDEA
        return len(bb.outline)

    def _fallback_section_content(self, section_title: str, thesis: str, focus: str) -> str:
        issue = (focus or thesis or section_title).strip()
        return (
            f"{section_title}. The controlling legal question is {issue}. In degraded mode, "
            "this section preserves a full analytical structure: identify the governing "
            "rule, break that rule into elements, and apply each element to the strongest "
            "available facts rather than relying on conclusory labels. Where doctrine is "
            "ambiguous, the analysis should identify which authority is controlling and why "
            "its reasoning is more persuasive under text, precedent, and institutional role."
            "\n\n"
            "A complete account must also address the best counterargument. If opposing "
            "authority narrows the rule or challenges intent, the argument should distinguish "
            "those authorities, explain limits to their reach, and state the likely outcome "
            "under both narrow and broad constructions of the governing standard."
        )

    def _ensure_fallback_draft(self, bb: Blackboard, raw_idea: str) -> int:
        if not bb.outline:
            self._ensure_minimal_outline(bb)
        thesis = (bb.thesis or raw_idea or "the governing legal question").strip()
        kept = bb.selected_ideas()
        drafted = 0
        for idx, section in enumerate(bb.outline):
            if section.content.strip():
                continue
            focus = (
                kept[idx % len(kept)].text
                if kept
                else (raw_idea.strip() if raw_idea.strip() else section.title)
            )
            section.content = self._fallback_section_content(section.title, thesis, focus)
            section.citation_ids = []
            section.status = SectionStatus.DRAFTED
            drafted += 1
        return drafted

    # ---- full run (demo / tests) ---------------------------------------------------

    def run_all(
        self,
        raw_idea: str,
        title: str = "Untitled Research Paper",
        max_ideas: int = 3,
        bb: Blackboard | None = None,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> PipelineResult:
        """Run every stage in order and return the blackboard with a step log.

        ``on_event`` reports each stage as it completes. This method calls nine
        agents in sequence and can run for minutes; without it a caller sees
        nothing until the whole thing is done, which for the slowest route in
        the application is the longest silence in it.
        """

        if bb is None:
            bb = self.new_session(title)
        else:
            bb.title = title
        if raw_idea.strip() and not bb.thesis:
            bb.thesis = raw_idea.strip()
        steps: list[AgentResult] = []

        def record(result: AgentResult) -> None:
            """Append a completed stage and report it.

            Every stage in this method goes through here, including the degraded
            fallbacks -- a run where six of nine agents fell back is exactly the
            run a caller most needs to see happening, and reporting only the
            healthy path would make a limping run look like a fast one.
            """

            steps.append(result)
            if on_event is None:
                return
            with contextlib.suppress(Exception):
                on_event(
                    {
                        "event": "step_completed",
                        "index": len(steps),
                        "agent": result.agent,
                        "summary": result.summary,
                        # `_degraded_step` marks its payload; a stage that fell
                        # back is not the same event as one that succeeded, and
                        # a progress display that conflated them would report
                        # steady progress through a collapsing run.
                        "degraded": bool(result.payload.get("degraded")),
                    }
                )

        try:
            record(self.brainstorm(bb, scholar_input=raw_idea))
        except Exception:
            record(
                self._degraded_step(
                    self.interviewer.name,
                    "brainstorm unavailable; seeded thesis from scholar prompt and continued",
                    "brainstorm",
                )
            )
            if not bb.thesis:
                bb.thesis = raw_idea.strip() or "Untitled legal issue"

        try:
            record(self.ideate(bb, seed=raw_idea))
        except Exception:
            record(
                self._degraded_step(
                    self.ideator.name,
                    "ideation unavailable; switched to deterministic fallback ideas",
                    "ideate",
                )
            )

        if not bb.ideas:
            created = self._seed_fallback_ideas(bb, raw_idea, max_ideas)
            record(
                self._degraded_step(
                    self.ideator.name,
                    f"generated {created} deterministic fallback ideas",
                    "ideate-fallback",
                )
            )

        # Auto-select the first few candidate ideas (the UI lets the scholar choose).
        selections = [(idea.id, i) for i, idea in enumerate(bb.ideas[:max_ideas])]
        if selections:
            self.select_ideas(bb, selections)
        elif not bb.selected_ideas():
            created = self._seed_fallback_ideas(bb, raw_idea, max_ideas)
            self.select_ideas(
                bb,
                [(idea.id, i) for i, idea in enumerate(bb.ideas[: max(1, int(max_ideas))])],
            )
            record(
                self._degraded_step(
                    self.ideator.name,
                    f"generated {created} deterministic fallback ideas for selection",
                    "selection-fallback",
                )
            )

        try:
            record(self.build_outline(bb))
        except Exception:
            record(
                self._degraded_step(
                    self.architect.name,
                    "outline generation unavailable; created minimal fallback outline",
                    "outline",
                )
            )
        if not bb.outline:
            created = self._ensure_minimal_outline(bb)
            record(
                self._degraded_step(
                    self.architect.name,
                    f"created minimal fallback outline with {created} sections",
                    "outline-fallback",
                )
            )

        try:
            record(self.runtime.run(self.legal_researcher, self.context(bb)))
        except Exception:
            record(
                self._degraded_step(
                    self.legal_researcher.name,
                    "legal research unavailable; continued with available doctrinal context",
                    "research-legal",
                )
            )
        try:
            record(self.runtime.run(self.finance_analyst, self.context(bb)))
        except Exception:
            record(
                self._degraded_step(
                    self.finance_analyst.name,
                    "finance analysis unavailable; continued without supplemental finance notes",
                    "research-finance",
                )
            )

        draft_failed = False
        try:
            record(self.draft(bb))
        except Exception:
            draft_failed = True
            record(
                self._degraded_step(
                    self.writer.name,
                    "draft generation unavailable; switched to deterministic manuscript fallback",
                    "draft",
                )
            )

        if draft_failed or not any(section.content.strip() for section in bb.outline):
            drafted = self._ensure_fallback_draft(bb, raw_idea)
            record(
                self._degraded_step(
                    self.writer.name,
                    f"produced deterministic fallback prose for {drafted} sections",
                    "draft-fallback",
                )
            )

        try:
            record(self.edit_voice(bb))  # voice pass before citations are formatted
        except Exception:
            record(
                self._degraded_step(
                    self.editor.name,
                    "voice pass unavailable; preserved draft prose",
                    "voice",
                )
            )
        try:
            record(self.mechanism_gate(bb))  # no argument may rest on an unstated operation
        except Exception:
            record(
                self._degraded_step(
                    "Mechanism Gate",
                    "mechanism gate unavailable; prose was not scanned for unstated operations",
                    "mechanism",
                )
            )
        try:
            record(self.verify(bb))  # gatekeeper: removes unverifiable cites
        except Exception:
            record(
                self._degraded_step(
                    self.verifier_agent.name,
                    "verification unavailable; preserved manuscript and marked run as degraded",
                    "verify",
                )
            )
        try:
            record(self.format_citations(bb))  # strips removed cites, numbers footnotes
        except Exception:
            record(
                self._degraded_step(
                    self.citation_formatter.name,
                    "citation formatting unavailable; preserved inline draft text",
                    "format",
                )
            )

        try:
            self.assess_novelty(bb)
        except Exception:
            record(
                self._degraded_step(
                    "Novelty Assessor",
                    "novelty scoring unavailable; manuscript generation still completed",
                    "novelty",
                )
            )
        return PipelineResult(blackboard=bb, steps=steps)
