"""End-to-end pipeline (spec §0).

Wires the expert pool, router, retriever, corpus, formatter, guard and verifier into
an :class:`AgentContext`, then drives the nine agents through the stages: brainstorm
-> ideate -> select -> outline -> research -> draft -> verify -> format -> voice ->
novelty -> document/PDF. Operates on a caller-supplied :class:`Blackboard` so the API
can keep one per session.
"""

from __future__ import annotations

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
from .models import IdeaStatus
from .pdf.base import Footnote, PaperDocument, SectionRender, build_renderer
from .router import Router
from .voice.novelty import assess_novelty


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
        guard = CitationGuard(self.retriever, bb.new_citation_id)
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

    def draft(self, bb: Blackboard) -> AgentResult:
        return self.runtime.run(self.writer, self.context(bb))

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

    def assess_novelty(self, bb: Blackboard) -> None:
        bb.novelty = assess_novelty(bb.thesis or "", bb.retrieved)

    # ---- outputs -------------------------------------------------------------------

    def verification_report(self, bb: Blackboard) -> str:
        return build_verification_report(bb.citations, bb.verifications, self.corpus)

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
                SectionRender(title=section.title, paragraphs=[section.content], footnotes=footnotes)
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

    # ---- full run (demo / tests) ---------------------------------------------------

    def run_all(
        self, raw_idea: str, title: str = "Untitled Research Paper", max_ideas: int = 3
    ) -> PipelineResult:
        bb = self.new_session(title)
        steps: list[AgentResult] = []
        steps.append(self.brainstorm(bb, scholar_input=raw_idea))
        steps.append(self.ideate(bb, seed=raw_idea))

        # Auto-select the first few candidate ideas (the UI lets the scholar choose).
        selections = [(idea.id, i) for i, idea in enumerate(bb.ideas[:max_ideas])]
        self.select_ideas(bb, selections)

        steps.append(self.build_outline(bb))
        steps.extend(self.research(bb))
        steps.append(self.draft(bb))
        steps.append(self.edit_voice(bb))  # voice pass before citations are formatted
        steps.append(self.verify(bb))  # gatekeeper: removes unverifiable cites
        steps.append(self.format_citations(bb))  # strips removed cites, numbers footnotes
        self.assess_novelty(bb)
        return PipelineResult(blackboard=bb, steps=steps)
