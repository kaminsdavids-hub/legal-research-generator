"""Editor grammar-chain behavior and citation-token safety."""

from __future__ import annotations

import re

from legal_research.agents.writer import WriterAgent
from legal_research.blackboard import Blackboard
from legal_research.llm.base import ChatMessage, GenerationConfig
from legal_research.llm.pool import GEMMA, HERMES, HERMES3, SAUL
from legal_research.models import Citation, CiteStatus
from legal_research.pipeline import LegalResearchPipeline

_TOKEN = re.compile(r"\{\{cite:[^}]+\}\}")


class _SuffixLLM:
    def __init__(self, name: str, suffix: str) -> None:
        self.name = name
        self._suffix = suffix

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> str:
        text = ""
        for m in reversed(messages):
            if m.role == "user":
                text = m.content
                break
        return f"{text.strip()} {self._suffix}".strip()

    def stream(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ):
        yield self.chat(messages, config)


def test_grammar_chain_preserves_citation_tokens(
    pipeline: LegalResearchPipeline, blackboard: Blackboard
) -> None:
    sec = blackboard.add_section("Introduction")
    sec.content = (
        "Courts demand scienter in Rule 10b-5 actions.{{cite:cite-001}}\n\n"
        "That doctrine separates fraud from negligence.{{cite:cite-002}}"
    )
    sec.citation_ids = ["cite-001", "cite-002"]

    blackboard.add_citation(
        Citation(
            id="cite-001",
            record_id="us-425-185",
            proposition="Section 10(b) requires scienter.",
            from_retrieval=True,
            status=CiteStatus.PENDING,
        )
    )
    blackboard.add_citation(
        Citation(
            id="cite-002",
            record_id="us-425-185",
            proposition="Scienter distinguishes fraud from negligence.",
            from_retrieval=True,
            status=CiteStatus.PENDING,
        )
    )

    pipeline.pool.clients[HERMES] = _SuffixLLM(HERMES, "(nano)")
    pipeline.pool.clients[GEMMA] = _SuffixLLM(GEMMA, "(gemma)")
    pipeline.pool.clients[HERMES3] = _SuffixLLM(HERMES3, "(hermes3)")

    before = _TOKEN.findall(sec.content)
    result = pipeline.edit_voice(blackboard)
    after = _TOKEN.findall(sec.content)

    assert before == after
    assert result.payload["grammar_sections"] == 1
    assert result.payload["grammar_roles"] == ["gemma", "hermes", "hermes3"]


def test_writer_agent_is_saul_first() -> None:
    assert WriterAgent.expert_role == SAUL
