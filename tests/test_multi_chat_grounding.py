"""End-to-end grounding behaviour for :class:`MultiModelChat`.

Covers the three guarantees the chat panel must honour:

1. panel prompts carry a verified authority packet and citation-discipline rules;
2. fabricated authorities trigger a repair pass, and the repair is only accepted
   when it strictly reduces citation risk;
3. residual citation risk is surfaced to the reader and in the API payload.
"""

from __future__ import annotations

import pytest

import legal_research.multi_chat as multi_chat_module
from legal_research.chat_grounding import ChatCitationAuditor, GroundingStatus
from legal_research.citations.corpus import CorpusRecord, corpus_from_records
from legal_research.citations.retriever import MockRetriever
from legal_research.citations.support import LexicalSupportScorer
from legal_research.config import Settings, reset_settings
from legal_research.models import SourceType
from legal_research.multi_chat import MultiModelChat

GOOD_ANALYSIS = (
    "The controlling rule requires scienter, an intent to deceive, manipulate, or "
    "defraud, and the doctrine breaks into distinct elements that must each be "
    "satisfied on the record. The first element addresses the falsity of the "
    "statement, and the evidence here consists of internal documents and a public "
    "filing whose timeline is undisputed.\n\n"
    "Applying the standard element by element, the factual predicates establish "
    "awareness at the time of the statement, while the inference of intent depends "
    "on testimony that remains contested. The strongest counterargument is that the "
    "record shows negligence rather than deliberate deception.\n\n"
    "The dispositive uncertainty is whether the contemporaneous documents show "
    "actual awareness, because that single factual question determines whether the "
    "scienter element is satisfied under the controlling authority."
)

FABRICATED_ANALYSIS = (
    "The controlling rule on scienter comes from Texas Gulf Sulphur v. Wisenberg, "
    "471 U.S. 477 (1985), which sets out the governing elements and the burden of "
    "proof. That decision breaks the doctrine into elements that must each be "
    "satisfied by record evidence.\n\n"
    "Applying the elements to the facts, the internal documents and the public "
    "statement supply the factual predicates, while the inference of intent rests "
    "on contested testimony. The strongest counterargument is that the record "
    "establishes only negligence.\n\n"
    "The dispositive factual uncertainty is whether the documents prove awareness "
    "at the time of the statement, which controls the outcome."
)


@pytest.fixture()
def auditor() -> ChatCitationAuditor:
    corpus = corpus_from_records(
        [
            CorpusRecord(
                id="us-425-185",
                type=SourceType.CASE,
                title="Ernst & Ernst v. Hochfelder",
                reporter="U.S.",
                volume=425,
                page=185,
                court="U.S.",
                year=1976,
                passages=[
                    "Section 10(b) and Rule 10b-5 require a showing of scienter, that is, "
                    "an intent to deceive, manipulate, or defraud.",
                ],
            ),
        ]
    )
    return ChatCitationAuditor(
        corpus,
        MockRetriever(corpus),
        LexicalSupportScorer(threshold=0.2),
    )


@pytest.fixture()
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "60")
    reset_settings()
    return Settings()


def _install_llm(monkeypatch, handler) -> list[dict]:
    """Install a fake LLM and return the list that records every call."""

    calls: list[dict] = []

    class _FakeLLM:
        def __init__(self, name: str, base_url: str, model: str, api_key: str, timeout: float):
            self.name = name

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            system = messages[0].content if messages else ""
            calls.append({"name": self.name, "system": system})
            return handler(self.name, system)

        def stream(self, messages, config=None):  # noqa: ANN001, ARG002
            return iter(())

    monkeypatch.setattr(multi_chat_module, "OpenAICompatLLM", _FakeLLM)
    return calls


def test_panel_prompt_includes_authority_packet(monkeypatch, settings, auditor) -> None:
    calls = _install_llm(monkeypatch, lambda name, system: GOOD_ANALYSIS)

    chat = MultiModelChat(settings, auditor=auditor)
    chat.chat("What does scienter require under Rule 10b-5?", history=[])

    panel_systems = [c["system"] for c in calls if c["name"] == "gpt_oss"]
    assert panel_systems
    assert "VERIFIED AUTHORITY PACKET" in panel_systems[0]
    assert "Never invent case names" in panel_systems[0]
    assert "Hochfelder" in panel_systems[0]


def test_citation_discipline_applies_without_authority_packet(settings) -> None:
    chat = MultiModelChat(settings)
    discipline = chat._citation_discipline("")
    assert "Do NOT cite specific case names" in discipline


def test_fabricated_citation_triggers_repair_and_is_replaced(
    monkeypatch, settings, auditor
) -> None:
    def handler(name: str, system: str) -> str:
        if "citation integrity editor" in system:
            return GOOD_ANALYSIS
        if "legal-analysis verifier" in system:
            return "RISK: low\nCONCERN: none\nSUGGESTION: none"
        return FABRICATED_ANALYSIS

    _install_llm(monkeypatch, handler)

    chat = MultiModelChat(settings, auditor=auditor)
    result = chat.chat("What does scienter require under Rule 10b-5?", history=[])

    assert "471 U.S. 477" not in result.final_answer
    assert result.grounding.available
    assert result.grounding.clean


def test_failed_repair_keeps_original_and_warns_reader(monkeypatch, settings, auditor) -> None:
    def handler(name: str, system: str) -> str:
        if "citation integrity editor" in system:
            # Repair still cites the fabricated authority: must be rejected.
            return FABRICATED_ANALYSIS
        if "legal-analysis verifier" in system:
            return "RISK: high\nCONCERN: unverified authority\nSUGGESTION: verify"
        return FABRICATED_ANALYSIS

    _install_llm(monkeypatch, handler)

    chat = MultiModelChat(settings, auditor=auditor)
    result = chat.chat("What does scienter require under Rule 10b-5?", history=[])

    assert "Citation grounding check" in result.final_answer
    assert "471 U.S. 477" in result.final_answer
    assert result.grounding.fabricated
    statuses = {f.status for f in result.grounding.findings}
    assert GroundingStatus.NOT_IN_CORPUS in statuses


def test_clean_answer_gets_no_grounding_notice(monkeypatch, settings, auditor) -> None:
    _install_llm(
        monkeypatch,
        lambda name, system: (
            "RISK: low\nCONCERN: none\nSUGGESTION: none"
            if "legal-analysis verifier" in system
            else GOOD_ANALYSIS
        ),
    )

    chat = MultiModelChat(settings, auditor=auditor)
    result = chat.chat("What does scienter require under Rule 10b-5?", history=[])

    assert "Citation grounding check" not in result.final_answer
    assert result.grounding.clean


def test_grounding_degrades_gracefully_when_corpus_missing(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "openai")
    monkeypatch.setenv("LRG_CORPUS_PATH", "/nonexistent/corpus.jsonl")
    monkeypatch.setenv("LRG_MULTI_CHAT_MAX_LATENCY_SECONDS", "60")
    reset_settings()

    _install_llm(monkeypatch, lambda name, system: GOOD_ANALYSIS)

    chat = MultiModelChat(Settings())
    result = chat.chat("What does scienter require under Rule 10b-5?", history=[])

    assert result.final_answer
    assert result.grounding.available is False
    assert "unavailable" in result.grounding.summary().lower()


def test_grounding_can_be_disabled(monkeypatch, settings) -> None:
    monkeypatch.setenv("LRG_MULTI_CHAT_GROUNDING_ENABLED", "false")
    reset_settings()
    _install_llm(monkeypatch, lambda name, system: GOOD_ANALYSIS)

    chat = MultiModelChat(Settings())
    result = chat.chat("What does scienter require under Rule 10b-5?", history=[])

    assert result.grounding.available is False
    assert "disabled" in result.grounding.note.lower()


def test_api_multi_chat_exposes_grounding(monkeypatch) -> None:
    monkeypatch.setenv("LRG_LLM_MODE", "mock")
    monkeypatch.setenv("LRG_RETRIEVER_MODE", "mock")
    # This test sends no key, so it must not inherit one from the developer's
    # .env -- see the same pin in tests/test_api.py. It failed in isolation and
    # passed in the full suite purely on import order, because the app binds
    # ApiKeyMiddleware once at first import.
    monkeypatch.setenv("LRG_API_KEY", "")
    reset_settings()

    from fastapi.testclient import TestClient

    from legal_research.api.app import app

    client = TestClient(app)
    session_id = client.post("/api/sessions", json={"title": "grounding"}).json()["session_id"]
    resp = client.post(
        f"/api/sessions/{session_id}/multi-chat",
        json={"message": "What does scienter require under Rule 10b-5?", "history": []},
    )

    assert resp.status_code == 200
    grounding = resp.json()["grounding"]
    assert set(grounding) >= {
        "available",
        "summary",
        "findings",
        "authorities",
        "verified_count",
        "unverified_count",
    }
