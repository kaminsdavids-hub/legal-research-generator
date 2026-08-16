"""Multi-model chat module with dual-model verification.

Runs a fixed panel of local models, synthesizes a single answer, then asks two
verifier models to critique and score grounding.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from .chat_grounding import (
    AuthorityPacket,
    ChatCitationAuditor,
    GroundingReport,
    format_grounding_notice,
)
from .config import Settings
from .llm.base import ChatMessage, DecodingPolicy, LLMClient
from .llm.mock import MockLLM
from .llm.openai_compat import OpenAICompatLLM

ChatRole = Literal["user", "assistant"]

# Largest share of the total latency budget that may be held back for the
# post-panel stages, leaving the rest for the panel itself.
_MAX_RESERVE_FRACTION = 0.55


@dataclass(frozen=True)
class MultiChatTurn:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class ModelAnswer:
    model: str
    content: str


@dataclass(frozen=True)
class VerifierResult:
    model: str
    verdict: str


@dataclass(frozen=True)
class MultiChatResult:
    final_answer: str
    model_answers: list[ModelAnswer]
    verifiers: list[VerifierResult]
    grounding: GroundingReport = field(default_factory=GroundingReport)


class MultiModelChat:
    """Drives the secondary chat module from dedicated model settings."""

    def __init__(
        self,
        settings: Settings,
        auditor: ChatCitationAuditor | None = None,
    ) -> None:
        self._settings = settings
        self._auditor = auditor
        self._auditor_ready = auditor is not None
        self._auditor_note = ""
        if settings.llm_warmup_enabled:
            threading.Thread(target=self._warm_up_panel, daemon=True).start()

    # -- grounding ---------------------------------------------------------------

    def _warm_up_panel(self) -> None:
        """Preload each panel/verifier model so the first user request is not
        the one that pays the Ollama load cost."""

        models = [
            ("gpt_oss", self._settings.multi_chat_gpt_oss_model),
            ("gemma4", self._settings.multi_chat_gemma4_model),
            ("apertus", self._settings.multi_chat_apertus_model),
            ("nemotron", self._settings.multi_chat_nemotron_model),
            ("hermes3", self._settings.multi_chat_hermes3_model),
            ("gemma3_verifier", self._settings.multi_chat_verifier_gemma3_model),
            ("saul_verifier", self._settings.multi_chat_verifier_saul_model),
        ]
        for name, model in models:
            client = self._client(name, model, timeout_seconds=120.0)
            if callable(getattr(client, "warm_up", None)):
                with contextlib.suppress(Exception):
                    client.warm_up(self._settings.llm_warmup_prompt)

    def _get_auditor(self) -> ChatCitationAuditor | None:
        """Lazily construct the citation auditor.

        Corpus/retriever construction is expensive and can fail (missing corpus
        file, absent optional dependencies). Grounding is best-effort: on failure
        the chat proceeds unguarded and the report records why.
        """

        if self._auditor_ready:
            return self._auditor
        self._auditor_ready = True
        if not self._settings.multi_chat_grounding_enabled:
            self._auditor_note = "Citation grounding is disabled by configuration."
            return None
        try:
            from .citations.corpus import load_corpus
            from .citations.retriever import build_retriever
            from .citations.support import build_support_scorer

            corpus = load_corpus(self._settings.corpus_path)
            self._auditor = ChatCitationAuditor(
                corpus,
                build_retriever(self._settings, corpus),
                build_support_scorer(self._settings),
                min_support=self._settings.multi_chat_grounding_min_support,
            )
        except Exception as exc:  # noqa: BLE001 - grounding must never break chat
            self._auditor = None
            self._auditor_note = f"Citation grounding unavailable ({type(exc).__name__})."
        return self._auditor

    def _client(self, name: str, model: str, timeout_seconds: float | None = None) -> LLMClient:
        if self._settings.llm_mode == "mock":
            return MockLLM(name)
        timeout = self._settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
        return OpenAICompatLLM(
            name=name,
            base_url=self._settings.multi_chat_base_url,
            model=model,
            api_key=self._settings.llm_api_key,
            timeout=max(5.0, float(timeout)),
        )

    def _remaining_seconds(self, started_at: float) -> float:
        return float(self._settings.multi_chat_max_latency_seconds) - (
            time.monotonic() - started_at
        )

    @staticmethod
    def _stage_timeout_seconds(
        remaining_seconds: float, stage_cap_seconds: float, reserve_seconds: float = 0.0
    ) -> float:
        """Timeout for one stage, leaving ``reserve_seconds`` for what comes after.

        Without a reserve, an early stage is free to consume the entire remaining
        budget and every later stage then sees ``remaining <= 0`` and emits its
        canned "time budget exceeded" text.
        """

        budget = min(float(stage_cap_seconds), remaining_seconds - max(0.0, reserve_seconds))
        return max(5.0, budget)

    def _downstream_reserve_seconds(self) -> float:
        """Time that must survive the panel for synthesis, grounding, and verifiers.

        These stages are what turn five separate analyses into an answer, so they
        are reserved up front rather than left to whatever the panel does not
        spend. The panel's own share is the remainder.

        The reserve is clamped to :data:`_MAX_RESERVE_FRACTION` of the total
        budget. The stage caps can exceed the whole latency budget on their own
        (they sum to ~121s against a 60s budget in the test profile), and an
        unclamped reserve would then skip every panel model -- the same
        over-subscription failure, just moved to the other end of the pipeline.
        """

        s = self._settings
        total = float(s.multi_chat_max_latency_seconds)
        wanted = (
            float(s.multi_chat_synthesis_timeout_seconds)
            + float(s.multi_chat_citation_repair_timeout_seconds)
            + 2.0 * float(s.multi_chat_per_verifier_timeout_seconds)
        )
        return min(wanted, total * _MAX_RESERVE_FRACTION)

    def _panel_budget(self, panel_size: int) -> tuple[float, float]:
        """Return ``(wall_budget, per_model_timeout)`` for the panel.

        The configured caps deliberately over-subscribe the total latency budget
        (5x75 + 45 + 40 + 2x18 far exceeds 260s), so the per-model timeout is
        derived instead of trusted: the panel gets whatever the downstream reserve
        leaves, divided by the number of concurrency waves, and capped by
        ``multi_chat_per_model_timeout_seconds``.
        """

        s = self._settings
        wall_budget = max(
            10.0, float(s.multi_chat_max_latency_seconds) - self._downstream_reserve_seconds()
        )
        size = max(1, panel_size)
        concurrency = max(1, min(int(s.multi_chat_panel_concurrency), size))
        waves = math.ceil(size / concurrency)
        per_model = min(float(s.multi_chat_per_model_timeout_seconds), wall_budget / waves)
        return wall_budget, max(5.0, per_model)

    @staticmethod
    def _best_panel_answer(model_answers: list[ModelAnswer]) -> str:
        candidates = [
            answer.content.strip()
            for answer in model_answers
            if answer.content.strip() and not answer.content.strip().startswith("(")
        ]
        if not candidates:
            return ""
        return max(candidates, key=len)

    @staticmethod
    def _shorten_prompt(prompt: str, max_chars: int = 380) -> str:
        cleaned = " ".join(prompt.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _analysis_signal_score(text: str) -> tuple[int, int, int]:
        normalized = MultiModelChat._normalize_text(text)
        doctrine_terms = (
            "controlling",
            "doctrine",
            "rule",
            "standard",
            "element",
            "test",
            "precedent",
            "authority",
            "counterargument",
            "burden",
            "scienter",
            "causation",
            "materiality",
        )
        fact_terms = (
            "fact",
            "facts",
            "record",
            "evidence",
            "alleg",
            "inference",
            "uncertain",
            "uncertainty",
            "timeline",
            "document",
            "witness",
            "statement",
        )
        boilerplate_terms = (
            "it depends",
            "cannot be determined",
            "more information is needed",
            "consult a lawyer",
            "in conclusion",
            "overall",
            "this is a complex issue",
            "various factors",
            "many factors",
            "both sides have arguments",
        )
        doctrine_hits = sum(1 for term in doctrine_terms if term in normalized)
        fact_hits = sum(1 for term in fact_terms if term in normalized)
        boilerplate_hits = sum(1 for term in boilerplate_terms if term in normalized)
        return doctrine_hits, fact_hits, boilerplate_hits

    @staticmethod
    def _is_weak_analysis(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True
        if cleaned.startswith("("):
            return True
        normalized = MultiModelChat._normalize_text(cleaned)
        if len(normalized) < 260:
            return True
        doctrine_hits, fact_hits, boilerplate_hits = MultiModelChat._analysis_signal_score(cleaned)
        if doctrine_hits < 2:
            return True
        if fact_hits < 1:
            return True
        return bool(boilerplate_hits >= 2 and doctrine_hits < 4)

    @staticmethod
    def _deterministic_analytic_fallback(prompt: str) -> str:
        topic = MultiModelChat._shorten_prompt(prompt, max_chars=220)
        return (
            "Degraded-mode analytical answer (models unavailable): "
            f"On the question '{topic}', the most defensible approach is to proceed in three steps.\n\n"
            "First, identify the controlling legal rule and break it into explicit elements or"
            " sub-tests, rather than treating the doctrine as a single conclusion. Second, map"
            " the available facts to each element with adversarial discipline: note where proof"
            " is strong, where it is ambiguous, and which inferences rely on contested"
            " assumptions. Third, test the argument against the strongest counter-position and"
            " explain why that counter-position does or does not defeat liability under the"
            " governing standard.\n\n"
            "Factually, the key unknowns are usually mental-state evidence, causation links,"
            " and whether alternative explanations fit the record better. A high-integrity"
            " answer should therefore state what is known, what remains uncertain, and what"
            " additional authority or facts would most likely change the outcome. To avoid"
            " boilerplate reasoning, a serious answer should compare competing factual"
            " narratives, explain which narrative better satisfies each doctrinal element, and"
            " identify the decisive factual predicates a court would likely treat as"
            " outcome-determinative."
        )

    @staticmethod
    def _is_unavailable_answer(text: str) -> bool:
        t = text.strip().lower()
        return (
            not t
            or t.startswith("(unavailable:")
            or t.startswith("(skipped:")
            or t.startswith("(no response")
        )

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
        return type(exc).__name__.lower() in {"readtimeout", "timeoutexception", "timeout"}

    @staticmethod
    def answered(content: str) -> bool:
        """Whether a panel slot holds a real model answer.

        Every substitute this module produces announces itself, and this is the
        one place that knows the whole list. It exists because the progress
        events reported `answered=True` for any non-empty string, which made a
        panel where all five models timed out and were replaced with canned text
        render as five successes -- the exact conflation the run-all recorder
        was careful to avoid, reintroduced one module over.
        """

        stripped = content.strip()
        if not stripped:
            return False
        return not stripped.startswith(
            ("Degraded panel answer (", "(skipped:", "(no response)", "(unavailable:")
        )

    @staticmethod
    def _timeout_model_fallback(prompt: str, model: str) -> str:
        topic = MultiModelChat._shorten_prompt(prompt, max_chars=180)
        return (
            f"Degraded panel answer ({model}; timeout recovered): "
            f"For '{topic}', begin with the controlling legal standard and map each factual "
            "assertion to a specific element of that standard. Distinguish direct evidence from "
            "inference, identify the strongest counterargument, and explain what factual or "
            "doctrinal uncertainty remains outcome-determinative."
        )

    @staticmethod
    def _low_substance_model_fallback(prompt: str, model: str) -> str:
        topic = MultiModelChat._shorten_prompt(prompt, max_chars=180)
        return (
            f"Degraded panel answer ({model}; low-substance rewrite fallback): "
            f"For '{topic}', the analysis should identify the controlling legal standard, "
            "break that standard into elements, and test each element against concrete record "
            "facts. It should then confront the strongest contrary authority or factual "
            "inference, explain why that challenge narrows (or defeats) the claim, and specify "
            "which unresolved factual predicates are most likely to control disposition."
        )

    def _upgrade_panel_answer_if_needed(
        self,
        *,
        name: str,
        model: str,
        prompt: str,
        prior: list[ChatMessage],
        started_at: float,
        draft: str,
    ) -> str:
        if self._is_unavailable_answer(draft):
            return draft
        if not self._is_weak_analysis(draft):
            return draft

        remaining = self._remaining_seconds(started_at)
        if remaining <= 8.0:
            return self._low_substance_model_fallback(prompt, model)

        timeout = self._stage_timeout_seconds(
            remaining,
            max(8.0, self._settings.multi_chat_per_model_timeout_seconds * 0.55),
        )
        client = self._client(name, model, timeout_seconds=timeout)
        concise_prompt = self._shorten_prompt(prompt, max_chars=260)
        try:
            revised = client.chat(
                [
                    ChatMessage(
                        "system",
                        "You are rewriting a legal panel draft that was too generic. "
                        "Return 3 dense paragraphs of original legal analysis, not bullets. "
                        "Paragraph 1: state the controlling rule and element structure. "
                        "Paragraph 2: map concrete factual predicates to each element and "
                        "separate direct proof from inference. Paragraph 3: present the "
                        "strongest counterargument and explain exactly which factual uncertainty "
                        "is outcome-determinative. Ban boilerplate and generic caveats.",
                    ),
                    *prior,
                    ChatMessage(
                        "user",
                        (f"Question:\n{concise_prompt}\n\n" f"Weak draft to improve:\n{draft}"),
                    ),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
            if revised and not self._is_weak_analysis(revised):
                return revised
        except Exception:  # noqa: BLE001
            pass

        return self._low_substance_model_fallback(prompt, model)

    def _elevate_final_answer_if_needed(
        self,
        *,
        prompt: str,
        prior: list[ChatMessage],
        model_answers: list[ModelAnswer],
        started_at: float,
        draft: str,
    ) -> str:
        if not self._is_weak_analysis(draft):
            return draft

        remaining = self._remaining_seconds(started_at)
        if remaining > 0:
            timeout = self._stage_timeout_seconds(
                remaining,
                max(8.0, self._settings.multi_chat_rescue_timeout_seconds * 0.7),
            )
            rescue = self._client(
                "panel_rescue",
                self._settings.multi_chat_rescue_model,
                timeout_seconds=timeout,
            )
            combined = "\n\n".join(f"[{a.model}]\n{a.content}" for a in model_answers)
            try:
                revised = rescue.chat(
                    [
                        ChatMessage(
                            "system",
                            "You are the final merits editor for a legal analysis jury. "
                            "The candidate answer was too generic. Rewrite it into a "
                            "rigorous, intellectually distinctive response in 3-5 paragraphs. "
                            "Must include: controlling doctrinal rule, explicit element-by-"
                            "element factual application, strongest opposing interpretation, and "
                            "a clear explanation of which unresolved facts are dispositive.",
                        ),
                        *prior,
                        ChatMessage(
                            "user",
                            (
                                f"Question:\n{prompt}\n\n"
                                f"Panel materials:\n{combined}\n\n"
                                f"Weak candidate answer to rewrite:\n{draft}"
                            ),
                        ),
                    ],
                    DecodingPolicy.BALANCED.config,
                ).strip()
                if revised and not self._is_weak_analysis(revised):
                    return revised
            except Exception:  # noqa: BLE001
                pass

        best_panel = self._best_panel_answer(model_answers)
        if best_panel and not self._is_weak_analysis(best_panel):
            return best_panel
        return self._deterministic_analytic_fallback(prompt)

    def _repair_citations(
        self,
        *,
        prompt: str,
        draft: str,
        report: GroundingReport,
        authority_packet: str,
        started_at: float,
    ) -> str:
        """Ask the rescue model to remove or replace unverifiable authorities."""

        remaining = self._remaining_seconds(started_at)
        if remaining <= 8.0:
            return ""

        timeout = self._stage_timeout_seconds(
            remaining,
            max(10.0, self._settings.multi_chat_citation_repair_timeout_seconds),
        )
        client = self._client(
            "citation_repair",
            self._settings.multi_chat_rescue_model,
            timeout_seconds=timeout,
        )

        problem_lines = "\n".join(
            f"- {finding.citation}: {finding.detail}" for finding in report.problems
        )
        packet_block = f"\n\n{authority_packet}" if authority_packet else ""

        try:
            revised = client.chat(
                [
                    ChatMessage(
                        "system",
                        "You are a citation integrity editor for legal analysis. "
                        "The draft below cites authorities that could not be verified "
                        "against the verified corpus. Rewrite the draft so that every "
                        "remaining citation is either drawn from the verified authority "
                        "packet or removed entirely. Do NOT invent case names, reporter "
                        "cites, statutes, or regulations. When you remove a citation, "
                        "preserve the underlying legal reasoning and state the principle "
                        "in unattributed terms (for example, 'the prevailing standard "
                        "requires...'). Keep the analytical depth, element-by-element "
                        "structure, and counterargument intact. Return only the rewritten "
                        "analysis.",
                    ),
                    ChatMessage(
                        "user",
                        (
                            f"Question:\n{prompt}\n\n"
                            f"Unverifiable citations:\n{problem_lines}{packet_block}\n\n"
                            f"Draft to correct:\n{draft}"
                        ),
                    ),
                ],
                DecodingPolicy.COLD.config,
            ).strip()
        except Exception:  # noqa: BLE001
            return ""

        if not revised or self._is_weak_analysis(revised):
            return ""
        return revised

    def _ground_final_answer(
        self,
        *,
        prompt: str,
        draft: str,
        packet: AuthorityPacket,
        started_at: float,
    ) -> tuple[str, GroundingReport]:
        """Audit citations, attempt one repair pass, then annotate residual risk."""

        auditor = self._get_auditor()
        if auditor is None:
            return draft, GroundingReport(available=False, note=self._auditor_note)

        try:
            report = auditor.audit(draft, retrieved_ids=packet.record_ids)
        except Exception as exc:  # noqa: BLE001
            return draft, GroundingReport(
                available=False,
                note=f"Citation audit failed ({type(exc).__name__}).",
            )

        answer = draft
        if report.problems:
            repaired = self._repair_citations(
                prompt=prompt,
                draft=draft,
                report=report,
                authority_packet=packet.text,
                started_at=started_at,
            )
            if repaired:
                try:
                    repaired_report = auditor.audit(repaired, retrieved_ids=packet.record_ids)
                except Exception:  # noqa: BLE001
                    repaired_report = None
                # Only accept the rewrite if it strictly reduces citation risk.
                if repaired_report is not None and len(repaired_report.problems) < len(
                    report.problems
                ):
                    answer, report = repaired, repaired_report

        notice = format_grounding_notice(report)
        if notice:
            answer = f"{answer}\n\n---\n\n{notice}"
        return answer, report

    @staticmethod
    def _citation_discipline(authority_packet: str) -> str:
        if authority_packet:
            return (
                "\n\nCITATION DISCIPLINE: You may cite ONLY the authorities listed in the "
                "verified authority packet below. Never invent case names, reporter "
                "citations, statutes, or regulations, and never guess a volume or page "
                "number. If the packet does not support a point, state the legal principle "
                "without attributing it to a specific case.\n\n" + authority_packet
            )
        return (
            "\n\nCITATION DISCIPLINE: No verified authority packet is available for this "
            "question. Do NOT cite specific case names, reporter citations, statutes, or "
            "regulations from memory, because unverified citations are treated as errors. "
            "State legal principles in unattributed terms instead."
        )

    def _query_panel_model(
        self,
        *,
        name: str,
        model: str,
        prompt: str,
        prior: list[ChatMessage],
        user_turn: ChatMessage,
        started_at: float,
        authority_packet: str = "",
        per_model_timeout: float | None = None,
    ) -> str:
        # The panel must not eat the time synthesis and verification need, so the
        # reserve is subtracted here rather than after the panel has spent it.
        reserve = self._downstream_reserve_seconds()
        remaining = self._remaining_seconds(started_at)
        if remaining - reserve <= 0:
            return "(skipped: model jury time budget exceeded)"

        cap = (
            self._settings.multi_chat_per_model_timeout_seconds
            if per_model_timeout is None
            else per_model_timeout
        )
        timeout = self._stage_timeout_seconds(remaining, cap, reserve_seconds=reserve)
        client = self._client(name, model, timeout_seconds=timeout)
        try:
            content = client.chat(
                [
                    ChatMessage(
                        "system",
                        # 1-2 paragraphs, not 3-4. Measured on this hardware: five
                        # models writing four paragraphs each take 114-158s under
                        # mutual contention, against a 139s panel budget, so every
                        # model but the smallest timed out and was replaced with
                        # canned text -- a panel that produced nothing at all. The
                        # analytical requirements below are unchanged; only the
                        # length is cut, because length is what costs. The
                        # synthesis discards most of the prose anyway, and the
                        # retry prompt already asked for 1-2 paragraphs, so this
                        # also stops the first attempt and its retry disagreeing
                        # about what a good answer looks like.
                        "You are one member of a legal-research chat panel. "
                        "Write a rigorous doctrinal analysis in 1-2 tight paragraphs, not bullets. "
                        "Identify the controlling rule, break it into explicit elements/tests, "
                        "map concrete factual predicates to each element, and stress-test the "
                        "result against the strongest counterargument. Distinguish proven facts "
                        "from inferences and state exactly which uncertainties are "
                        "outcome-determinative. Ban boilerplate language."
                        + self._citation_discipline(authority_packet),
                    ),
                    *prior,
                    user_turn,
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
            if not content:
                return "(no response)"
            return self._upgrade_panel_answer_if_needed(
                name=name,
                model=model,
                prompt=prompt,
                prior=prior,
                started_at=started_at,
                draft=content,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_timeout_exception(exc):
                return f"(unavailable: {type(exc).__name__})"

        retry_remaining = self._remaining_seconds(started_at)
        if retry_remaining - reserve > 0:
            retry_timeout = self._stage_timeout_seconds(
                retry_remaining,
                max(8.0, cap * 0.7),
                reserve_seconds=reserve,
            )
            retry_client = self._client(name, model, timeout_seconds=retry_timeout)
            concise_prompt = self._shorten_prompt(prompt, max_chars=220)
            try:
                retried = retry_client.chat(
                    [
                        ChatMessage(
                            "system",
                            "Produce a compact but substantive legal analysis. "
                            "Use 1-2 short paragraphs; prioritize controlling doctrine, "
                            "element-by-element application, and the strongest counterargument.",
                        ),
                        *prior,
                        ChatMessage("user", concise_prompt),
                    ],
                    DecodingPolicy.COLD.config,
                ).strip()
                if retried:
                    return self._upgrade_panel_answer_if_needed(
                        name=name,
                        model=model,
                        prompt=prompt,
                        prior=prior,
                        started_at=started_at,
                        draft=retried,
                    )
            except Exception:  # noqa: BLE001
                pass

        return self._timeout_model_fallback(prompt, model)

    def _needs_rescue_synthesis(self, model_answers: list[ModelAnswer]) -> bool:
        return all(self._is_unavailable_answer(answer.content) for answer in model_answers)

    def _rescue_final_answer(
        self,
        prompt: str,
        prior: list[ChatMessage],
        started_at: float,
        model_answers: list[ModelAnswer] | None = None,
        force: bool = False,
    ) -> str:
        remaining = self._remaining_seconds(started_at)
        if remaining <= 0 and not force:
            return ""
        effective_remaining = remaining if remaining > 0 else 5.0
        timeout = self._stage_timeout_seconds(
            effective_remaining,
            max(
                self._settings.multi_chat_rescue_timeout_seconds,
                self._settings.multi_chat_synthesis_timeout_seconds,
            ),
        )
        rescue = self._client(
            "panel_rescue",
            self._settings.multi_chat_rescue_model,
            timeout_seconds=timeout,
        )
        available_panel = self._best_panel_answer(model_answers or [])
        panel_context = ""
        if available_panel:
            panel_context = (
                "\n\nBest available panel draft (use this if useful, but improve precision):\n"
                f"{available_panel}"
            )
        concise_prompt = self._shorten_prompt(prompt)
        try:
            return rescue.chat(
                [
                    ChatMessage(
                        "system",
                        "You are a legal research assistant in degraded mode. "
                        "The multi-model jury did not produce a reliable synthesis. "
                        "Answer the user's question directly in a dense analytical format "
                        "(2-4 coherent paragraphs), prioritizing governing doctrine, "
                        "elements/tests, and strongest counterarguments. Provide factual "
                        "uncertainty boundaries clearly and avoid generic filler.",
                    ),
                    *prior,
                    ChatMessage("user", f"User question:\n{concise_prompt}{panel_context}"),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
        except Exception:  # noqa: BLE001
            return ""

    def _fallback_final_answer(self, prompt: str, model_answers: list[ModelAnswer]) -> str:
        best_panel = self._best_panel_answer(model_answers)
        if best_panel:
            return best_panel
        return self._deterministic_analytic_fallback(prompt)

    def _history_messages(self, history: list[MultiChatTurn]) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for turn in history:
            role: ChatRole = "assistant" if turn.role == "assistant" else "user"
            text = turn.content.strip()
            if not text:
                continue
            messages.append(ChatMessage(role, text))
        return messages

    def chat(
        self,
        message: str,
        history: list[MultiChatTurn] | None = None,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> MultiChatResult:
        """Run the panel, synthesise, verify, and return the whole exchange.

        ``on_event`` receives progress as it happens: which model has answered,
        how long it took, how much it said. A five-model panel takes tens of
        seconds and returns nothing until the last one is done, so without this
        a caller can only show a spinner and hope. It is called from the pool
        threads, so it must be thread-safe, and it is wrapped so that a callback
        which raises cannot break the exchange -- reporting on the work must not
        be able to destroy the work.
        """

        prompt = message.strip()
        if not prompt:
            raise ValueError("message must not be empty")

        def report(kind: str, **fields: object) -> None:
            if on_event is None:
                return
            with contextlib.suppress(Exception):
                # Reporting on the work must not be able to destroy the work.
                on_event({"event": kind, **fields})

        started_at = time.monotonic()
        prior = self._history_messages(history or [])
        user_turn = ChatMessage("user", prompt)

        panel_specs = [
            ("gpt_oss", self._settings.multi_chat_gpt_oss_model),
            ("gemma4", self._settings.multi_chat_gemma4_model),
            ("apertus", self._settings.multi_chat_apertus_model),
            ("nemotron", self._settings.multi_chat_nemotron_model),
            ("hermes3", self._settings.multi_chat_hermes3_model),
        ]

        packet = AuthorityPacket()
        auditor = self._get_auditor()
        if auditor is not None:
            try:
                packet = auditor.authority_packet(prompt, k=self._settings.multi_chat_grounding_k)
            except Exception:  # noqa: BLE001 - retrieval must never break the chat
                packet = AuthorityPacket()
        authority_packet = packet.text

        _, per_model_timeout = self._panel_budget(len(panel_specs))

        report("panel_started", models=[model for _, model in panel_specs])

        def _ask(spec: tuple[str, str]) -> ModelAnswer:
            name, model = spec
            began = time.monotonic()
            content = self._query_panel_model(
                name=name,
                model=model,
                prompt=prompt,
                prior=prior,
                user_turn=user_turn,
                started_at=started_at,
                authority_packet=authority_packet,
                per_model_timeout=per_model_timeout,
            )
            # Emitted in completion order, not the configured order the results
            # are collected in. Completion order is what the person waiting sees.
            report(
                "model_answered",
                name=name,
                model=model,
                seconds=round(time.monotonic() - began, 1),
                characters=len(content),
                # Not `bool(content)`: a model that timed out comes back with a
                # canned substitute, which is non-empty and is not an answer.
                answered=self.answered(content),
            )
            return ModelAnswer(model=model, content=content)

        concurrency = max(1, min(self._settings.multi_chat_panel_concurrency, len(panel_specs)))
        if concurrency == 1:
            model_answers = [_ask(spec) for spec in panel_specs]
        else:
            # `map` preserves input order, so the panel is reported in its
            # configured order regardless of which model finishes first.
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                model_answers = list(pool.map(_ask, panel_specs))

        # Real answers, not non-empty strings: the same distinction the
        # per-model event draws. "synthesising 5 answers" over five canned
        # substitutes is the panel congratulating itself on having failed.
        report("synthesising", answers=sum(1 for a in model_answers if self.answered(a.content)))

        final_answer = ""
        remaining = self._remaining_seconds(started_at)
        # Synthesis in turn leaves room for grounding repair and both verifiers.
        synthesis_reserve = self._downstream_reserve_seconds() - float(
            self._settings.multi_chat_synthesis_timeout_seconds
        )
        if remaining > 0:
            timeout = self._stage_timeout_seconds(
                remaining,
                self._settings.multi_chat_synthesis_timeout_seconds,
                reserve_seconds=synthesis_reserve,
            )
            synthesis = self._client(
                "panel_synth", self._settings.multi_chat_gpt_oss_model, timeout_seconds=timeout
            )
            combined = "\n\n".join(f"[{a.model}]\n{a.content}" for a in model_answers)
            try:
                final_answer = synthesis.chat(
                    [
                        ChatMessage(
                            "system",
                            "You are synthesizing answers from multiple local models. "
                            "Produce one integrated, detailed prosaic answer (roughly 3-6 "
                            "paragraphs) that preserves legal nuance, cites controlling standards "
                            "at a high level, explicitly notes uncertainty, and avoids overclaiming. "
                            "Prefer smooth narrative over bullet points. Discard any citation from "
                            "the panel that does not appear in the verified authority packet."
                            + self._citation_discipline(authority_packet),
                        ),
                        ChatMessage(
                            "user", f"User question:\n{prompt}\n\nPanel responses:\n{combined}"
                        ),
                    ],
                    DecodingPolicy.BALANCED.config,
                ).strip()
            except Exception:  # noqa: BLE001
                final_answer = ""
        if not final_answer:
            rescued = self._rescue_final_answer(
                prompt,
                prior,
                started_at,
                model_answers=model_answers,
                force=self._needs_rescue_synthesis(model_answers),
            )
            if rescued:
                final_answer = rescued
        if not final_answer:
            final_answer = self._fallback_final_answer(prompt, model_answers)
        final_answer = self._elevate_final_answer_if_needed(
            prompt=prompt,
            prior=prior,
            model_answers=model_answers,
            started_at=started_at,
            draft=final_answer,
        )

        final_answer, grounding = self._ground_final_answer(
            prompt=prompt,
            draft=final_answer,
            packet=packet,
            started_at=started_at,
        )
        if packet.labels:
            grounding = GroundingReport(
                available=grounding.available,
                findings=grounding.findings,
                authorities=packet.labels,
                note=grounding.note,
            )

        verifier_specs = [
            ("gemma3_verifier", self._settings.multi_chat_verifier_gemma3_model),
            ("saul_verifier", self._settings.multi_chat_verifier_saul_model),
        ]
        verifiers: list[VerifierResult] = []
        for name, model in verifier_specs:
            remaining = self._remaining_seconds(started_at)
            if remaining <= 0:
                verdict = (
                    "RISK: medium\n"
                    "CONCERN: Verifier skipped because model jury time budget was exceeded.\n"
                    "SUGGESTION: Re-run with a shorter prompt if verifier output is needed."
                )
                verifiers.append(VerifierResult(model=model, verdict=verdict))
                continue
            timeout = self._stage_timeout_seconds(
                remaining, self._settings.multi_chat_per_verifier_timeout_seconds
            )
            verifier = self._client(name, model, timeout_seconds=timeout)
            try:
                verdict = verifier.chat(
                    [
                        ChatMessage(
                            "system",
                            "You are a legal-analysis verifier. Evaluate the candidate answer for "
                            "accuracy risk and legal overstatement. Return 3 short lines exactly: "
                            "RISK: <low|medium|high>; CONCERN: <one sentence>; "
                            "SUGGESTION: <one sentence>.",
                        ),
                        ChatMessage(
                            "user",
                            f"Question:\n{prompt}\n\nCandidate answer:\n{final_answer}",
                        ),
                    ],
                    DecodingPolicy.COLD.config,
                ).strip()
            except Exception as exc:  # noqa: BLE001
                if self._is_timeout_exception(exc):
                    verdict = (
                        "RISK: medium\n"
                        "CONCERN: Verifier timed out under latency constraints.\n"
                        "SUGGESTION: Use panel answer cautiously and re-run verifier pass when resources permit."
                    )
                else:
                    verdict = (
                        "RISK: medium\n"
                        f"CONCERN: Verifier unavailable ({type(exc).__name__}).\n"
                        "SUGGESTION: Manually review before relying on this answer."
                    )
            if not verdict:
                verdict = "RISK: medium\nCONCERN: Verifier returned no content.\nSUGGESTION: Manually review before relying on this answer."
            verifiers.append(VerifierResult(model=model, verdict=verdict))

        return MultiChatResult(
            final_answer=final_answer,
            model_answers=model_answers,
            verifiers=verifiers,
            grounding=grounding,
        )
