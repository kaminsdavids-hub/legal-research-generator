"""Editor / Reviser (spec §2.7).

Line/substantive editing: tightens prose, enforces the anti-"AI voice" lint, applies
the scholar's revision requests on a selection, and tracks changes as EditRecords.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.base import ChatMessage, DecodingPolicy
from ..llm.pool import HERMES, WRITER
from ..voice.anti_ai_voice import lint_ai_voice, rewrite_for_voice
from .base import Agent, AgentContext, AgentResult

_CITE_TOKEN = re.compile(r"\{\{cite:[^}]+\}\}")

_SYSTEM = (
    "TASK: edit\n"
    "You are a meticulous legal editor. Apply the requested revision to the selection "
    "while preserving citations and meaning. Return only the revised text."
)

_GRAMMAR_SYSTEM = (
    "TASK: edit\n"
    "You are a legal copy editor. Correct grammar, punctuation, and clarity while "
    "preserving legal meaning. Do not add citations or legal claims. Return only the "
    "revised paragraph."
)

_SOCRATIC_SYSTEM = (
    "TASK: interview\n"
    "You are a Socratic legal-writing coach. Diagnose substantive weaknesses in the "
    "paragraph and respond with concise probing questions plus one concrete doctrinal "
    "or structural improvement direction. Avoid generic filler and repeated phrasing."
)

_SOCRATIC_APPLY_SYSTEM = (
    "TASK: edit\n"
    "You are a legal editor. Rewrite the paragraph to implement the requested "
    "substantive improvements while preserving legal precision and argumentative "
    "clarity. Avoid bland summary sentences and repetitive transitions. Return only the "
    "revised paragraph."
)

_SOCRATIC_CYCLE_BREAK_SYSTEM = (
    "TASK: interview\n"
    "You are a senior legal-writing coach. The discussion became repetitive or "
    "intellectually stale. Break the cycle by introducing fresh analytic leverage. "
    "Do not repeat earlier phrasing. Deliver concise but high-density coaching with: "
    "(1) one new doctrinal tension, (2) one concrete argumentative move, and "
    "(3) one hard counterquestion."
)

_SOCRATIC_APPLY_CYCLE_BREAK_SYSTEM = (
    "TASK: edit\n"
    "You are a senior legal editor fixing stale analysis. Rewrite with materially new "
    "intellectual content (doctrinal mechanism, counterargument handling, or policy "
    "tradeoff), while preserving legal precision. Avoid boilerplate transitions and "
    "recycled wording. Return only the revised paragraph."
)

_SOCRATIC_MODE_GUIDANCE: dict[str, str] = {
    "strengthen_doctrine": (
        "Interrogate doctrinal fit: identify the controlling test/elements, expose weak links in "
        "the current reasoning, and ask for one stronger doctrinal move."
    ),
    "expand_analysis": (
        "Push for deeper analysis: ask for missing analytical steps, richer explanation of legal "
        "mechanics, and a fuller chain of reasoning."
    ),
    "counter_rebuttal": (
        "Stress-test argument quality: force articulation of the strongest counterargument and then "
        "the narrowest persuasive rebuttal."
    ),
    "policy_implications": (
        "Develop consequences: probe institutional, practical, and policy implications that sharpen "
        "the paragraph's stakes."
    ),
    "comparative_framework": (
        "Broaden perspective: compare alternative frameworks, jurisdictions, or tests and explain why "
        "one framing is superior for this argument."
    ),
}

_WORD_RE = re.compile(r"[a-z0-9']+")
_CYCLE_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "your",
    "have",
    "been",
    "they",
    "their",
    "into",
    "about",
    "where",
    "while",
    "what",
    "which",
    "there",
    "would",
    "should",
    "could",
    "legal",
    "paragraph",
}

_INTELLECTUAL_SIGNAL_TERMS = (
    "doctrine",
    "doctrinal",
    "element",
    "test",
    "standard",
    "counterargument",
    "rebuttal",
    "precedent",
    "framework",
    "policy",
    "jurisdiction",
    "implication",
)

_SOCRATIC_APPLY_WORD_RANGES: dict[str, tuple[int, int]] = {
    "strengthen_doctrine": (180, 280),
    "expand_analysis": (260, 420),
    "counter_rebuttal": (240, 380),
    "policy_implications": (220, 340),
    "comparative_framework": (220, 360),
}


def _normalize_socratic_mode(mode: str) -> str:
    mode_key = (mode or "").strip().lower()
    if mode_key in _SOCRATIC_MODE_GUIDANCE:
        return mode_key
    return "strengthen_doctrine"


def _history_turns(history: list[Any]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for turn in history:
        role = str(getattr(turn, "role", "") if not isinstance(turn, dict) else turn.get("role", ""))
        content = str(
            getattr(turn, "content", "") if not isinstance(turn, dict) else turn.get("content", "")
        ).strip()
        if role in {"user", "assistant"} and content:
            turns.append((role, content))
    return turns


def _meaningful_tokens(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    filtered = [w for w in words if len(w) > 3 and w not in _CYCLE_STOPWORDS]
    return filtered or words


def _token_overlap(a: str, b: str) -> float:
    left = set(_meaningful_tokens(a))
    right = set(_meaningful_tokens(b))
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _needs_socratic_cycle_break(
    message: str,
    assistant: str,
    history_turns: list[tuple[str, str]],
) -> bool:
    if not assistant.strip():
        return True

    recent_user = [content for role, content in history_turns if role == "user"][-6:]
    repeated_prompt_count = sum(
        1 for prior in recent_user if _token_overlap(prior, message) >= 0.9
    )
    repeated_prompt = repeated_prompt_count >= 2

    recent_assistant = [content for role, content in history_turns if role == "assistant"][-3:]
    repeated_assistant = any(_token_overlap(prior, assistant) >= 0.88 for prior in recent_assistant)

    low_substance = len(_meaningful_tokens(assistant)) < 16
    return repeated_prompt or repeated_assistant or low_substance


def _needs_rewrite_cycle_break(original: str, revised: str) -> bool:
    revised_text = revised.strip()
    if not revised_text:
        return True
    too_similar = _token_overlap(original, revised_text) >= 0.9
    low_substance = len(_meaningful_tokens(revised_text)) < 20
    lacks_signals = not any(term in revised_text.lower() for term in _INTELLECTUAL_SIGNAL_TERMS)
    return too_similar or low_substance or lacks_signals


def _assistant_novelty_score(assistant: str, history_turns: list[tuple[str, str]]) -> float:
    text = assistant.strip()
    if not text:
        return 0.0
    prior_assistant = [content for role, content in history_turns if role == "assistant"][-4:]
    if not prior_assistant:
        return 1.0
    max_overlap = max((_token_overlap(prev, text) for prev in prior_assistant), default=0.0)
    return round(max(0.0, 1.0 - max_overlap), 3)


def _strip_tokens(text: str) -> str:
    return _CITE_TOKEN.sub("", text).strip()


def _collect_tokens(text: str) -> list[str]:
    return _CITE_TOKEN.findall(text)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if p.strip()).strip()


def _split_with_tokens(text: str) -> tuple[list[str], list[list[str]]]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    prose: list[str] = []
    tokens_by_paragraph: list[list[str]] = []
    for paragraph in paragraphs:
        tokens_by_paragraph.append(_CITE_TOKEN.findall(paragraph))
        prose.append(_CITE_TOKEN.sub("", paragraph).strip())
    return prose, tokens_by_paragraph


def _rebuild_with_tokens(paragraphs: list[str], tokens_by_paragraph: list[list[str]]) -> str:
    rebuilt: list[str] = []
    for idx, tokens in enumerate(tokens_by_paragraph):
        prose = paragraphs[idx].strip() if idx < len(paragraphs) else ""
        merged = (prose + "".join(tokens)).strip() if prose else "".join(tokens)
        if merged:
            rebuilt.append(merged)
    for extra in paragraphs[len(tokens_by_paragraph) :]:
        extra = extra.strip()
        if extra:
            rebuilt.append(extra)
    return _join_paragraphs(rebuilt)


def _normalize_paragraph_count(paragraphs: list[str], target: int) -> list[str]:
    if target <= 0:
        return []
    if len(paragraphs) == target:
        return paragraphs
    if len(paragraphs) < target:
        return [*paragraphs, *("" for _ in range(target - len(paragraphs)))]
    head = paragraphs[: target - 1]
    tail = " ".join(p.strip() for p in paragraphs[target - 1 :] if p.strip()).strip()
    return [*head, tail]


class EditorAgent(Agent):
    name = "Editor / Reviser"
    expert_role = WRITER

    def act(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        if kwargs.get("socratic"):
            return self._socratic_paragraph(ctx, **kwargs)

        section_id: str | None = kwargs.get("section_id")
        instruction: str | None = kwargs.get("instruction")
        bb = ctx.blackboard

        if section_id and instruction:
            return self._revise_selection(ctx, section_id, instruction)

        # Default pass: enforce anti-AI-voice across all drafted sections.
        rewrites = 0
        grammar_sections = 0
        grammar_roles_used: set[str] = set()
        fail_open = bool(ctx.settings.grammar_chain_fail_open)
        for section in bb.outline:
            if not section.content:
                continue

            current = section.content
            if ctx.settings.grammar_chain_enabled:
                try:
                    current, roles = self._apply_grammar_chain(ctx, section.id, current)
                    if roles:
                        grammar_sections += 1
                        grammar_roles_used.update(roles)
                except Exception:
                    if not fail_open:
                        raise

            prose = _strip_tokens(current)
            report = lint_ai_voice(prose)
            if report.passed:
                section.content = current
                continue
            revised = rewrite_for_voice(prose)
            new_content = self._replace_prose_preserving_tokens(current, revised)
            bb.add_edit(
                section_id=section.id,
                before=current,
                after=new_content,
                note=f"anti-AI-voice rewrite: {[f.kind for f in report.failures()]}",
                author=self.name,
            )
            section.content = new_content
            rewrites += 1

        return AgentResult(
            agent=self.name,
            summary=(
                "voice pass complete; "
                f"grammar-chain touched {grammar_sections} sections "
                f"({','.join(sorted(grammar_roles_used)) or 'none'}); "
                f"rewrote {rewrites} sections"
            ),
            payload={
                "rewrites": rewrites,
                "grammar_sections": grammar_sections,
                "grammar_roles": sorted(grammar_roles_used),
            },
        )

    def _apply_grammar_chain(
        self, ctx: AgentContext, section_id: str, content: str
    ) -> tuple[str, list[str]]:
        roles = [r.strip() for r in ctx.settings.grammar_chain_roles.split(",") if r.strip()]
        if not roles:
            return content, []

        current = content
        original = content
        applied: list[str] = []
        fail_open = bool(ctx.settings.grammar_chain_fail_open)

        for role in roles:
            before = current
            try:
                current = self._run_grammar_pass(ctx, role, current)
            except Exception:
                if not fail_open:
                    raise
                current = before
                continue
            if current != before:
                applied.append(role)

        if current != original:
            ctx.blackboard.add_edit(
                section_id=section_id,
                before=original,
                after=current,
                note=f"grammar chain pass: {','.join(applied) or 'no-op'}",
                author=self.name,
            )
        return current, applied

    def _run_grammar_pass(self, ctx: AgentContext, role: str, content: str) -> str:
        original_tokens = _collect_tokens(content)
        paragraphs, tokens_by_paragraph = _split_with_tokens(content)
        if not paragraphs:
            return content

        client = ctx.pool.get(role)
        revised_paragraphs: list[str] = []
        for paragraph in paragraphs:
            if not paragraph:
                revised_paragraphs.append(paragraph)
                continue
            revised = client.chat(
                [
                    ChatMessage("system", _GRAMMAR_SYSTEM),
                    ChatMessage("user", paragraph),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()
            revised_paragraphs.append(revised or paragraph)

        candidate = _rebuild_with_tokens(revised_paragraphs, tokens_by_paragraph)
        if _collect_tokens(candidate) != original_tokens:
            raise ValueError(f"grammar pass {role} changed citation token set")
        return candidate

    def _replace_prose_preserving_tokens(self, original_with_tokens: str, revised_prose: str) -> str:
        _, tokens_by_paragraph = _split_with_tokens(original_with_tokens)
        revised_paragraphs = _split_paragraphs(revised_prose)
        revised_paragraphs = _normalize_paragraph_count(revised_paragraphs, len(tokens_by_paragraph))
        out = _rebuild_with_tokens(revised_paragraphs, tokens_by_paragraph)
        if _collect_tokens(out) != _collect_tokens(original_with_tokens):
            raise ValueError("citation token mismatch while applying rewrite")
        return out

    def _revise_selection(
        self, ctx: AgentContext, section_id: str, instruction: str
    ) -> AgentResult:
        bb = ctx.blackboard
        section = bb.get_section(section_id)
        client = ctx.pool.get(self.expert_role)
        revised = client.chat(
            [
                ChatMessage("system", _SYSTEM),
                ChatMessage("user", f"Instruction: {instruction}\n\nSelection:\n{_strip_tokens(section.content)}"),
            ],
            DecodingPolicy.BALANCED.config,
        ).strip()
        new_content = self._replace_prose_preserving_tokens(section.content, revised)
        record = bb.add_edit(
            section_id=section_id,
            before=section.content,
            after=new_content,
            note=instruction,
            author=self.name,
        )
        section.content = new_content
        return AgentResult(
            agent=self.name,
            summary=f"revised {section_id} per instruction",
            payload={"edit_id": record.id},
        )

    def _socratic_paragraph(self, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        section_id = str(kwargs.get("section_id") or "")
        paragraph_index = int(kwargs.get("paragraph_index") or 0)
        message = str(kwargs.get("message") or "").strip()
        apply_revision = bool(kwargs.get("apply_revision"))
        mode = _normalize_socratic_mode(str(kwargs.get("mode") or "strengthen_doctrine"))
        mode_guidance = _SOCRATIC_MODE_GUIDANCE[mode]
        history = kwargs.get("history") or []
        turns = _history_turns(history)
        if not section_id:
            raise KeyError("missing section_id")
        if not message:
            message = "Help me strengthen this paragraph's substantive legal analysis."

        bb = ctx.blackboard
        section = bb.get_section(section_id)
        clean = _strip_tokens(section.content)
        paragraphs = _split_paragraphs(clean)
        if not paragraphs:
            paragraphs = [clean or section.title]

        idx = max(0, min(paragraph_index, len(paragraphs) - 1))
        paragraph = paragraphs[idx]

        mentor = ctx.pool.get(HERMES)
        messages = [
            ChatMessage("system", _SOCRATIC_SYSTEM),
            ChatMessage(
                "user",
                (
                    f"Socratic mode: {mode}\n"
                    f"Mode objective: {mode_guidance}\n\n"
                    f"Section: {section.title}\n"
                    f"Paragraph needing substantive revision:\n{paragraph}"
                ),
            ),
        ]
        for role, content in turns[-8:]:
            if role in {"user", "assistant"} and content:
                messages.append(ChatMessage(role, content))
        messages.append(ChatMessage("user", message))
        assistant = mentor.chat(messages, DecodingPolicy.BALANCED.config).strip()
        cycle_break_triggered = False

        if _needs_socratic_cycle_break(message, assistant, turns):
            cycle_break_triggered = True
            cycle_break_messages = [
                ChatMessage("system", _SOCRATIC_CYCLE_BREAK_SYSTEM),
                ChatMessage(
                    "user",
                    (
                        f"Socratic mode: {mode}\n"
                        f"Mode objective: {mode_guidance}\n\n"
                        f"Section: {section.title}\n"
                        f"Paragraph needing substantive revision:\n{paragraph}"
                    ),
                ),
            ]
            for role, content in turns[-8:]:
                cycle_break_messages.append(ChatMessage(role, content))
            cycle_break_messages.append(ChatMessage("assistant", assistant))
            cycle_break_messages.append(
                ChatMessage(
                    "user",
                    (
                        "Cycle-break request: provide a fresh, high-intellectual coaching response "
                        "that avoids repeated phrasing and introduces genuinely new analysis."
                    ),
                )
            )
            improved = mentor.chat(cycle_break_messages, DecodingPolicy.BALANCED.config).strip()
            if improved:
                assistant = improved

        novelty_score = _assistant_novelty_score(assistant, turns)

        suggested_revision = ""
        rewrite_delta_score = 0.0
        if apply_revision:
            writer = ctx.pool.get(self.expert_role)
            min_words, max_words = _SOCRATIC_APPLY_WORD_RANGES[mode]
            suggested_revision = writer.chat(
                [
                    ChatMessage("system", _SOCRATIC_APPLY_SYSTEM),
                    ChatMessage(
                        "user",
                        (
                            f"Socratic mode: {mode}\n"
                            f"Mode objective: {mode_guidance}\n"
                            f"Target revised paragraph length: about {min_words}-{max_words} words.\n\n"
                            f"Section: {section.title}\n"
                            f"Original paragraph:\n{paragraph}\n\n"
                            f"Coaching guidance:\n{assistant}\n\n"
                            f"User request:\n{message}"
                        ),
                    ),
                ],
                DecodingPolicy.BALANCED.config,
            ).strip()

            if _needs_rewrite_cycle_break(paragraph, suggested_revision):
                improved_revision = writer.chat(
                    [
                        ChatMessage("system", _SOCRATIC_APPLY_CYCLE_BREAK_SYSTEM),
                        ChatMessage(
                            "user",
                            (
                                f"Socratic mode: {mode}\n"
                                f"Mode objective: {mode_guidance}\n"
                                f"Target revised paragraph length: about {min_words}-{max_words} words.\n\n"
                                f"Section: {section.title}\n"
                                f"Original paragraph:\n{paragraph}\n\n"
                                f"Previous stale rewrite attempt:\n{suggested_revision}\n\n"
                                f"Coaching guidance:\n{assistant}\n\n"
                                f"User request:\n{message}"
                            ),
                        ),
                    ],
                    DecodingPolicy.BALANCED.config,
                ).strip()
                if improved_revision:
                    suggested_revision = improved_revision

            rewrite_delta_score = round(
                max(0.0, 1.0 - _token_overlap(paragraph, suggested_revision or paragraph)),
                3,
            )

            paragraphs[idx] = suggested_revision or paragraph
            new_content = self._replace_prose_preserving_tokens(
                section.content,
                _join_paragraphs(paragraphs),
            )
            bb.add_edit(
                section_id=section_id,
                before=section.content,
                after=new_content,
                note=f"socratic paragraph {idx + 1}: {message}",
                author=self.name,
            )
            section.content = new_content

        return AgentResult(
            agent=self.name,
            summary=f"socratic {mode} coaching for {section_id} p{idx + 1}",
            payload={
                "section_id": section_id,
                "paragraph_index": idx,
                "assistant": assistant,
                "suggested_revision": suggested_revision,
                "applied": apply_revision,
                "mode": mode,
                "cycle_break_triggered": cycle_break_triggered,
                "novelty_score": novelty_score,
                "rewrite_delta_score": rewrite_delta_score,
            },
        )
