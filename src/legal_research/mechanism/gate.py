"""The gate: a deterministic lexical layer, an optional LLM critic, one repair pass.

Precedence is the whole point, and it is enforced here in code rather than in a
prompt:

    the lexical layer may add or escalate a finding; the semantic layer may
    never soften one.

A model asked "is this specific enough?" will, often enough to matter, say yes
about prose it or a sibling model wrote. Two things make that impossible here.
The critic has **no channel for approval** — it returns findings or an empty
list, and an empty list is not evidence of anything, so there is nothing for it
to clear a lexical finding with. And :func:`combine` keeps the lexical severity
as a floor per paragraph, so a critic that reports the same passage as merely
under-specified cannot demote a BLACK_BOX.

The repair pass is subject to the same discipline in the other direction: a
rewrite is accepted only if a **re-scan** shows it strictly improved. A model
that returns confident prose which still names no operation leaves the finding
standing, unresolved and reported, rather than being trusted because it tried.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, Protocol

from ..models import MechanismFinding
from .blackbox import (
    _STRUCTURE_MARKERS,
    Finding,
    Severity,
    _sentence_bounds,
    scan_by_paragraph,
    scan_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MechanismCritic",
    "combine",
    "lexical_findings",
    "paragraph_severity",
    "repair_paragraph",
    "to_records",
]

_SEVERITY_ORDER: dict[str, int] = {
    "": 0,
    Severity.UNDER_SPECIFIED.value: 1,
    Severity.BLACK_BOX.value: 2,
}

_JSON_BLOCK = re.compile(r"\[.*\]|\{.*\}", re.DOTALL)

#: Least of the original a repair may keep. Set from the observed failure: the
#: rewrite that gutted a section kept 29%, and a genuine "name the operation"
#: edit is roughly length-preserving, since it replaces a label with a clause.
RETENTION_FLOOR = 0.6

#: Structure the rewrite names that the original did not. Any at all is a
#: refusal: an architecture, an objective, a threshold or a feature set that
#: exists only in the repair came from the model's weights, and material may not
#: enter a manuscript that way — the same rule the CitationGuard enforces for
#: authorities.
MAX_FABRICATED_MARKERS = 0

# No example phrases. An earlier version of this prompt illustrated the rule
# with "the algorithm determines" / "denied by an AI system" / "the model is
# biased", and a 4B critic returned those three strings as its findings on a
# manuscript that contained none of them — it was completing the prompt, not
# reading the text. Illustrations are a liability in a task whose whole output
# is supposed to be quotations from the input.
CRITIC_SYSTEM = (
    "TASK: verify\n"
    "You are a specificity examiner for legal scholarship. For each paragraph, "
    "find every assertion that rests on an automated system whose OPERATION the "
    "paragraph never states: the passage treats the system as the decider, the "
    "instrument, or the cause of a legal consequence without naming the inputs "
    "it reads, the operation performed on them, or the output relied on.\n"
    "Discussing AI as a SUBJECT of the paper is not a defect. Only mechanism "
    "position is.\n"
    "A statute, regulation, rule, court, agency or person is NOT an automated "
    "system. That a legal test is stated without its criteria may be a weakness, "
    "but it is not this defect, and reporting it here is an error.\n"
    '"term" MUST be copied character-for-character from the paragraph you are '
    "reporting on. A term that is not in the paragraph is discarded, and the "
    "finding with it.\n"
    'Respond ONLY with JSON: {"findings": [{"paragraph_index": 0, '
    '"severity": "BLACK_BOX"|"UNDER_SPECIFIED", "term": "<quoted from the '
    'paragraph>", "reason": "..."}]}\n'
    "Emit no prose and no other key. An empty findings list is a valid answer."
)

REPAIR_SYSTEM = (
    "TASK: edit\n"
    "You are a legal editor. The paragraph asserts something about an automated "
    "system without saying what the system does.\n"
    "You may do exactly two things. (1) If the paragraph states the operation "
    "somewhere else, move it into the sentence that relies on it. (2) If it does "
    "not, CONFINE THE CLAIM: say plainly that the passage does not state how the "
    "system operates, and narrow the assertion to what it does support.\n"
    "You may NOT supply the operation. Do not name an architecture, a training "
    "set, a threshold, a feature, a score or a statistical method that is not "
    "already in the paragraph, and do not add citations or facts. A rewrite that "
    "invents the mechanism is rejected automatically and wastes the pass.\n"
    "Keep the paragraph's length and legal meaning. Return only the revised "
    "paragraph."
)


class ChatClient(Protocol):
    """Structural match for :class:`legal_research.llm.base.LLMClient`.

    ``messages`` is typed as ``Any`` rather than ``list[ChatMessage]`` so a
    scripted stub is a legal client here: the gate's tests must be able to run
    the precedence and repair rules with no model anywhere.
    """

    def chat(self, messages: Any, config: Any = None) -> str: ...


# --------------------------------------------------------------------------- #
# Deterministic layer
# --------------------------------------------------------------------------- #
def lexical_findings(
    section_id: str, section_title: str, prose: str
) -> list[MechanismFinding]:
    """Scan a section's prose, one record per finding, paragraph-indexed."""

    records: list[MechanismFinding] = []
    for report in scan_by_paragraph(prose, where=section_title or section_id):
        records.extend(
            to_records(section_id, section_title, report.index, report.findings, "lexical")
        )
    return records


def to_records(
    section_id: str,
    section_title: str,
    paragraph_index: int,
    findings: Iterable[Finding],
    source: str,
) -> list[MechanismFinding]:
    return [
        MechanismFinding(
            section_id=section_id,
            section_title=section_title,
            paragraph_index=paragraph_index,
            severity=finding.severity.value,
            term=finding.term,
            reason=finding.reason,
            excerpt=finding.excerpt,
            source=source,
        )
        for finding in findings
    ]


def paragraph_severity(findings: Iterable[MechanismFinding]) -> str:
    """Worst severity among ``findings``; ``""`` when there are none."""

    return max((f.severity for f in findings), key=lambda s: _SEVERITY_ORDER.get(s, 0), default="")


# --------------------------------------------------------------------------- #
# Semantic layer
# --------------------------------------------------------------------------- #
class MechanismCritic:
    """An LLM prompted as a specificity examiner, findings only.

    It sees a whole section so it can catch what the regexes cannot — an
    assertion that rests on an unstated operation while using none of the
    vocabulary, which is the failure mode a lexical layer is structurally blind
    to. It cannot clear anything: see the module docstring.
    """

    def __init__(self, client: ChatClient, *, config: Any = None) -> None:
        self._client = client
        self._config = config

    def findings(
        self, section_id: str, section_title: str, paragraphs: Sequence[str]
    ) -> list[MechanismFinding]:
        if not paragraphs:
            return []

        from ..llm.base import ChatMessage, DecodingPolicy

        payload = json.dumps(
            {
                "section": section_title or section_id,
                "paragraphs": [
                    {"paragraph_index": i, "text": text} for i, text in enumerate(paragraphs)
                ],
            },
            indent=2,
        )
        cold = DecodingPolicy.COLD.config
        config = self._config or replace(cold, response_format="json_object")
        messages = [ChatMessage("system", CRITIC_SYSTEM), ChatMessage("user", payload)]

        raw = self._client.chat(messages, config)
        try:
            return _parse_findings(raw, section_id, section_title, paragraphs)
        except ValueError as first:
            # One retry, with the parser's own complaint quoted back. A model
            # that emitted an unescaped quotation mark inside a quoted passage
            # can usually fix that when told which character broke it; a model
            # that cannot is about to fail the same way twice, cheaply.
            logger.warning(
                "mechanism critic: unparseable reply for %s (%s); retrying once "
                "with the parse error quoted back.",
                section_title or section_id,
                first,
            )
            retry = self._client.chat(
                [
                    *messages,
                    ChatMessage("assistant", raw[:2000]),
                    ChatMessage(
                        "user",
                        f"That reply could not be parsed as JSON: {first}. Return the "
                        "same findings as one valid JSON object. Escape every "
                        "quotation mark inside a quoted term, and use no newlines "
                        "inside strings.",
                    ),
                ],
                config,
            )
            return _parse_findings(retry, section_id, section_title, paragraphs)


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _json_findings(raw: str) -> list[Any]:
    """The findings list from a reply, salvaging one if the whole will not parse.

    Whole-document parsing first. Failing that, each top-level ``{...}`` object
    is parsed on its own and the broken ones are skipped — a reply that went bad
    on the third of six findings should cost three findings, not six.

    Salvage is *logged and counted*, never silent, and it cannot turn an
    entirely unparseable reply into a clean empty list: if nothing at all parses,
    this raises, and the caller reports the critic as unavailable rather than as
    having found nothing. That distinction is the one this repository has paid
    for three times (REMEDIATION §11.6, §14, §21).
    """

    match = _JSON_BLOCK.search(raw or "")
    if not match:
        raise ValueError(f"mechanism critic returned no JSON: {(raw or '')[:200]!r}")

    block = match.group(0)
    try:
        payload: Any = json.loads(block)
    except json.JSONDecodeError as exc:
        salvaged = _salvage_objects(block)
        if not salvaged:
            raise ValueError(f"mechanism critic returned unparseable JSON: {exc}") from exc
        logger.warning(
            "mechanism critic: reply was malformed (%s); salvaged %d finding "
            "object(s) from it. Findings after the break are lost.",
            exc,
            len(salvaged),
        )
        return salvaged

    if isinstance(payload, dict):
        payload = payload.get("findings", [])
    return list(payload or [])


def _salvage_objects(block: str) -> list[Any]:
    """Parse each brace-balanced object independently, skipping the broken ones."""

    objects: list[Any] = []
    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(block):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            # Every closing brace closes *some* object, and the findings are
            # nested inside the envelope: scanning only depth-0 objects would
            # try the envelope, fail on the one broken finding inside it, and
            # salvage nothing — which is what a first version of this did.
            start = starts.pop()
            with contextlib.suppress(json.JSONDecodeError):
                candidate = json.loads(block[start : index + 1])
                if isinstance(candidate, dict) and "term" in candidate:
                    objects.append(candidate)
    return objects


def _parse_findings(
    raw: str, section_id: str, section_title: str, paragraphs: Sequence[str]
) -> list[MechanismFinding]:
    """Parse the critic's JSON, tolerating prose around it.

    A malformed response must not read as "no findings" — that would let a
    broken critic wave a draft through, which is the silent-degradation shape
    this repository has now paid for three times (REMEDIATION §11.6, §14, §21).
    Unparseable output raises; the caller decides whether an unusable critic is
    fatal or is dropped with a warning, and either way says which happened.

    **A finding whose term is not in the paragraph is dropped.** On a real
    2,600-word manuscript a 4B critic reported terms lifted from its own prompt,
    and reasoning about landlord-tenant law in a paper on export control. Every
    one of those was refutable without a model: the quoted string was not in the
    text. Asking the critic to quote and then checking the quote is the cheapest
    verification in the package, and it is deterministic, so a weak critic
    degrades to *silence* rather than to fiction.
    """

    payload = _json_findings(raw)
    haystacks = [_normalise(p) for p in paragraphs]
    records: list[MechanismFinding] = []
    unquoted = 0
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", Severity.BLACK_BOX.value)).upper()
        if severity not in _SEVERITY_ORDER or severity == "":
            severity = Severity.BLACK_BOX.value
        index = int(item.get("paragraph_index", 0) or 0)
        # An out-of-range index would silently attach a finding to the wrong
        # paragraph, so it is clamped before the quote is checked.
        index = max(0, min(index, max(0, len(paragraphs) - 1)))
        term = str(item.get("term", "")).strip()
        needle = _normalise(term)
        if not needle or not any(needle in hay for hay in haystacks):
            unquoted += 1
            continue
        if needle not in haystacks[index]:
            # Quoted from the section, but attributed to the wrong paragraph.
            # Re-point it rather than dropping it: the quote is real.
            index = next(i for i, hay in enumerate(haystacks) if needle in hay)
        records.append(
            MechanismFinding(
                section_id=section_id,
                section_title=section_title,
                paragraph_index=index,
                severity=severity,
                term=term,
                reason=str(item.get("reason", "")).strip()
                or "the critic reported an unstated operation",
                excerpt=_locate(paragraphs[index], term),
                source="semantic",
            )
        )
    if unquoted:
        logger.warning(
            "mechanism critic: dropped %d finding(s) in %s whose quoted term is "
            "not in the paragraph. A critic that cannot quote the text it is "
            "reading is describing something else.",
            unquoted,
            section_title or section_id,
        )
    return records


def _locate(paragraph: str, term: str) -> str:
    """The quoted term in context, so the reader sees the sentence, not the label."""

    words = " ".join(paragraph.split())
    lowered = words.casefold()
    start = lowered.find(_normalise(term))
    if start < 0:  # pragma: no cover - the caller has already checked
        return ""
    left = max(0, start - 70)
    right = min(len(words), start + len(term) + 70)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(words) else ""
    return f"{prefix}{words[left:right]}{suffix}"


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #
def combine(
    lexical: Sequence[MechanismFinding], semantic: Sequence[MechanismFinding]
) -> list[MechanismFinding]:
    """Merge the layers. Lexical findings survive verbatim; semantic ones add.

    Where both layers speak about the same paragraph, the lexical severity is a
    floor: a semantic finding that would report the paragraph as less serious is
    escalated to the lexical severity rather than being taken at its word. A
    semantic finding on a paragraph the lexical layer passed is kept as-is —
    that is the case the critic exists for.
    """

    floor: dict[tuple[str, int], str] = {}
    for finding in lexical:
        key = (finding.section_id, finding.paragraph_index)
        if _SEVERITY_ORDER.get(finding.severity, 0) > _SEVERITY_ORDER.get(floor.get(key, ""), 0):
            floor[key] = finding.severity

    merged: list[MechanismFinding] = list(lexical)
    for finding in semantic:
        key = (finding.section_id, finding.paragraph_index)
        raised = floor.get(key, "")
        if _SEVERITY_ORDER.get(raised, 0) > _SEVERITY_ORDER.get(finding.severity, 0):
            finding = finding.model_copy(update={"severity": raised})
        merged.append(finding)
    return sorted(
        merged,
        key=lambda f: (
            f.section_id,
            f.paragraph_index,
            -_SEVERITY_ORDER.get(f.severity, 0),
            f.source,
        ),
    )


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #
def repair_paragraph(
    client: ChatClient,
    paragraph: str,
    findings: Sequence[MechanismFinding],
    *,
    config: Any = None,
) -> tuple[str, bool]:
    """One bounded rewrite attempt. Returns ``(text, improved)``.

    ``improved`` is decided by re-scanning the rewrite, never by asking the
    model whether it succeeded. A rewrite is accepted only when it lowers the
    paragraph's worst severity or reduces the finding count without introducing
    a worse one; otherwise the original paragraph is returned unchanged and its
    findings stay on the report. Silently keeping a rewrite that fixed nothing
    would turn the gate into a stage that reliably produces churn and reports
    success.

    **Fewer findings is not enough on its own**, and a real run is why. Asked to
    repair a 450-word section, an 8B writer returned 130 words that had dropped
    most of the argument, and another that supplied the missing mechanism by
    inventing it — "the algorithms read inputs that include case facts,
    applicable statutes, precedent" — from a paragraph that said no such thing.
    Both cleared a scan-only test, because deleting the passage and fabricating
    its mechanism each remove the finding. So two more conditions apply, and
    both are deterministic:

    * **Retention.** A rewrite shorter than :data:`RETENTION_FLOOR` of the
      original is refused. Cutting an argument is a decision for its author.
    * **Novelty.** A rewrite whose content words are largely absent from the
      original is refused, because a mechanism that appears only in the repair
      was supplied by the model, and this package's entire premise is that
      material may not enter that way.

    A refused rewrite is not a failure of the gate. The finding survives, is
    reported, and goes to the person who knows what the system actually did.
    """

    from ..llm.base import ChatMessage, DecodingPolicy

    before = scan_text(paragraph)
    if not before:
        return paragraph, False

    span = _repair_span(paragraph, before)
    original = paragraph[span[0] : span[1]]
    complaint = "\n".join(f"- {f.term!r}: {f.reason}" for f in findings) or "\n".join(
        f"- {f.term!r}: {f.reason}" for f in before
    )
    revised_span = str(
        client.chat(
            [
                ChatMessage("system", REPAIR_SYSTEM),
                ChatMessage(
                    "user",
                    f"Findings to resolve:\n{complaint}\n\n"
                    f"Surrounding paragraph, for context only — do NOT rewrite it:\n"
                    f"{paragraph}\n\n"
                    f"Rewrite ONLY this passage, and return only the rewritten "
                    f"passage:\n{original}",
                ),
            ],
            config or DecodingPolicy.COLD.config,
        )
    ).strip()
    if not revised_span:
        return paragraph, False
    revised = f"{paragraph[: span[0]]}{revised_span} {paragraph[span[1] :]}".strip()
    revised = re.sub(r"[ \t]{2,}", " ", revised)

    after = scan_text(revised)
    if _worst(after) > _worst(before):
        return paragraph, False
    if not (_worst(after) < _worst(before) or len(after) < len(before)):
        return paragraph, False

    retained = len(revised_span.split()) / max(1, len(original.split()))
    if retained < RETENTION_FLOOR:
        logger.warning(
            "mechanism repair refused: the rewrite kept %.0f%% of the passage. "
            "Removing the finding by removing the argument is the author's call, "
            "not the gate's.",
            retained * 100,
        )
        return paragraph, False

    # Retention is measured against the span, which is what was rewritten;
    # fabrication against the whole paragraph, which is what the model was shown.
    # Moving an operation stated two sentences away is the repair working, not a
    # fabrication, and comparing to the span alone would refuse it.
    fabricated = _fabricated_structure(paragraph, revised_span)
    if len(fabricated) > MAX_FABRICATED_MARKERS:
        logger.warning(
            "mechanism repair refused: the rewrite names structure the passage "
            "never did (%s). The operation was supplied by the model, which is "
            "the one way of satisfying this gate that must not work.",
            ", ".join(sorted(fabricated)[:5]),
        )
        return paragraph, False
    return revised, True


def _repair_span(paragraph: str, findings: Sequence[Finding]) -> tuple[int, int]:
    """Character bounds of the passage a repair is allowed to touch.

    Exactly the sentences the findings sit in — not the paragraph, and not a
    neighbourhood around them. The surrounding prose is supplied to the model as
    context and is spliced back untouched, because a sentence inside the
    rewritable span is a sentence the model may delete: an earlier version
    included one sentence on either side "for context" and the rewrite quietly
    swallowed the paragraph's conclusion.

    On a real manuscript the voice pass had flattened each
    section into a single 400-word block, so "rewrite the paragraph" meant
    "rewrite the section", and an 8B model handed the whole argument back
    restructured, 62% shorter, with a fabricated mechanism in the middle of it.

    That flattening is fixed at its source in
    :func:`legal_research.voice.anti_ai_voice.rewrite_for_voice`, but the
    granularity must not depend on upstream prose being well-formed: an ingested
    brief, a pasted draft, or any long paragraph would reintroduce it. A bounded
    span also makes the retention and fabrication checks meaningful, since both
    are measured against the passage actually rewritten.
    """

    bounds = _sentence_bounds(paragraph)
    hit = {
        index
        for index, (start, end) in enumerate(bounds)
        for finding in findings
        if start <= finding.span[0] < end
    }
    if not hit:
        return 0, len(paragraph)
    return bounds[min(hit)][0], bounds[max(hit)][1]


def _worst(findings: Sequence[Finding]) -> int:
    return max((_SEVERITY_ORDER[f.severity.value] for f in findings), default=0)


def _fabricated_structure(original: str, revised: str) -> set[str]:
    """Structure markers present in the rewrite and absent from the original.

    This reuses the detector's own vocabulary of concrete structure — named
    estimators, objectives, training data, thresholds, feature sets — which is
    exactly the material a repair would have to invent in order to clear the
    gate dishonestly. Comparing marker sets rather than word overlap is what
    lets an honest repair through: confining a claim, or moving an operation the
    passage already stated into the sentence that relies on it, introduces no
    marker that was not already there.

    It is a floor, not a proof of good faith. A fabrication phrased without any
    marker still passes here, and the verifier and the citation guard remain the
    checks that a source says what the text claims.
    """

    def markers(text: str) -> set[str]:
        found: set[str] = set()
        for marker in _STRUCTURE_MARKERS:
            found.update(_normalise(m.group(0)) for m in marker.finditer(text))
        return found

    return markers(revised) - markers(original)
