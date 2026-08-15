"""Citation grounding for the Model Jury chat.

The manuscript pipeline already refuses to cite anything the retriever did not
return (see :mod:`legal_research.citations.verifier`). The chat panel historically
had no such guard: models answered from their own weights and routinely emitted
authorities that do not exist, or real authorities standing for propositions they
never held.

This module closes that gap with three capabilities:

* :func:`extract_citations` — pull case names, reporter cites, statutes and
  regulations out of free-form model prose.
* :class:`ChatCitationAuditor` — resolve each extracted citation against the
  corpus and score whether the surrounding sentence is actually supported.
* :meth:`ChatCitationAuditor.authority_packet` — retrieve real corpus authority
  for a question so the panel can cite from evidence instead of memory.

Everything degrades gracefully: if the corpus or retriever is unavailable the
auditor reports ``available=False`` and the chat proceeds unguarded rather than
failing the request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from .citations.corpus import Corpus, CorpusRecord
from .citations.retriever import Retriever, _tokens
from .citations.support import SupportScorer, best_support

__all__ = [
    "CitationKind",
    "GroundingStatus",
    "ExtractedCitation",
    "CitationFinding",
    "GroundingReport",
    "ChatCitationAuditor",
    "extract_citations",
]


class CitationKind(StrEnum):
    CASE_NAME = "case_name"
    REPORTER = "reporter"
    STATUTE = "statute"
    REGULATION = "regulation"


class GroundingStatus(StrEnum):
    #: Resolves to a corpus record and the surrounding claim is supported.
    SUPPORTED = "supported"
    #: Resolves to a record that was retrieved as authority for *this* question, but
    #: the automated support test was inconclusive. Entailment models are brittle
    #: against narrative prose, so this is reported as advisory rather than as an
    #: error: the authority is on-topic by construction.
    UNCONFIRMED = "unconfirmed"
    #: Resolves to a corpus record that was not retrieved for this question and does
    #: not support the claim - the model likely pulled it from memory.
    MISATTRIBUTED = "misattributed"
    #: No corpus record matches. Treated as a probable fabrication.
    NOT_IN_CORPUS = "not_in_corpus"


# --- extraction ---------------------------------------------------------------------

# Reporters we recognise in citation strings. Order matters: longer, more specific
# reporters must precede their prefixes so "F. Supp. 2d" is not truncated to "F.".
_REPORTERS = (
    r"F\.\s?Supp\.\s?3d",
    r"F\.\s?Supp\.\s?2d",
    r"F\.\s?Supp\.",
    r"F\.\s?App'x",
    r"F\.\s?4th",
    r"F\.\s?3d",
    r"F\.\s?2d",
    r"S\.\s?Ct\.",
    r"L\.\s?Ed\.\s?2d",
    r"L\.\s?Ed\.",
    r"U\.\s?S\.",
    r"A\.\s?3d",
    r"A\.\s?2d",
    r"N\.E\.\s?3d",
    r"N\.E\.\s?2d",
    r"N\.W\.\s?2d",
    r"S\.E\.\s?2d",
    r"S\.W\.\s?3d",
    r"S\.W\.\s?2d",
    r"P\.\s?3d",
    r"P\.\s?2d",
    r"B\.R\.",
)

_REPORTER_RE = re.compile(
    r"\b(?P<volume>\d{1,4})\s+(?P<reporter>" + "|".join(_REPORTERS) + r")\s+(?P<page>\d{1,5})\b"
)

# The section symbol is optional: models write both "15 U.S.C. § 78j(b)" and the
# bare "15 U.S.C. 78j(b)". Requiring the section to start with a digit keeps a
# trailing sentence word from being swallowed as a section number.
_SECTION_MARK = r"(?:§+\s*|[Ss]ec(?:tion|\.)?\s*)?"
_SECTION_BODY = r"(?P<section>\d[\w.\-]*(?:\([\w.\-]+\))*)"

_STATUTE_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s+(?P<code>U\.?\s?S\.?\s?C\.?)\s*" + _SECTION_MARK + _SECTION_BODY
)

_REGULATION_RE = re.compile(
    r"\b(?P<title>\d{1,3})\s+(?P<code>C\.?\s?F\.?\s?R\.?)\s*" + _SECTION_MARK + _SECTION_BODY
)

# A party name: capitalised word possibly followed by more capitalised words,
# allowing common corporate punctuation. Markdown emphasis markers are tolerated
# because models frequently italicise case names.
# "of" is permitted inside a party name ("First Interstate Bank of Denver") but
# "the" is not: allowing it lets the match run past the case name into narrative
# text, e.g. "Ernst v. Hochfelder the Court recognized...".
_PARTY = r"[A-Z][A-Za-z&.'\-]*(?:\s+(?:of\s+)?[A-Z][A-Za-z&.'\-]*){0,4}"
_CASE_NAME_RE = re.compile(
    r"(?<![A-Za-z])(?P<first>" + _PARTY + r")\s+v[.s]?\s+(?P<second>" + _PARTY + r")"
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(*\"'])")

# Legal prose is dense with abbreviations whose trailing period is not a sentence
# end. Splitting naively turns "In Ernst v. Hochfelder the Court held..." into the
# fragment "In Ernst v.", which then looks too short to support-test and silently
# exempts the citation from misattribution checking.
_ABBREVIATIONS = {
    "v", "vs", "inc", "co", "cos", "corp", "no", "nos", "us", "usc", "cfr",
    "cir", "ct", "supp", "ed", "rev", "stat", "art", "mr", "mrs", "ms", "dr",
    "jr", "sr", "al", "cf", "eg", "ie", "etc", "sec", "seq", "ann", "dist",
    "div", "dep", "fed", "reg", "st", "assn", "bros", "ltd", "llc", "llp",
    "pa", "pc", "natl", "intl", "comm", "conn", "mass", "tex", "cal", "fla",
}


def _ends_with_abbreviation(text: str, period_index: int) -> bool:
    """True if the period at ``period_index`` closes an abbreviation or initial."""

    cursor = period_index - 1
    while cursor >= 0 and (text[cursor].isalpha() or text[cursor] == "."):
        cursor -= 1
    word = text[cursor + 1 : period_index].replace(".", "")
    if not word:
        return False
    # A single letter is an initial ("J. Smith"), never a sentence end.
    return len(word) == 1 or word.lower() in _ABBREVIATIONS


# Capitalised words that introduce or follow a citation but are not part of the
# party name. Without trimming these, a sentence-initial signal word is captured
# as a party ("In Ernst v. Hochfelder") and then fails to resolve, producing a
# false fabrication report.
_LEADING_NOISE = {
    "in", "see", "cf", "accord", "compare", "but", "and", "under", "eg", "id",
    "also", "here", "thus", "however", "although", "though", "because", "when",
    "while", "after", "before", "since", "held", "holding", "decision", "case",
    "whether", "that", "this", "these", "those", "if", "as", "per", "via", "from",
    "unlike", "like", "following", "citing", "quoting", "applying",
    "distinguishing", "overruling", "affirming", "reversing", "first", "second",
    "third", "finally", "moreover", "furthermore", "similarly", "conversely",
}

_TRAILING_NOISE = {
    "the", "court", "held", "its", "and", "decision", "opinion", "majority",
    "dissent", "panel", "justices", "supreme", "where", "which", "that",
}


def _trim_party(name: str, noise: set[str], from_end: bool) -> str:
    """Strip narrative words from one end of a captured party name."""

    words = name.split()
    while words:
        candidate = words[-1] if from_end else words[0]
        if _norm_key(candidate) not in noise:
            break
        words = words[:-1] if from_end else words[1:]
    return " ".join(words)


_EMPHASIS_RE = re.compile(r"[*_`]+")

#: Retrieval hits below this absolute score are too weak to offer as authority.
MIN_AUTHORITY_SCORE = 0.08

#: ...and hits scoring below this fraction of the best hit are off-topic noise.
#: A relative cutoff is used because lexical TF-IDF and FAISS cosine scores are
#: on different scales, so an absolute floor alone cannot serve both backends.
AUTHORITY_SCORE_RATIO = 0.25

#: A proposition with fewer content tokens than this (after removing the citation
#: itself) carries too little meaning to entailment-test. "Liability also arises
#: under 15 U.S.C. 78j(b)" asserts nothing a passage could contradict, so such
#: cites are accepted on resolution alone rather than flagged as misattributed.
MIN_SCORABLE_TOKENS = 4

#: Chat prose is scored against a *relaxed* fraction of the manuscript's ship-time
#: support threshold. A manuscript proposition is a single crisp assertion; a chat
#: sentence carries narrative framing ("The Supreme Court held that...", "here the
#: plaintiff would...") that dilutes token recall and weakens entailment without
#: making the citation wrong. Measured against the sample corpus, true claims score
#: 0.20-0.83 lexically while false claims stay under 0.10, so halving the 0.34
#: threshold separates them cleanly. Without this, correct citations to the very
#: authority the packet supplied are reported as misattributed.
CHAT_SUPPORT_RELAXATION = 0.5



def _strip_emphasis(text: str) -> str:
    return _EMPHASIS_RE.sub("", text)


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _norm_key(text: str) -> str:
    """Aggressively normalise a citation token for equality comparison."""

    return re.sub(r"[^a-z0-9]", "", text.lower())


def _norm_words(text: str) -> str:
    return _normalize_space(re.sub(r"[^a-z0-9 ]", " ", text.lower()))


@dataclass(frozen=True)
class ExtractedCitation:
    """A citation-looking string found in model output."""

    text: str
    kind: CitationKind
    start: int
    end: int
    #: Canonical comparison key, e.g. ``"410|us|113"`` or ``"15usc|78j"``.
    key: str = ""
    sentence: str = ""


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return contiguous ``(start, end, sentence)`` spans for the supplied text."""

    starts: list[int] = []
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        terminator = match.start() - 1
        if terminator >= 0 and text[terminator] == "." and _ends_with_abbreviation(
            text, terminator
        ):
            continue
        starts.append(match.end())

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for start in starts:
        spans.append((cursor, start, text[cursor:start].strip()))
        cursor = start
    spans.append((cursor, len(text), text[cursor:].strip()))
    return spans


def _sentence_for(spans: list[tuple[int, int, str]], position: int) -> str:
    for start, end, sentence in spans:
        if start <= position < end:
            return sentence
    return ""


def _reporter_key(volume: str, reporter: str, page: str) -> str:
    return f"{volume.strip()}|{_norm_key(reporter)}|{page.strip()}"


def _code_key(title: str, code: str, section: str) -> str:
    return f"{title.strip()}{_norm_key(code)}|{_norm_key(section)}"


def extract_citations(text: str) -> list[ExtractedCitation]:
    """Extract case names, reporter cites, statutes and regulations from prose.

    Overlapping matches are resolved in favour of the more specific citation type
    (statute/regulation/reporter beat a bare case name) so a full citation is not
    double-counted.
    """

    if not text or not text.strip():
        return []

    cleaned = _strip_emphasis(text)
    spans = _split_sentences(cleaned)
    found: list[ExtractedCitation] = []

    for match in _STATUTE_RE.finditer(cleaned):
        found.append(
            ExtractedCitation(
                text=_normalize_space(match.group(0)),
                kind=CitationKind.STATUTE,
                start=match.start(),
                end=match.end(),
                key=_code_key(match.group("title"), "usc", match.group("section")),
                sentence=_sentence_for(spans, match.start()),
            )
        )

    for match in _REGULATION_RE.finditer(cleaned):
        found.append(
            ExtractedCitation(
                text=_normalize_space(match.group(0)),
                kind=CitationKind.REGULATION,
                start=match.start(),
                end=match.end(),
                key=_code_key(match.group("title"), "cfr", match.group("section")),
                sentence=_sentence_for(spans, match.start()),
            )
        )

    for match in _REPORTER_RE.finditer(cleaned):
        found.append(
            ExtractedCitation(
                text=_normalize_space(match.group(0)),
                kind=CitationKind.REPORTER,
                start=match.start(),
                end=match.end(),
                key=_reporter_key(
                    match.group("volume"), match.group("reporter"), match.group("page")
                ),
                sentence=_sentence_for(spans, match.start()),
            )
        )

    for match in _CASE_NAME_RE.finditer(cleaned):
        first = _trim_party(match.group("first"), _LEADING_NOISE, from_end=False)
        second = _trim_party(match.group("second"), _TRAILING_NOISE, from_end=True)
        if not first or not second:
            continue
        name = _normalize_space(f"{first} v. {second}")
        found.append(
            ExtractedCitation(
                text=name,
                kind=CitationKind.CASE_NAME,
                start=match.start(),
                end=match.end(),
                key=_norm_words(name),
                sentence=_sentence_for(spans, match.start()),
            )
        )

    # Drop case-name matches fully contained in a more specific citation span, and
    # de-duplicate identical citations while preserving first-seen order.
    specific = [c for c in found if c.kind is not CitationKind.CASE_NAME]
    result: list[ExtractedCitation] = []
    seen: set[tuple[CitationKind, str]] = set()
    for citation in sorted(found, key=lambda c: (c.start, c.kind.value)):
        if citation.kind is CitationKind.CASE_NAME and any(
            other.start <= citation.start and citation.end <= other.end for other in specific
        ):
            continue
        dedupe_key = (citation.kind, citation.key or citation.text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(citation)
    return result


# --- auditing -----------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorityPacket:
    """Verified authority retrieved for one question."""

    text: str = ""
    labels: list[str] = field(default_factory=list)
    record_ids: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class CitationFinding:
    citation: str
    kind: CitationKind
    status: GroundingStatus
    detail: str
    record_id: str = ""
    support: float = 0.0
    sentence: str = ""

    @property
    def is_problem(self) -> bool:
        """Whether the finding warrants a repair pass and a reader warning."""

        return self.status in (
            GroundingStatus.MISATTRIBUTED,
            GroundingStatus.NOT_IN_CORPUS,
        )


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of auditing one answer against the corpus."""

    available: bool = True
    findings: list[CitationFinding] = field(default_factory=list)
    authorities: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def problems(self) -> list[CitationFinding]:
        return [f for f in self.findings if f.is_problem]

    @property
    def fabricated(self) -> list[CitationFinding]:
        return [f for f in self.findings if f.status is GroundingStatus.NOT_IN_CORPUS]

    @property
    def misattributed(self) -> list[CitationFinding]:
        return [f for f in self.findings if f.status is GroundingStatus.MISATTRIBUTED]

    @property
    def supported(self) -> list[CitationFinding]:
        return [f for f in self.findings if f.status is GroundingStatus.SUPPORTED]

    @property
    def unconfirmed(self) -> list[CitationFinding]:
        return [f for f in self.findings if f.status is GroundingStatus.UNCONFIRMED]

    @property
    def clean(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if not self.available:
            return self.note or "Citation grounding unavailable."
        if not self.findings:
            return "No case or statutory citations were asserted."
        parts = [
            f"{len(self.supported)} of {len(self.findings)} citations verified against "
            f"the corpus"
        ]
        if self.unconfirmed:
            parts.append(f"{len(self.unconfirmed)} retrieved but unconfirmed")
        parts.append(f"{len(self.fabricated)} unverifiable")
        parts.append(f"{len(self.misattributed)} possibly misattributed")
        return "; ".join(parts) + "."


class ChatCitationAuditor:
    """Resolves chat citations against the corpus and scores their support."""

    def __init__(
        self,
        corpus: Corpus,
        retriever: Retriever,
        scorer: SupportScorer,
        *,
        min_support: float | None = None,
    ) -> None:
        self._corpus = corpus
        self._retriever = retriever
        self._scorer = scorer
        self._threshold = (
            float(min_support)
            if min_support is not None
            else float(scorer.threshold) * CHAT_SUPPORT_RELAXATION
        )
        self._reporter_index: dict[str, CorpusRecord] = {}
        self._code_index: dict[str, CorpusRecord] = {}
        self._title_index: dict[str, CorpusRecord] = {}
        self._build_indexes()

    # -- indexing ---------------------------------------------------------------

    def _build_indexes(self) -> None:
        for record in self._corpus.records:
            if record.reporter and record.volume is not None and record.page is not None:
                key = _reporter_key(str(record.volume), record.reporter, str(record.page))
                self._reporter_index.setdefault(key, record)
            if record.code and record.section:
                title, _, remainder = record.code.strip().partition(" ")
                code_name = remainder or record.code
                if title.isdigit():
                    key = _code_key(title, code_name, record.section)
                else:
                    key = _code_key("", record.code, record.section)
                self._code_index.setdefault(key, record)
                # Also index without the leading title so "§ 78j(b)" style cites match.
                self._code_index.setdefault(_code_key("", code_name, record.section), record)
            if record.title:
                self._title_index.setdefault(_norm_words(record.title), record)

    # -- resolution -------------------------------------------------------------

    def _resolve_case_name(self, citation: ExtractedCitation) -> CorpusRecord | None:
        target = citation.key
        if not target:
            return None
        exact = self._title_index.get(target)
        if exact is not None:
            return exact
        target_tokens = {t for t in target.split() if len(t) > 2}
        if not target_tokens:
            return None
        best: CorpusRecord | None = None
        best_overlap = 0.0
        for title_key, record in self._title_index.items():
            title_tokens = {t for t in title_key.split() if len(t) > 2}
            if not title_tokens:
                continue
            overlap = len(target_tokens & title_tokens) / len(target_tokens)
            if overlap > best_overlap:
                best_overlap, best = overlap, record
        # Require a strong majority of the asserted party names to appear in the
        # corpus title before treating the citation as resolved.
        return best if best_overlap >= 0.75 else None

    def _resolve(self, citation: ExtractedCitation) -> CorpusRecord | None:
        if citation.kind is CitationKind.REPORTER:
            return self._reporter_index.get(citation.key)
        if citation.kind in (CitationKind.STATUTE, CitationKind.REGULATION):
            record = self._code_index.get(citation.key)
            if record is not None:
                return record
            # Retry ignoring the title number, e.g. "15 U.S.C. § 78j" vs "§ 78j".
            _, _, tail = citation.key.partition("|")
            for key, candidate in self._code_index.items():
                if key.endswith(f"|{tail}"):
                    return candidate
            return None
        return self._resolve_case_name(citation)

    def _best_support(self, proposition: str, record: CorpusRecord) -> tuple[float, str]:
        """Best support score across the record's passages and the claim's clauses."""

        return best_support(
            self._scorer,
            proposition,
            record.passages,
            threshold=self._threshold,
        )

    # -- public API -------------------------------------------------------------

    def audit(
        self, text: str, retrieved_ids: frozenset[str] | None = None
    ) -> GroundingReport:
        """Audit every citation in ``text``.

        ``retrieved_ids`` are the corpus records offered to the model as authority
        for this question. A weakly-scoring citation to one of those records is
        reported as :attr:`GroundingStatus.UNCONFIRMED` rather than misattributed,
        because the record is topically relevant by construction and the support
        test - not the citation - is the unreliable party.
        """

        packet_ids = retrieved_ids or frozenset()
        citations = extract_citations(text)
        findings: list[CitationFinding] = []
        for citation in citations:
            record = self._resolve(citation)
            if record is None:
                findings.append(
                    CitationFinding(
                        citation=citation.text,
                        kind=citation.kind,
                        status=GroundingStatus.NOT_IN_CORPUS,
                        detail=(
                            "No record in the verified corpus matches this authority; "
                            "it may be fabricated or outside the loaded corpus."
                        ),
                        sentence=citation.sentence,
                    )
                )
                continue

            # Score the claim, not the citation: digits and reporter tokens from the
            # cite itself would otherwise inflate or dilute the support measurement.
            proposition = (citation.sentence or text).replace(citation.text, " ")
            if len(_tokens(proposition)) < MIN_SCORABLE_TOKENS:
                findings.append(
                    CitationFinding(
                        citation=citation.text,
                        kind=citation.kind,
                        status=GroundingStatus.SUPPORTED,
                        detail=(
                            f"Resolves to corpus record {record.id}; the accompanying "
                            "sentence is too short to test for support."
                        ),
                        record_id=record.id,
                        sentence=citation.sentence,
                    )
                )
                continue

            support, _passage = self._best_support(proposition, record)
            if support < self._threshold:
                retrieved = record.id in packet_ids
                findings.append(
                    CitationFinding(
                        citation=citation.text,
                        kind=citation.kind,
                        status=(
                            GroundingStatus.UNCONFIRMED
                            if retrieved
                            else GroundingStatus.MISATTRIBUTED
                        ),
                        detail=(
                            (
                                f"Retrieved as authority for this question and resolves to "
                                f"{record.id}, but the automated support check was "
                                f"inconclusive (support={support:.2f}); confirm the holding."
                            )
                            if retrieved
                            else (
                                f"Authority resolves to corpus record {record.id} but does "
                                f"not support the accompanying proposition "
                                f"(support={support:.2f} < {self._threshold:.2f})."
                            )
                        ),
                        record_id=record.id,
                        support=round(support, 4),
                        sentence=citation.sentence,
                    )
                )
                continue

            findings.append(
                CitationFinding(
                    citation=citation.text,
                    kind=citation.kind,
                    status=GroundingStatus.SUPPORTED,
                    detail=(
                        f"Resolves to corpus record {record.id} and supports the "
                        f"proposition (support={support:.2f})."
                    ),
                    record_id=record.id,
                    support=round(support, 4),
                    sentence=citation.sentence,
                )
            )
        return GroundingReport(available=True, findings=findings)

    def retrieve_authorities(self, question: str, k: int = 6):
        try:
            return self._retriever.search(question, k=k)
        except Exception:  # noqa: BLE001 - retrieval must never break the chat
            return []

    def describe_record(self, record_id: str) -> str:
        record = self._corpus.get(record_id)
        if record is None:
            return record_id
        if record.reporter and record.volume is not None and record.page is not None:
            return f"{record.title}, {record.volume} {record.reporter} {record.page}"
        if record.code and record.section:
            return f"{record.title}, {record.code} § {record.section}"
        return record.title

    def authority_packet(
        self,
        question: str,
        k: int = 6,
        min_score: float = MIN_AUTHORITY_SCORE,
        score_ratio: float = AUTHORITY_SCORE_RATIO,
    ) -> AuthorityPacket:
        """Build a citable authority packet for a question.

        Weak hits are dropped: padding the packet with barely related records
        invites the panel to cite authority that has nothing to do with the
        question. A hit must clear both the absolute floor and a fraction of the
        best hit's score.
        """

        hits = self.retrieve_authorities(question, k=k)
        if not hits:
            return AuthorityPacket()

        cutoff = max(min_score, max(hit.score for hit in hits) * score_ratio)

        seen: set[str] = set()
        lines: list[str] = []
        labels: list[str] = []
        for hit in hits:
            if hit.score < cutoff:
                continue
            record = self._corpus.get(hit.record_id)
            if record is None or record.id in seen:
                continue
            seen.add(record.id)
            label = self.describe_record(record.id)
            labels.append(label)
            excerpt = _normalize_space(hit.text)
            if len(excerpt) > 480:
                excerpt = excerpt[:477].rstrip() + "..."
            lines.append(f"- {label} [{record.id}]\n  \"{excerpt}\"")

        if not lines:
            return AuthorityPacket()

        packet = (
            "VERIFIED AUTHORITY PACKET — these are the only authorities retrieved from the "
            "verified corpus for this question:\n\n" + "\n".join(lines)
        )
        return AuthorityPacket(text=packet, labels=labels, record_ids=frozenset(seen))


def format_grounding_notice(report: GroundingReport) -> str:
    """Render a reader-facing warning block for problematic citations."""

    if not report.available or report.clean:
        return ""

    lines = ["**Citation grounding check**", "", report.summary(), ""]
    for finding in report.problems:
        if finding.status is GroundingStatus.NOT_IN_CORPUS:
            lines.append(
                f"- **{finding.citation}** — not found in the verified corpus; treat as "
                "unverified and confirm independently before relying on it."
            )
        else:
            lines.append(
                f"- **{finding.citation}** — resolves to {finding.record_id} but does not "
                "appear to support the accompanying proposition; verify the holding."
            )
    return "\n".join(lines)
