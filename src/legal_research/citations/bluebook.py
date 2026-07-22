"""Citation formatting (spec §2.8, §6).

Bluebook is the default; the style is pluggable (ALWD / OSCOLA subclasses). A
stateful :class:`FootnoteBuilder` produces correct short forms — ``id.`` for an
immediately-repeated authority, a case short form or ``supra`` for a later repeat,
and full cites on first appearance — plus a table of authorities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import SourceType
from .corpus import CorpusRecord


def _case_short_name(title: str) -> str:
    first_party = title.split(" v. ")[0]
    first_party = first_party.split(",")[0].strip()
    words = first_party.split()
    if len(words) > 2:
        return " ".join(words[:2])
    return first_party


class CitationFormatter:
    """Bluebook formatter."""

    style = "bluebook"

    def full(self, record: CorpusRecord, pin: str | None = None) -> str:
        if record.type is SourceType.CASE:
            core = f"{record.title}, {record.volume} {record.reporter} {record.page}"
            if pin:
                core += f", {pin}"
            year = f" ({record.year})" if record.year else ""
            return f"{core}{year}."
        if record.type in (SourceType.STATUTE, SourceType.REGULATION):
            year = f" ({record.year})" if record.year else ""
            return f"{record.code} § {record.section}{year}."
        # secondary
        year = f" ({record.year})" if record.year else ""
        author = f"{record.code}, " if record.code else ""
        return f"{author}{record.title}{year}."

    def short(self, record: CorpusRecord, pin: str | None = None) -> str:
        if record.type is SourceType.CASE:
            at = f" at {pin}" if pin else f" at {record.page}"
            return f"{_case_short_name(record.title)}, {record.volume} {record.reporter}{at}."
        if record.type in (SourceType.STATUTE, SourceType.REGULATION):
            return f"{record.code} § {record.section}."
        return f"{record.title}."

    def supra(self, record: CorpusRecord, footnote: int, pin: str | None = None) -> str:
        label = record.code or _case_short_name(record.title) or record.title
        at = f", at {pin}" if pin else ""
        return f"{label}, supra note {footnote}{at}."

    def id_(self, pin: str | None = None) -> str:
        return f"Id. at {pin}." if pin else "Id."

    def toa_entry(self, record: CorpusRecord) -> str:
        return self.full(record).rstrip(".")


class AlwdFormatter(CitationFormatter):
    style = "alwd"
    # ALWD is largely identical for these record types; kept as a distinct hook.


class OscolaFormatter(CitationFormatter):
    style = "oscola"

    def full(self, record: CorpusRecord, pin: str | None = None) -> str:
        if record.type is SourceType.CASE:
            # OSCOLA: no comma before the year and neutral-ish ordering.
            core = f"{record.title} {record.volume} {record.reporter} {record.page}"
            year = f" ({record.year})" if record.year else ""
            return f"{core}{year}"
        return super().full(record, pin).rstrip(".")


_FORMATTERS: dict[str, type[CitationFormatter]] = {
    "bluebook": CitationFormatter,
    "alwd": AlwdFormatter,
    "oscola": OscolaFormatter,
}


def build_formatter(style: str = "bluebook") -> CitationFormatter:
    cls = _FORMATTERS.get(style.lower(), CitationFormatter)
    return cls()


@dataclass
class FootnoteBuilder:
    """Emits footnote text with correct short-form / id. / supra handling."""

    formatter: CitationFormatter
    _first_footnote: dict[str, int] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)
    _last_record: str | None = None
    _count: int = 0

    def cite(self, record: CorpusRecord, pin: str | None = None) -> tuple[int, str]:
        self._count += 1
        n = self._count
        rid = record.id

        if self._last_record == rid:
            text = self.formatter.id_(pin)
        elif rid in self._seen:
            if record.type is SourceType.CASE:
                text = self.formatter.short(record, pin)
            else:
                text = self.formatter.supra(record, self._first_footnote[rid], pin)
        else:
            text = self.formatter.full(record, pin)
            self._first_footnote[rid] = n

        self._seen.add(rid)
        self._last_record = rid
        return n, text


def table_of_authorities(
    records: list[CorpusRecord], formatter: CitationFormatter
) -> dict[str, list[str]]:
    """Group authorities by category and format each once, sorted alphabetically."""

    buckets: dict[str, list[str]] = {"Cases": [], "Statutes": [], "Regulations": [], "Other Authorities": []}
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            continue
        seen.add(record.id)
        entry = formatter.toa_entry(record)
        if record.type is SourceType.CASE:
            buckets["Cases"].append(entry)
        elif record.type is SourceType.STATUTE:
            buckets["Statutes"].append(entry)
        elif record.type is SourceType.REGULATION:
            buckets["Regulations"].append(entry)
        else:
            buckets["Other Authorities"].append(entry)
    return {k: sorted(v) for k, v in buckets.items() if v}
