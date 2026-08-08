"""CitationChannel: hard guard against model-emitted citation strings."""

from __future__ import annotations

import re


class CitationDetected(Exception):
    """Raised when a model emits a string that looks like a legal citation."""

    def __init__(self, text: str, hits: list[str]) -> None:
        self.text = text
        self.hits = hits
        super().__init__(f"citation string(s) detected: {hits}")


# Reporters that indicate a formal citation.  Longer, more specific reporters are
# listed before their prefixes so "F. Supp. 2d" is not truncated to "F.".
#
# Hand-maintained, and deliberately so: `src/legal_research/citations/bluebook.py`
# formats citations from a record's `reporter` field but holds no reporter table
# to generate this from, so there is nothing to read here (checked 2026-08-07).
# `N.E.2d` and `N.E.3d` each used to appear twice; duplicates are harmless in an
# alternation but they are also the first sign the list has drifted.
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
    r"N\.\s?E\.\s?3d",
    r"N\.\s?E\.\s?2d",
    r"N\.\s?W\.\s?2d",
    r"S\.\s?E\.\s?2d",
    r"S\.\s?W\.\s?3d",
    r"S\.\s?W\.\s?2d",
    r"P\.\s?3d",
    r"P\.\s?2d",
    r"B\.\s?R\.",
    r"Cal\.\s?App\.\s?4th",
    r"Cal\.\s?App\.\s?3d",
    r"N\.\s?Y\.\s?2d",
    r"N\.\s?Y\.\s?3d",
    r"So\.\s?2d",
    r"So\.\s?3d",
)

# Reporter cite: volume + reporter + page, with tolerant whitespace so a cite
# split across a newline is still caught after whitespace normalization.
_REPORTER_RE = re.compile(
    r"\b(?P<volume>\d{1,4})\s+(?P<reporter>" + "|".join(_REPORTERS) + r")\s+(?P<page>\d{1,5})\b",
    re.IGNORECASE,
)

# Party name: capitalised token, possibly containing common punctuation, with a
# small number of follow-on capitalised words.  "of" is allowed inside a party
# name; "the" is not, because it lets the match run into narrative text.
_PARTY = r"[A-Z][A-Za-z&.'\-]+(?:\s+(?:of\s+)?[A-Z][A-Za-z&.'\-]+){0,4}"

# "X v. Y" / "X vs. Y" / "X v Y" case-name pattern.  Party names must start with
# an uppercase letter; the pattern is compiled with IGNORECASE only for the
# connector so common abbreviations like "V." still match.
_CASE_NAME_RE = re.compile(
    r"(?<![A-Za-z])(?P<first>" + _PARTY + r")\s+(?i:v)[.s]?\s+(?P<second>" + _PARTY + r")",
)


class CitationChannel:
    """Scanner that rejects any model turn containing a citation string."""

    def __init__(self) -> None:
        self.reporter_re = _REPORTER_RE
        self.case_name_re = _CASE_NAME_RE

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse runs of whitespace so split-cite patterns are not broken."""
        return " ".join(text.split())

    def find_hits(self, text: str) -> list[str]:
        """Return every citation-looking substring found in *text*."""
        normalized = self._normalize_whitespace(text)
        hits: list[str] = []
        for match in self.reporter_re.finditer(normalized):
            hits.append(match.group(0))
        for match in self.case_name_re.finditer(normalized):
            hit = match.group(0)
            # Avoid double-reporting the same span when a reporter cite also
            # contains a case name.
            if hit not in hits:
                hits.append(hit)
        return hits

    def scan(self, text: str, role: str = "assistant") -> None:
        """Raise :class:`CitationDetected` if *text* contains any citation pattern.

        User-provided text is exempt because the invariant targets generation
        models, not the human.  Every other role (assistant/system) is scanned.
        """
        if role == "user":
            return
        hits = self.find_hits(text)
        if hits:
            raise CitationDetected(text, hits)
