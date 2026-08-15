"""Retrieval stage: turn a plain-English ``court_hint`` into candidate citations.

This is the missing link between generation and verification. The debaters emit
propositions and a plain-English *description* of the authority they need — by
design, since the citation channel forbids them from naming one. Verification,
in turn, only looks at slots that already carry a ``normalized_cite``. Nothing
bridged the two, so for a spec-compliant exchange the filled-slot list was always
empty, the network was never touched, and every slot stayed ``pending`` forever.

A :class:`CiteRetriever` proposes candidates. A candidate is *not* authority: it
moves the slot to :attr:`SlotStatus.PROPOSED`, and only a CourtListener lookup
can move it to ``verified``.

This module is transport-agnostic. The concrete adapter over the application's
corpus lives in :mod:`modules.dialectic.service`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CiteRetriever(Protocol):
    """Propose candidate citations for a proposition and its court hint."""

    def propose(self, court_hint: str, proposition: str) -> list[str]:
        """Return normalized citation strings, best first.

        Implementations must return an empty list rather than raise when they
        have nothing to offer: a retrieval miss is a normal outcome, not an
        error, and must not void the turn.
        """
        ...


@runtime_checkable
class CiteAnnotator(Protocol):
    """Optional capability: report that an authority is not currently operative.

    Separate from :class:`CiteRetriever` because most retrievers have no notion
    of good-law status, and a retriever that lacks it should still be usable.
    The engine checks for this method and skips annotation when it is absent.

    A rescinded rule that reaches the reader looking like operative law is a
    serious error, so the annotation is attached to the slot — which also puts
    it in front of the synthesis role, the only role that runs after retrieval
    and can therefore reason about what was actually retrieved.
    """

    def annotate(self, cite: str) -> str:
        """Return a status note for *cite*, or "" when it is in force."""
        ...


class StubCiteRetriever:
    """Deterministic offline retriever backed by an explicit mapping.

    Used by tests so the retrieval stage can be exercised with no corpus, no
    network, and no GPU. Keys are matched case-insensitively as substrings of
    the combined court hint and proposition, longest key first, so a more
    specific hint wins over a general one.
    """

    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self._mapping = dict(mapping or {})

    def propose(self, court_hint: str, proposition: str) -> list[str]:
        haystack = f"{court_hint} {proposition}".lower()
        for key in sorted(self._mapping, key=len, reverse=True):
            if key.lower() in haystack:
                return list(self._mapping[key])
        return []


class NullCiteRetriever:
    """A retriever that proposes nothing.

    Distinct from configuring no retriever at all: this one is present and
    simply found no candidate, which is a retrieval miss rather than a
    structurally unverifiable slot.
    """

    def propose(self, court_hint: str, proposition: str) -> list[str]:
        return []
