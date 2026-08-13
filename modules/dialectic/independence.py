"""Independence guard: reject an antithesis that merely negates the thesis.

Telling the antithesis to contradict the thesis directly fixed one problem and
created another. Before, the two sides argued the same position and no crux
existed. After, the antithesis reliably contradicted — but often by taking a
thesis proposition and inserting "not":

    thesis:     "The Supreme Court has recognized that modern cell phones ..."
    antithesis: "The Supreme Court has not recognized that modern cell phones ..."

That is a genuine contradiction, and the NLI pass is right to flag it, but it is
a much weaker adversarial signal than a distinct opposing theory. A mirror
concedes the thesis's framing, its predicate, and its choice of authority, and
disputes only the polarity. Nothing is learned from it that the thesis did not
already assert.

This guard makes independence *enforced* rather than merely requested in a
prompt, in the same spirit as :mod:`modules.dialectic.channel`. Unlike the
citation channel it is a quality property, not a safety one, so it degrades
visibly instead of failing closed: see ``DialecticChat._generate_position``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import textnorm


@dataclass(frozen=True)
class Mirror:
    """One candidate proposition that merely negates an opposing one."""

    candidate: str
    opposing: str
    jaccard: float
    containment: float

    def describe(self) -> str:
        return (
            f"{self.candidate!r} merely negates {self.opposing!r} "
            f"(jaccard {self.jaccard:.2f}, containment {self.containment:.2f})"
        )


class MirrorDetected(Exception):
    """Raised when a position restates the opposing side with the polarity flipped."""

    def __init__(self, mirrors: list[Mirror]) -> None:
        self.mirrors = mirrors
        super().__init__(
            f"{len(mirrors)} proposition(s) merely negate the opposing side: "
            + "; ".join(m.describe() for m in mirrors)
        )


class IndependenceGuard:
    """Detect polarity-flip restatements of an opposing position.

    A candidate is a mirror when its polarity differs from the opposing
    proposition *and* the two carry essentially the same vocabulary. Both
    conditions are required: shared vocabulary alone is expected (the two sides
    are arguing about the same thing), and a polarity difference alone is what a
    real disagreement looks like. It is the combination — same words, flipped
    sign — that identifies a restatement rather than an argument.

    Words are stemmed before comparison so "applies"/"apply" do not read as
    different vocabulary, and negation markers are dropped from the content set
    so the inserted "not" does not itself lower the overlap.
    """

    # Calibrated against the mirrors and independent counter-theories observed in
    # the live run (REMEDIATION 10.3). The two populations separate cleanly:
    #
    #   real mirrors            jaccard 0.55 - 1.00,  containment 0.81 - 1.00
    #   independent theories    jaccard 0.00 - 0.12,  containment 0.00 - 0.25
    #
    # The thresholds sit in the gap, not at the edge of either population, so a
    # somewhat wordier mirror or a somewhat more on-topic counter-theory does
    # not flip the verdict.

    #: Symmetric overlap above which two propositions share their vocabulary.
    MIN_JACCARD = 0.45

    #: Share of the shorter proposition contained in the longer. Catches a
    #: mirror that also truncates, which Jaccard alone scores too low.
    MIN_CONTAINMENT = 0.65

    #: Below this many content words a proposition is too short to judge; a
    #: terse claim like "No warrant is required" legitimately shares almost all
    #: its vocabulary with its opposite.
    MIN_CONTENT_WORDS = 5

    def measure(self, candidate: str, opposing: str) -> Mirror | None:
        """Return a :class:`Mirror` when *candidate* merely negates *opposing*."""
        if textnorm.has_negation(candidate) == textnorm.has_negation(opposing):
            return None

        cand_words = textnorm.content_words(candidate, stemmed=True)
        opp_words = textnorm.content_words(opposing, stemmed=True)
        if (
            len(cand_words) < self.MIN_CONTENT_WORDS
            or len(opp_words) < self.MIN_CONTENT_WORDS
        ):
            return None

        j = textnorm.jaccard(cand_words, opp_words)
        shorter, longer = (
            (cand_words, opp_words)
            if len(cand_words) <= len(opp_words)
            else (opp_words, cand_words)
        )
        c = textnorm.containment(shorter, longer)

        if j >= self.MIN_JACCARD or c >= self.MIN_CONTAINMENT:
            return Mirror(candidate=candidate, opposing=opposing, jaccard=j, containment=c)
        return None

    def find_mirrors(self, candidates: list[str], opposing: list[str]) -> list[Mirror]:
        """Every candidate that mirrors some opposing proposition, best match first."""
        found: list[Mirror] = []
        for candidate in candidates:
            best: Mirror | None = None
            for other in opposing:
                mirror = self.measure(candidate, other)
                if mirror is None:
                    continue
                if best is None or mirror.jaccard > best.jaccard:
                    best = mirror
            if best is not None:
                found.append(best)
        return found

    def scan(self, candidates: list[str], opposing: list[str]) -> None:
        """Raise :class:`MirrorDetected` if any candidate merely negates the other side."""
        if not opposing:
            return
        mirrors = self.find_mirrors(candidates, opposing)
        if mirrors:
            raise MirrorDetected(mirrors)
