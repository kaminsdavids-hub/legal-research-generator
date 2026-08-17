"""Dialectic chat orchestrator: thesis / antithesis / synthesis + crux extraction."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .channel import CitationChannel, CitationDetected
from .copy import copy_crux_table
from .crux import CruxExtractor
from .independence import IndependenceGuard, Mirror, MirrorDetected
from .models import (
    NOT_OPERATIVE,
    BudgetLedger,
    CitationSlot,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)
from .nli import NLIEvaluator
from .retrieval import CiteRetriever
from .roles import detect_family
from .verification import CourtListenerClient, is_case_citation, verify_position


class FamilyCollision(Exception):
    """Raised when two debate roles share the same base model family."""


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat interface; matches the project's LLMClient protocol."""

    name: str

    def chat(self, messages: list[Any], config: Any | None = None) -> str: ...


# Signals in a slot's `court_hint` about which *kind* of authority it wants, used
# by `_select_candidate` when similarity ranking and the hint disagree.
#
# `_CASE_HINTS` wins whenever it matches, because a hint naming courts is
# unambiguous while a hint naming rules is not: "government regulation of speech"
# is a subject, "export control regulations" is an instrument, and only the
# second should pull a C.F.R. section. Bare "regulation"/"regulations" is
# therefore absent from `_INSTRUMENT_HINTS` on purpose — every entry there names
# a statutory instrument outright or is domain-specific enough not to describe a
# subject. Both lists are matched as substrings, so "statut" covers statute,
# statutes and statutory.
_CASE_HINTS = (
    "precedent",
    "court",
    "circuit",
    "case law",
    "opinion",
    "holding",
    "decision",
    " v. ",
)
_INSTRUMENT_HINTS = (
    "statut",
    "c.f.r",
    "u.s.c",
    "code of federal regulations",
    "federal register",
    "rulemaking",
    "export control regulation",
    "export administration regulation",
)

# Each side is told which way to argue. Without this the two debaters, generated
# independently from the same question, frequently argued the SAME side — and a
# dialectic in which both sides agree produces no cruxes by construction. Observed
# live: for "does the exclusionary rule apply to good-faith reliance", thesis and
# antithesis both returned "the exclusionary rule should not apply", and the NLI
# pass correctly reported entailment. See REMEDIATION 10.
_STANCE = {
    "thesis": (
        "Your stance is AFFIRMATIVE: argue that the answer to the question is "
        "YES. Every proposition you emit must support that answer."
    ),
    "antithesis": (
        "Your stance is NEGATIVE: argue that the answer to the question is NO. "
        "Every proposition you emit must support that answer. You must take the "
        "opposing side even if you find it weaker — do not agree with the "
        "thesis, do not restate its claims, and do not hedge toward it. Your job "
        "is to make the strongest case against."
    ),
}

_POSITION_PROMPT = (
    "You are the {side} in a legal dialectic.\n"
    "{stance}\n"
    "Respond ONLY with a JSON object: {{\"propositions\": ["
    "{{\"proposition\": \"...\", \"court_hint\": \"...\", \"weight\": \"...\"}}]}}.\n"
    "Emit 2 or 3 propositions. Each is a single outcome-bearing legal claim, "
    "stated as a complete sentence that can be directly affirmed or denied.\n"
    "The court_hint is a plain-English DESCRIPTION of the authority you need "
    "(e.g., \"Supreme Court stop-and-frisk precedent\" or \"circuit split on "
    "digital privacy\"). It must NOT name a case or contain a citation.\n"
    "weight must be one of: controlling, persuasive, supporting, contra. It "
    "describes the strength of the AUTHORITY you would cite, not your confidence "
    "and not your stance. Use \"contra\" only for authority that cuts against "
    "your own position; never use it to label your own argument.\n"
    "HARD RULE: the only keys read from each proposition object are "
    "\"proposition\", \"court_hint\", and \"weight\". No other key is read, and "
    "emitting one is treated as a violation. In particular do NOT emit a "
    "\"normalized_cite\" key: citations are resolved downstream by retrieval and "
    "verification, never supplied by you.\n"
    "HARD RULE: no field may contain a reporter citation, a case name in the form "
    "'X v. Y', 'Id.', 'supra', a statute section, or any string that looks like a "
    "legal citation. A response containing one is discarded entirely."
)

# Demanding "the negation of each thesis proposition" produced contradictions,
# but degenerate ones: the antithesis inserted "not" into the thesis's own
# sentence. A mirror concedes the thesis's framing, predicate and choice of
# authority and disputes only the sign, so nothing is learned from it. This asks
# instead for an independent theory that happens to be incompatible — and
# `IndependenceGuard` enforces it, because a prompt alone did not.
_REBUT_PROMPT = (
    "\n\nThe thesis has argued the following:\n"
    "{propositions}\n\n"
    "Build your OWN theory of the case that leads to the opposite conclusion. "
    "Your propositions must be incompatible with the thesis, but they must be "
    "independent arguments, NOT restatements.\n"
    "FORBIDDEN: producing a proposition by taking one of the thesis's sentences "
    "and inserting \"not\", \"no\", or \"cannot\". A proposition that reuses the "
    "thesis's wording with the polarity flipped is discarded and you will be "
    "asked again.\n"
    "REQUIRED: ground each of your propositions in a DIFFERENT doctrine, "
    "standard, test, or line of authority than the one the thesis relies on, and "
    "make that ground explicit in the proposition. Attack the thesis's framing "
    "-- its choice of rule, its analogy, the scope of its exception, the "
    "standard of review, a threshold question it skipped -- rather than its "
    "conclusion."
)

#: Phrases that count as saying an authority is no longer operative. Kept beside
#: the prompt that demands them so the two cannot drift apart.
_NON_OPERATIVE_ACK = (
    "rescinded",
    "superseded",
    "no longer operative",
    "no longer in force",
    "not currently operative",
    "not operative",
    "repealed",
    "withdrawn",
    "never took effect",
    "no longer good law",
)


def _acknowledges_non_operative(text: str) -> bool:
    """True when *text* states, in prose, that an authority is not in force."""
    lowered = text.lower()
    return any(marker in lowered for marker in _NON_OPERATIVE_ACK)


_STATUS_FEEDBACK = (
    "\n\nYour previous answer was REJECTED. It relies on {cites}, which is NO "
    "LONGER OPERATIVE LAW, without saying so. State plainly and in your own "
    "words that this authority has been rescinded and now carries only "
    "precedential weight, then give your synthesis."
)


_MIRROR_FEEDBACK = (
    "\n\nYour previous answer was REJECTED. These propositions merely negated "
    "the thesis instead of arguing independently:\n{mirrors}\n"
    "Do not restate the thesis with the polarity flipped. Argue from a different "
    "doctrine or a different framing of the question."
)


#: Told to the synthesis when the citation channel rejected its last answer.
#:
#: The synthesis loop's other rejection branch — the one for an unacknowledged
#: rescinded authority — already fed its reason back via _STATUS_FEEDBACK. This
#: one re-rolled on temperature alone, so the model saw an identical prompt and
#: made an identical mistake three times in a row. Observed on every run:
#: "citation string(s) rejected: ['Bernstein v. U.S. Dep']; after 3 attempts",
#: which spent the whole budget and returned no synthesis at all.
#:
#: The offending strings are quoted back because the rule is easy to satisfy and
#: hard to guess at: "do not cite" reads as a style note until you are shown
#: that naming the parties is what tripped it. The substitutions are spelled out
#: for the same reason — a model told only what it may not write tends to write
#: nothing, and an empty synthesis is not an improvement on a rejected one.
_CITATION_FEEDBACK = (
    "\n\nYour previous answer was REJECTED for naming authority. These strings "
    "tripped the citation channel:\n{hits}\n"
    "Refer to authority by what it decided, never by name or reporter: write "
    "\"the Ninth Circuit's encryption source-code decision\", \"the published-"
    "information exception\", or \"the controlling deemed-export rule\" instead "
    "of party names, 'X v. Y', or volume-reporter-page strings. Keep every "
    "other part of your answer — only the naming has to change."
)


#: The position's version of the same correction.
#:
#: Separate from `_CITATION_FEEDBACK` because the remedy differs: a position is
#: not being asked to write around the authority, it already has a field for
#: describing it. The scan covers `court_hint` as well as `proposition`, since a
#: model blocked from citing in one will route the case name through the other,
#: so the fix is to say the description belongs in court_hint and must stay a
#: description.
_POSITION_CITATION_FEEDBACK = (
    "\n\nYour previous answer was REJECTED. These strings look like citations "
    "and are not allowed in any field:\n{hits}\n"
    "Do not name cases, parties, reporters or statute sections anywhere. Put a "
    "plain-English DESCRIPTION of the authority you need in \"court_hint\" "
    "instead — \"circuit precedent on encryption source code\", \"the published-"
    "information exception in the export regulations\" — and keep the "
    "proposition itself free of names. Retrieval resolves the citation; you "
    "describe what you need."
)


#: Told to a position whose JSON could not be read.
#:
#: The parse branch re-rolled silently, so a model that misunderstood the shape
#: kept misunderstanding it for the whole budget. The error text is quoted back
#: because the failures are specific and fixable — a markdown fence, a missing
#: "propositions" key, an extra key the prompt forbids — and none of them are
#: guessable from being asked the same question again at a higher temperature.
_PARSE_FEEDBACK = (
    "\n\nYour previous answer was REJECTED because it could not be read: "
    "{error}.\n"
    "Respond with a single JSON object and nothing else — no prose before or "
    "after it, no markdown fences. The shape is "
    "{{\"propositions\": [{{\"proposition\": \"...\", \"court_hint\": \"...\", "
    "\"weight\": \"...\"}}]}}, and those three keys are the only ones read."
)

_SYNTHESIS_PROMPT = (
    "You are the synthesis in a legal dialectic.\n"
    "The user question, thesis, and antithesis are provided below.\n"
    "Concisely reconcile the strongest points, name the decisive crux, and state "
    "the likely outcome.\n"
    "Do NOT cite specific cases, reporter strings, or 'X v. Y' names.\n"
    "Use only plain prose.\n"
    "CRITICAL: some authorities carry a status note saying they are rescinded, "
    "superseded, or otherwise no longer operative. You are the only role that "
    "sees those notes. If an argument rests on such an authority, you MUST say "
    "in your own words that it is no longer operative law and that it carries "
    "only precedential or persuasive weight. Presenting a rescinded rule as "
    "current law is the single worst error you can make here."
)


def _system(side: str) -> str:
    return _POSITION_PROMPT.format(side=side, stance=_STANCE.get(side, ""))


def _try_parse_slots(raw: str) -> list[CitationSlot]:
    """Best-effort parse of the model's JSON into CitationSlot objects."""
    text = raw.strip()
    # Some models wrap JSON in markdown fences; strip them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    data = json.loads(text)
    if "propositions" not in data:
        raise ValueError("missing 'propositions' key")
    slots: list[CitationSlot] = []
    for item in data["propositions"]:
        # `normalized_cite` is an OUTPUT of verification, never an input from a
        # model. Reading it back off the model's JSON gave a third field (after
        # `proposition` and `court_hint`, REMEDIATION 8b) through which a live
        # reporter cite could reach the user unscanned. Treat any attempt to
        # supply one as a channel violation so the turn is discarded and
        # regenerated, exactly as a citation in `proposition` would be.
        supplied_cite = str(item.get("normalized_cite", "")).strip()
        if supplied_cite:
            raise CitationDetected(supplied_cite, [supplied_cite])

        # Case-normalize before matching. A model emitting "Controlling" used to
        # be silently stored as `supporting`, which dropped it out of crux
        # extraction with no log line anywhere.
        raw_weight = str(item.get("weight", "")).strip()
        note = ""
        try:
            weight = Weight(raw_weight.lower())
        except ValueError:
            weight = Weight.SUPPORTING
            if raw_weight:
                note = f"weight {raw_weight!r} not recognised; coerced to {weight.value!r}"
            else:
                note = f"no weight supplied; defaulted to {weight.value!r}"

        slots.append(
            CitationSlot(
                proposition=str(item["proposition"]).strip(),
                court_hint=str(item.get("court_hint", "")).strip(),
                weight=weight,
                note=note,
            )
        )
    if not slots:
        raise ValueError("no propositions found")
    return slots


class DialecticChat:
    """Drive a three-role dialectic with citation-channel discipline."""

    def __init__(
        self,
        thesis_client: ChatClient,
        antithesis_client: ChatClient,
        synthesis_client: ChatClient,
        nli_client: ChatClient | None = None,
        courtlistener: CourtListenerClient | None = None,
        retriever: CiteRetriever | None = None,
        channel: CitationChannel | None = None,
        independence: IndependenceGuard | None = None,
        max_regenerations: int = 3,
    ) -> None:
        self.thesis_client = thesis_client
        self.antithesis_client = antithesis_client
        self.synthesis_client = synthesis_client
        self.nli_client = nli_client
        self.courtlistener = courtlistener
        self.retriever = retriever
        self.channel = channel or CitationChannel()
        # Rejects an antithesis that restates the thesis with the polarity
        # flipped. Prompting alone did not stop it (REMEDIATION 10.3).
        self.independence = independence or IndependenceGuard()
        #: Set by `_generate_synthesis` when the synthesis had to be accepted
        #: despite failing to disclose a non-operative authority.
        self._pending_synthesis_note = ""
        # The NLI client is a fourth model, and it must actually reach the
        # evaluator. Constructing it and leaving `CruxExtractor()` to build a
        # client-less `NLIEvaluator` meant every relation came from the
        # heuristic while the configuration advertised a model.
        self.crux_extractor = CruxExtractor(nli=NLIEvaluator(client=nli_client))
        self.max_regenerations = max(1, max_regenerations)
        self._assert_family_distinct(
            thesis_client.name,
            antithesis_client.name,
            synthesis_client.name,
            nli_client.name if nli_client is not None else None,
        )

    @staticmethod
    def _assert_family_distinct(
        thesis_model: str,
        antithesis_model: str,
        synthesis_model: str,
        nli_model: str | None = None,
    ) -> None:
        """Every configured role must come from a distinct base model family.

        The NLI model is included: it adjudicates the debate, so sharing a
        family with a debater reintroduces exactly the correlated error the
        guard exists to prevent.
        """
        roles: list[tuple[str, str]] = [
            ("thesis", thesis_model),
            ("antithesis", antithesis_model),
            ("synthesis", synthesis_model),
        ]
        if nli_model is not None:
            roles.append(("nli", nli_model))

        collisions: list[str] = []
        for i, (role_a, model_a) in enumerate(roles):
            for role_b, model_b in roles[i + 1 :]:
                family = detect_family(model_a)
                if family == detect_family(model_b):
                    collisions.append(
                        f"{role_a} ({model_a}) and {role_b} ({model_b}) "
                        f"share family '{family}'"
                    )
        if collisions:
            raise FamilyCollision("; ".join(collisions))

    @staticmethod
    def _opposing_claims(opposing: Position | None) -> list[str]:
        """Usable propositions from the opposing side, if it produced any.

        A position that failed generation carries only the failure placeholder,
        which is neither an argument to rebut nor something to measure
        independence against.
        """
        if opposing is None:
            return []
        return [
            slot.proposition
            for slot in opposing.propositions
            if slot.status != SlotStatus.NOT_FOUND and slot.proposition
        ]

    @staticmethod
    def _rebuttal_block(opposing: Position | None) -> str:
        """Render the opposing side's propositions for a direct rebuttal.

        Empty when there is nothing usable to rebut — notably when the opposing
        position failed generation, whose placeholder proposition must never be
        fed back into another model as if it were an argument.
        """
        claims = DialecticChat._opposing_claims(opposing)
        if not claims:
            return ""
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        return _REBUT_PROMPT.format(propositions=numbered)

    def _generate_position(
        self,
        side: str,
        question: str,
        client: ChatClient,
        opposing: Position | None = None,
    ) -> tuple[Position, int]:
        """Generate one side, retrying on parse failure, citation leak, or mirroring.

        When *opposing* is supplied, its propositions are shown to this side to
        be argued against. Generating both sides blind from the same question let
        them argue the same position, which yields no cruxes by construction
        (REMEDIATION 10.2); demanding direct negation then produced mirrors,
        which the independence guard rejects (REMEDIATION 10.3).

        Returns the position and the number of *retries* spent (0 when the first
        attempt was accepted). The caller records that on the turn so a reader
        can tell a clean first pass from one that burned the whole budget.
        """
        # The opposing propositions have already passed the citation channel, so
        # quoting them back cannot introduce a citation this side did not author.
        base_content = f"{question}{self._rebuttal_block(opposing)}"
        opposing_claims = self._opposing_claims(opposing)
        # Feedback accumulated from a rejected attempt, fed back on the re-roll.
        # Re-rolling on temperature alone re-runs the same mistake; naming the
        # offending propositions is what changes the next draft.
        feedback = ""

        # Initialised before the loop: the old `dir()` guard left this unbound on
        # the CitationDetected path, so a position rejected three times for
        # citations returned note="" and read as a parse failure.
        last_error = ""
        last_mirrored: tuple[list[CitationSlot], str, list[Mirror]] | None = None

        for attempt in range(self.max_regenerations):
            # A rejected turn must be re-rolled, not re-run. At temperature 0 with
            # a fixed seed every retry reproduces the offending output verbatim,
            # so the regeneration budget is spent without ever changing anything.
            raw = client.chat(
                [
                    {"role": "system", "content": _system(side)},
                    {"role": "user", "content": f"{base_content}{feedback}"},
                ],
                config={"temperature": 0.0 if attempt == 0 else 0.7, "seed": 7 + attempt},
            )
            try:
                slots = _try_parse_slots(raw)
                # Scan every model-authored field, not just the proposition. A
                # model blocked from citing in `proposition` will otherwise route
                # the case name through `court_hint` and defeat the invariant.
                for slot in slots:
                    self.channel.scan(slot.proposition, role="assistant")
                    self.channel.scan(slot.court_hint, role="assistant")
            except CitationDetected as exc:
                last_error = f"attempt {attempt}: citation string(s) rejected: {exc.hits}"
                feedback = _POSITION_CITATION_FEEDBACK.format(
                    hits="\n".join(f"  - {hit}" for hit in exc.hits)
                )
                continue
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = f"attempt {attempt}: parse failed: {exc}"
                feedback = _PARSE_FEEDBACK.format(error=exc)
                continue

            try:
                self.independence.scan(
                    [slot.proposition for slot in slots], opposing_claims
                )
            except MirrorDetected as exc:
                last_error = (
                    f"attempt {attempt}: {len(exc.mirrors)} proposition(s) merely "
                    f"negated the opposing side"
                )
                # Keep the best-so-far draft. Independence is a quality property,
                # not a safety one: a mirrored antithesis is worth more than no
                # antithesis, so this degrades visibly instead of failing closed.
                if last_mirrored is None or len(exc.mirrors) < len(last_mirrored[2]):
                    last_mirrored = (slots, raw, exc.mirrors)
                feedback = _MIRROR_FEEDBACK.format(
                    mirrors="\n".join(f"- {m.candidate}" for m in exc.mirrors)
                )
                continue

            return (
                Position(
                    side=side,  # type: ignore[arg-type]
                    model=client.name,
                    family=detect_family(client.name),
                    propositions=slots,
                    raw=raw,
                ),
                attempt,
            )

        if last_mirrored is not None:
            # Every attempt mirrored. Return the least-bad draft with each
            # mirroring proposition flagged, rather than discarding the position:
            # the reader needs to know the antithesis conceded the framing.
            slots, raw, mirrors = last_mirrored
            mirrored_text = {m.candidate for m in mirrors}
            flagged = [
                slot.model_copy(
                    update={
                        "note": (
                            f"{slot.note}; " if slot.note else ""
                        )
                        + "independence guard: this merely negates the opposing "
                        "proposition rather than arguing independently "
                        f"(unchanged after {self.max_regenerations} attempts)"
                    }
                )
                if slot.proposition in mirrored_text
                else slot
                for slot in slots
            ]
            return (
                Position(
                    side=side,  # type: ignore[arg-type]
                    model=client.name,
                    family=detect_family(client.name),
                    propositions=flagged,
                    raw=raw,
                ),
                self.max_regenerations,
            )

        # All regeneration attempts failed: return an empty, flagged position.
        return (
            Position(
                side=side,  # type: ignore[arg-type]
                model=client.name,
                family=detect_family(client.name),
                propositions=[
                    CitationSlot(
                        proposition="(generation failed or contained a citation string)",
                        status=SlotStatus.NOT_FOUND,
                        note=(
                            f"{last_error} "
                            f"(after {self.max_regenerations} attempts)"
                        ).strip(),
                    )
                ],
                raw="",
            ),
            self.max_regenerations,
        )

    def _generate_synthesis(self, question: str, turn: DialecticTurn) -> tuple[str, int]:
        """Synthesise from both positions AND the derived crux table.

        The prompt asks the model to name the decisive crux, so the crux table
        has to exist before this runs — it used to be generated afterwards, and
        the one piece of derived structure the module produces was withheld from
        the role that most needs it.

        The old prompt also rendered the thesis into the antithesis slot via a
        `model_copy` and pasted that alongside the real JSON for both sides, so
        the model saw the thesis three times and the antithesis once.
        """
        prompt = (
            f"Question: {question}\n\n"
            f"THESIS:\n{turn.thesis.model_dump_json()}\n\n"
            f"ANTITHESIS:\n{turn.antithesis.model_dump_json()}\n\n"
            f"{copy_crux_table(turn)}"
        )

        stale = self._non_operative_cites(turn)
        self._pending_synthesis_note = ""
        last_error = ""
        best: str | None = None
        feedback = ""

        for attempt in range(self.max_regenerations):
            # Same re-roll discipline the positions get. The channel is strict
            # enough that the `X v. Y` pattern fires on ordinary paraphrase, so
            # a single-shot synthesis fails routinely rather than rarely.
            raw = self.synthesis_client.chat(
                [
                    {"role": "system", "content": _SYNTHESIS_PROMPT},
                    {"role": "user", "content": f"{prompt}{feedback}"},
                ],
                config={"temperature": 0.0 if attempt == 0 else 0.7, "seed": 7 + attempt},
            )
            try:
                self.channel.scan(raw, role="assistant")
            except CitationDetected as exc:
                last_error = f"attempt {attempt}: citation string(s) rejected: {exc.hits}"
                # Say what tripped, as the status branch below does. Without
                # this the retry differed only by temperature and seed, so the
                # same phrasing came back and the budget bought nothing.
                feedback = _CITATION_FEEDBACK.format(
                    hits="\n".join(f"  - {hit}" for hit in exc.hits)
                )
                continue

            # The synthesis is the only role that sees an authority's status, so
            # it is the only one that can say a rule is no longer operative.
            # Asking it in the prompt was not enough: it complied on one run and
            # not the next two, which is how question D6 failed its temporal gate
            # while the module was otherwise correct.
            if stale and not _acknowledges_non_operative(raw):
                last_error = (
                    f"attempt {attempt}: synthesis did not state that "
                    f"{', '.join(sorted(stale))} is no longer operative"
                )
                best = best or raw.strip()
                feedback = _STATUS_FEEDBACK.format(cites=", ".join(sorted(stale)))
                continue

            return raw.strip(), attempt

        if best is not None:
            # Independence-guard discipline: this is a quality failure, not a
            # safety one, so keep the draft and mark it rather than discarding a
            # usable synthesis. The warning goes on `synthesis_note`, never into
            # `synthesis`: appending it there would make any check for "did the
            # response acknowledge the repeal" pass on our own words.
            self._pending_synthesis_note = (
                f"this synthesis relies on {', '.join(sorted(stale))}, which is no "
                f"longer operative law, and did not say so after "
                f"{self.max_regenerations} attempts; treat that authority as "
                f"precedential only"
            )
            return best, self.max_regenerations

        return (
            f"(synthesis contained a citation string and was rejected — {last_error}; "
            f"after {self.max_regenerations} attempts)",
            self.max_regenerations,
        )

    @staticmethod
    def _non_operative_cites(turn: DialecticTurn) -> set[str]:
        """Citations in this turn whose authority is flagged as not in force."""
        return {
            slot.normalized_cite
            for slot in (*turn.thesis.propositions, *turn.antithesis.propositions)
            if slot.normalized_cite and NOT_OPERATIVE in slot.note
        }

    def _verify_position(self, position: Position, ledger: BudgetLedger) -> Position:
        if self.courtlistener is None:
            return position
        result = verify_position(position, self.courtlistener, ledger=ledger)
        return position.model_copy(update={"propositions": result.citations})

    def _retrieve_position(self, position: Position) -> Position:
        """Propose candidate cites for each slot, moving them to `proposed`.

        A proposed cite is a candidate, not authority: the slot is marked
        `proposed` rather than `verified`, and only a CourtListener lookup can
        promote it. A retrieval miss leaves the slot untouched.
        """
        if self.retriever is None:
            return position

        proposed = []
        for slot in position.propositions:
            if slot.normalized_cite or slot.status != SlotStatus.PENDING:
                proposed.append(slot)
                continue
            try:
                candidates = self.retriever.propose(slot.court_hint, slot.proposition)
            except Exception as exc:  # noqa: BLE001 - retrieval must not void a turn
                proposed.append(
                    slot.model_copy(update={"note": f"retrieval failed: {exc}"})
                )
                continue
            if not candidates:
                proposed.append(
                    slot.model_copy(
                        update={"note": "retrieval proposed no candidate citation"}
                    )
                )
                continue
            proposed.append(
                slot.model_copy(
                    update={
                        "normalized_cite": self._select_candidate(slot, candidates),
                        "status": SlotStatus.PROPOSED,
                        "note": "candidate proposed by retrieval; awaiting verification",
                    }
                )
            )
        return position.model_copy(update={"propositions": proposed})

    @staticmethod
    def _select_candidate(slot: CitationSlot, candidates: list[str]) -> str:
        """Pick the candidate the slot actually asked for.

        Retrieval ranks on similarity, which on this corpus means a long,
        heavily-quoted opinion outscores the regulation a proposition is
        explicitly about: every slot on an export-control question was assigned
        a First Amendment case while 15 C.F.R. 734.13(b) sat unused at rank two.
        The ``court_hint`` says which instrument was wanted, so use it.

        Deliberately timid, and only overrides the top candidate when the hint
        names a *statutory instrument* and does not name case law. A hint like
        "First Amendment analysis of government regulation of speech" contains
        "regulation" while plainly asking for precedent, so bare "regulation" is
        not a signal and a case marker anywhere in the hint settles it. When the
        signals are mixed, absent, or nothing non-case was retrieved, the
        similarity ranking stands — an uncertain reading must not be allowed to
        attach a regulation to a proposition that wanted a case, which is a
        worse error than the one being fixed.
        """
        if not candidates:
            return ""
        hint = slot.court_hint.lower()
        if any(marker in hint for marker in _CASE_HINTS):
            return candidates[0]
        if not any(marker in hint for marker in _INSTRUMENT_HINTS):
            return candidates[0]
        for cite in candidates:
            if not is_case_citation(cite):
                return cite
        return candidates[0]

    def _confirm(self, cite: str) -> bool:
        """Whether the retriever holds *cite* as operative authority."""
        confirm = getattr(self.retriever, "confirm", None)
        if not callable(confirm):
            return False
        try:
            return bool(confirm(cite))
        except Exception:  # noqa: BLE001 - confirmation must not void a turn
            return False

    def _confirm_position(self, position: Position) -> Position:
        """Confirm slots CourtListener structurally cannot adjudicate.

        Runs AFTER ``_verify_position`` so a lookup always wins: a slot the
        endpoint already resolved is never revisited, and one it rejected stays
        rejected. Only the slots it could never speak to are considered — which
        is why the guard is ``is_case_citation``, the same predicate
        ``verify_position`` uses to decide what to send, rather than a status
        check. Reusing the predicate is deliberate: two different notions of
        "case citation" would eventually disagree and silently confirm something
        the endpoint had already refused.

        Runs BEFORE ``_annotate_position``, so a rescinded rule still collects
        its note. ``confirm`` refuses anything not in force, so the two cannot
        contradict each other, but the ordering means the note survives even if
        that ever changes.
        """
        confirmed = []
        for slot in position.propositions:
            if (
                slot.normalized_cite
                and slot.status is not SlotStatus.VERIFIED
                and not is_case_citation(slot.normalized_cite)
                and self._confirm(slot.normalized_cite)
            ):
                note = "confirmed against the local corpus, not by citation lookup"
                confirmed.append(
                    slot.model_copy(
                        update={
                            "status": SlotStatus.VERIFIED,
                            "verified_by": "corpus",
                            "note": f"{slot.note}; {note}" if slot.note else note,
                        }
                    )
                )
            else:
                confirmed.append(slot)
        return position.model_copy(update={"propositions": confirmed})

    def _annotate(self, cite: str) -> str:
        """Status note for *cite*, when the retriever can supply one."""
        annotate = getattr(self.retriever, "annotate", None)
        if not callable(annotate):
            return ""
        try:
            return str(annotate(cite) or "")
        except Exception:  # noqa: BLE001 - annotation must not void a turn
            return ""

    def _annotate_position(self, position: Position) -> Position:
        """Flag any slot resting on authority that is no longer operative.

        This runs AFTER verification, not during retrieval, because
        ``verify_position`` rewrites ``slot.note`` on every branch — annotating
        earlier meant the status was silently overwritten before the synthesis
        could ever see it, which is exactly what happened on the first attempt
        at this fix.
        """
        annotated = []
        for slot in position.propositions:
            note = self._annotate(slot.normalized_cite) if slot.normalized_cite else ""
            if note:
                annotated.append(
                    slot.model_copy(
                        update={"note": f"{slot.note}; {note}" if slot.note else note}
                    )
                )
            else:
                annotated.append(slot)
        return position.model_copy(update={"propositions": annotated})

    _NO_RETRIEVAL_NOTE = (
        "unverified by construction: no retrieval stage configured, so no "
        "candidate citation was ever proposed for this slot"
    )

    def _note_unretrieved(self, position: Position) -> Position:
        """Say why a slot is unverified instead of leaving a bare `pending`.

        A slot that no retrieval stage ever touched is not "in progress" — it is
        structurally unverifiable, and a reader must be told that rather than
        left to infer it from a status that reads as pending work.
        """
        annotated = []
        for slot in position.propositions:
            if slot.status == SlotStatus.PENDING and not slot.normalized_cite:
                # Append rather than overwrite: a weight-coercion note recorded
                # at parse time is diagnostic too and must survive.
                note = f"{slot.note}; {self._NO_RETRIEVAL_NOTE}" if slot.note else self._NO_RETRIEVAL_NOTE
                annotated.append(slot.model_copy(update={"note": note}))
            else:
                annotated.append(slot)
        return position.model_copy(update={"propositions": annotated})

    def chat(
        self,
        question: str,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> DialecticTurn:
        """Run one full dialectic turn.

        The pipeline is::

            generate ──► retrieve(court_hint, proposition) ──► candidate cites
                                                                     │
                                                    CourtListener verification
                                                                     │
                                             extract cruxes ──► synthesise

        Synthesis runs last because its prompt asks the model to name the
        decisive crux, and verification runs before extraction because
        ``PrecedenceRule.rank`` reads ``status``.

        ``on_event`` receives each stage as it completes. Unlike the multi-chat
        panel, this pipeline is strictly sequential and its stages are
        heterogeneous -- generation is a model call, verification is a network
        round-trip to CourtListener, crux extraction is an NLI pass -- so a
        caller shown only a spinner cannot tell a slow debate from a hung one.
        Wrapped, so a subscriber that raises cannot break the turn.
        """

        def report(kind: str, **fields: object) -> None:
            if on_event is None:
                return
            with contextlib.suppress(Exception):
                on_event({"event": kind, **fields})

        ledger = BudgetLedger()
        report("generating", side="thesis")
        thesis, thesis_retries = self._generate_position(
            "thesis", question, self.thesis_client
        )
        report(
            "position_generated",
            side="thesis",
            propositions=len(thesis.propositions),
            retries=thesis_retries,
        )

        # The antithesis sees the thesis and must contradict it directly. Both
        # sides generated blind from the same question frequently argued the
        # same position, and a dialectic whose sides agree has no cruxes.
        report("generating", side="antithesis")
        antithesis, antithesis_retries = self._generate_position(
            "antithesis", question, self.antithesis_client, opposing=thesis
        )
        report(
            "position_generated",
            side="antithesis",
            propositions=len(antithesis.propositions),
            retries=antithesis_retries,
        )

        # Retrieval sits between generation and verification: it is the only
        # thing that turns a plain-English `court_hint` into a candidate cite.
        # Without it `verify_position` finds no filled slots and returns early,
        # so the network is never touched and every slot stays pending forever.
        report("retrieving")
        thesis = self._retrieve_position(thesis)
        antithesis = self._retrieve_position(antithesis)
        report(
            "retrieved",
            filled=sum(
                1
                for position in (thesis, antithesis)
                for slot in position.propositions
                if slot.normalized_cite
            ),
        )

        # Annotate last: verification rewrites `note`, so a status attached any
        # earlier is overwritten before the synthesis can act on it.
        report("verifying")
        thesis = self._annotate_position(
            self._confirm_position(
                self._note_unretrieved(self._verify_position(thesis, ledger))
            )
        )
        antithesis = self._annotate_position(
            self._confirm_position(
                self._note_unretrieved(self._verify_position(antithesis, ledger))
            )
        )
        # The slowest stage, and the one that reaches the network. Reporting the
        # budget spent is what tells a reader whether it did any work at all.
        report("verified", calls_spent=ledger.calls_spent)

        turn = DialecticTurn(
            question=question,
            thesis=thesis,
            antithesis=antithesis,
            synthesis="",
            cruxes=[],
            ledger=ledger,
        )

        # Crux extraction is derived by a separate NLI pass, never by the
        # debaters, and it must precede synthesis so the crux table can be shown
        # to the synthesis model.
        turn.cruxes = self.crux_extractor.extract(thesis, antithesis)
        turn.crux_note = self._crux_note(turn)
        # An empty crux table is a real outcome, not a failure, so the count goes
        # out either way -- and `note` says why when it is zero.
        report("cruxes_extracted", count=len(turn.cruxes), note=turn.crux_note)

        report("synthesising")
        turn.synthesis, synthesis_retries = self._generate_synthesis(question, turn)
        turn.synthesis_note = self._pending_synthesis_note

        turn.regenerated = thesis_retries + antithesis_retries + synthesis_retries
        turn.calls_spent = ledger.calls_spent
        return turn

    @staticmethod
    def _crux_note(turn: DialecticTurn) -> str:
        """Explain an empty crux table instead of rendering nothing."""
        if turn.cruxes:
            return ""
        thesis_count = len(turn.thesis.propositions)
        antithesis_count = len(turn.antithesis.propositions)
        if not thesis_count or not antithesis_count:
            return "no cruxes: one side produced no propositions"
        return (
            f"no cruxes: the NLI pass found no contradiction across "
            f"{thesis_count} thesis x {antithesis_count} antithesis propositions "
            f"({thesis_count * antithesis_count} pairs compared)"
        )

    def _arm(
        self,
        thesis_client: ChatClient,
        antithesis_client: ChatClient,
        synthesis_client: ChatClient,
    ) -> DialecticChat:
        """Build a sibling engine for one arm of the correlation experiment.

        The family guard is deliberately suppressed here: the same-family arm
        exists precisely to run the configuration the guard forbids. Everything
        else is the production path. The previous harness bypassed both the
        guard and verification, so it counted cruxes over slots whose `status`
        was always `pending` — a different code path from production, where
        `PrecedenceRule.rank` reads `status`.
        """
        arm = object.__new__(DialecticChat)
        arm.thesis_client = thesis_client
        arm.antithesis_client = antithesis_client
        arm.synthesis_client = synthesis_client
        arm.nli_client = self.nli_client
        arm.courtlistener = self.courtlistener
        arm.retriever = self.retriever
        arm.channel = self.channel
        arm.independence = self.independence
        arm.crux_extractor = self.crux_extractor
        arm.max_regenerations = self.max_regenerations
        return arm

    def correlation_guard_report(
        self,
        questions: list[str],
        distinct_pair: tuple[ChatClient, ChatClient],
        same_family_pair: tuple[ChatClient, ChatClient],
        synthesis_client: ChatClient | None = None,
    ) -> CorrelationGuardReport:
        """Compare crux yield for a distinct-family and a same-family debate pair.

        Both arms run the full production path via :meth:`chat`.

        Callers must supply *genuinely different checkpoints* for the
        same-family arm. The previous version built it by wrapping the
        antithesis client and relabelling it with the thesis's name, so both
        roles ran the same client over byte-identical text. The resulting
        "collapse" measured only that identical text does not contradict
        itself — true, and not evidence about model families.
        """
        synth = synthesis_client or self.synthesis_client

        distinct_arm = self._arm(distinct_pair[0], distinct_pair[1], synth)
        same_arm = self._arm(same_family_pair[0], same_family_pair[1], synth)

        distinct_counts = [len(distinct_arm.chat(q).cruxes) for q in questions]
        same_counts = [len(same_arm.chat(q).cruxes) for q in questions]

        return CorrelationGuardReport(
            questions=list(questions),
            distinct_models=(distinct_pair[0].name, distinct_pair[1].name),
            same_family_models=(same_family_pair[0].name, same_family_pair[1].name),
            distinct_counts=distinct_counts,
            same_family_counts=same_counts,
        )


@dataclass
class CorrelationGuardReport:
    """Result of one correlation-guard run. Reports data, not a verdict."""

    questions: list[str]
    distinct_models: tuple[str, str]
    same_family_models: tuple[str, str]
    distinct_counts: list[int]
    same_family_counts: list[int]

    @staticmethod
    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def distinct_mean(self) -> float:
        return self._mean(self.distinct_counts)

    @property
    def same_family_mean(self) -> float:
        return self._mean(self.same_family_counts)

    @property
    def separates(self) -> bool:
        """Direction only: did the distinct-family arm yield more cruxes?

        **This is not a significance test.** It returns True for a 3-vs-2 split
        over 8 questions, which is indistinguishable from chance. Read it
        alongside :attr:`total_distinct` / :attr:`total_same_family` and the raw
        per-question counts; do not quote it as evidence on its own. Reporting a
        bare direction as confirmation is what produced the original §5 claim.
        """
        return self.distinct_mean > self.same_family_mean

    @property
    def total_distinct(self) -> int:
        return sum(self.distinct_counts)

    @property
    def total_same_family(self) -> int:
        return sum(self.same_family_counts)

    @property
    def disagreements(self) -> list[tuple[int, int]]:
        """Per-question (distinct, same) pairs where the two arms differ.

        If these point in both directions, the difference in means is noise.
        """
        return [
            (d, s)
            for d, s in zip(self.distinct_counts, self.same_family_counts, strict=True)
            if d != s
        ]

    def summary(self) -> str:
        both_ways = any(d > s for d, s in self.disagreements) and any(
            s > d for d, s in self.disagreements
        )
        return (
            f"distinct    {self.distinct_models[0]} / {self.distinct_models[1]}: "
            f"mean {self.distinct_mean:.2f}, total {self.total_distinct} "
            f"{self.distinct_counts}\n"
            f"same-family {self.same_family_models[0]} / {self.same_family_models[1]}: "
            f"mean {self.same_family_mean:.2f}, total {self.total_same_family} "
            f"{self.same_family_counts}\n"
            f"questions: {len(self.questions)}; "
            f"arms differ on {len(self.disagreements)} question(s)"
            f"{' — in BOTH directions' if both_ways else ''}\n"
            f"direction favours distinct-family: {self.separates} "
            f"(direction only, NOT a significance test)"
        )
