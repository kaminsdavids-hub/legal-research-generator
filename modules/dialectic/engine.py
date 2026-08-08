"""Dialectic chat orchestrator: thesis / antithesis / synthesis + crux extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .channel import CitationChannel, CitationDetected
from .copy import copy_crux_table
from .crux import CruxExtractor
from .models import (
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
from .verification import CourtListenerClient, verify_position


class FamilyCollision(Exception):
    """Raised when two debate roles share the same base model family."""


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat interface; matches the project's LLMClient protocol."""

    name: str

    def chat(self, messages: list[Any], config: Any | None = None) -> str: ...


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

_REBUT_PROMPT = (
    "\n\nThe thesis has asserted the following propositions. Contradict these "
    "DIRECTLY: each of your propositions should be the negation of one of them, "
    "addressing the same predicate, not a variation on a different point.\n"
    "{propositions}"
)

_SYNTHESIS_PROMPT = (
    "You are the synthesis in a legal dialectic.\n"
    "The user question, thesis, and antithesis are provided below.\n"
    "Concisely reconcile the strongest points, name the decisive crux, and state "
    "the likely outcome.\n"
    "Do NOT cite specific cases, reporter strings, or 'X v. Y' names.\n"
    "Use only plain prose."
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
        max_regenerations: int = 3,
    ) -> None:
        self.thesis_client = thesis_client
        self.antithesis_client = antithesis_client
        self.synthesis_client = synthesis_client
        self.nli_client = nli_client
        self.courtlistener = courtlistener
        self.retriever = retriever
        self.channel = channel or CitationChannel()
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
    def _rebuttal_block(opposing: Position | None) -> str:
        """Render the opposing side's propositions for a direct rebuttal.

        Empty when there is nothing usable to rebut — notably when the opposing
        position failed generation, whose placeholder proposition must never be
        fed back into another model as if it were an argument.
        """
        if opposing is None:
            return ""
        claims = [
            slot.proposition
            for slot in opposing.propositions
            if slot.status != SlotStatus.NOT_FOUND and slot.proposition
        ]
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
        """Generate one side, retrying on parse failure or citation leak.

        When *opposing* is supplied, its propositions are shown to this side to
        be contradicted directly. Generating both sides blind from the same
        question let them argue the same position, which yields no cruxes by
        construction (REMEDIATION 10).

        Returns the position and the number of *retries* spent (0 when the first
        attempt was accepted). The caller records that on the turn so a reader
        can tell a clean first pass from one that burned the whole budget.
        """
        # The opposing propositions have already passed the citation channel, so
        # quoting them back cannot introduce a citation this side did not author.
        user_content = f"{question}{self._rebuttal_block(opposing)}"
        messages = [
            {"role": "system", "content": _system(side)},
            {"role": "user", "content": user_content},
        ]
        # Initialised before the loop: the old `dir()` guard left this unbound on
        # the CitationDetected path, so a position rejected three times for
        # citations returned note="" and read as a parse failure.
        last_error = ""
        for attempt in range(self.max_regenerations):
            # A rejected turn must be re-rolled, not re-run. At temperature 0 with
            # a fixed seed every retry reproduces the offending output verbatim,
            # so the regeneration budget is spent without ever changing anything.
            raw = client.chat(
                messages,
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
                continue
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = f"attempt {attempt}: parse failed: {exc}"
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

        last_error = ""
        for attempt in range(self.max_regenerations):
            # Same re-roll discipline the positions get. The channel is strict
            # enough that the `X v. Y` pattern fires on ordinary paraphrase, so
            # a single-shot synthesis fails routinely rather than rarely.
            raw = self.synthesis_client.chat(
                [
                    {"role": "system", "content": _SYNTHESIS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                config={"temperature": 0.0 if attempt == 0 else 0.7, "seed": 7 + attempt},
            )
            try:
                self.channel.scan(raw, role="assistant")
            except CitationDetected as exc:
                last_error = f"attempt {attempt}: citation string(s) rejected: {exc.hits}"
                continue
            return raw.strip(), attempt

        return (
            f"(synthesis contained a citation string and was rejected — {last_error}; "
            f"after {self.max_regenerations} attempts)",
            self.max_regenerations,
        )

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
                        "normalized_cite": candidates[0],
                        "status": SlotStatus.PROPOSED,
                        "note": "candidate proposed by retrieval; awaiting verification",
                    }
                )
            )
        return position.model_copy(update={"propositions": proposed})

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

    def chat(self, question: str) -> DialecticTurn:
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
        """
        ledger = BudgetLedger()
        thesis, thesis_retries = self._generate_position(
            "thesis", question, self.thesis_client
        )
        # The antithesis sees the thesis and must contradict it directly. Both
        # sides generated blind from the same question frequently argued the
        # same position, and a dialectic whose sides agree has no cruxes.
        antithesis, antithesis_retries = self._generate_position(
            "antithesis", question, self.antithesis_client, opposing=thesis
        )

        # Retrieval sits between generation and verification: it is the only
        # thing that turns a plain-English `court_hint` into a candidate cite.
        # Without it `verify_position` finds no filled slots and returns early,
        # so the network is never touched and every slot stays pending forever.
        thesis = self._retrieve_position(thesis)
        antithesis = self._retrieve_position(antithesis)

        thesis = self._note_unretrieved(self._verify_position(thesis, ledger))
        antithesis = self._note_unretrieved(self._verify_position(antithesis, ledger))

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

        turn.synthesis, synthesis_retries = self._generate_synthesis(question, turn)

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
