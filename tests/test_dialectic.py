"""Dialectic chat module tests covering Stages 0-4 and canaries C1-C6."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from modules.dialectic.channel import CitationChannel, CitationDetected
from modules.dialectic.copy import copy_crux_table, copy_exchange, copy_position
from modules.dialectic.crux import CruxExtractor, PrecedenceRule
from modules.dialectic.engine import DialecticChat, FamilyCollision
from modules.dialectic.independence import IndependenceGuard, MirrorDetected
from modules.dialectic.models import (
    CitationSlot,
    Crux,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)
from modules.dialectic.nli import NLIEvaluator
from modules.dialectic.retrieval import CiteRetriever, NullCiteRetriever, StubCiteRetriever
from modules.dialectic.roles import detect_family
from modules.dialectic.verification import (
    BudgetExhausted,
    CourtListenerClient,
    RateBudget,
    verify_position,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeHTTP:
    """In-memory stand-in for httpx with programmable status/response."""

    def __init__(
        self,
        status_code: int = 200,
        payload: Any | None = None,
        exception: type[Exception] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload if payload is not None else []
        self.exception = exception
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, "kwargs": kwargs})
        if self.exception is not None:
            raise self.exception("forced failure")
        return _FakeResponse(self.status_code, self.payload)


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad status",
                request=None,
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        return self._payload


class _FakeLLM:
    """Deterministic chat client for tests."""

    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response = response

    def chat(self, messages: list[Any], config: Any | None = None) -> str:
        return self.response


def _slot_json(
    proposition: str,
    court_hint: str,
    weight: str,
    normalized_cite: str = "",
) -> str:
    return json.dumps(
        {
            "propositions": [
                {
                    "proposition": proposition,
                    "court_hint": court_hint,
                    "weight": weight,
                    "normalized_cite": normalized_cite,
                }
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Stage 0 — CitationChannel
# --------------------------------------------------------------------------- #
def test_citation_channel_reports_zero_false_negatives() -> None:
    channel = CitationChannel()

    reporters = [
        "U.S.",
        "F.2d",
        "F.3d",
        "F.4th",
        "F. Supp.",
        "F. Supp. 2d",
        "F. Supp. 3d",
        "F. App'x",
        "S. Ct.",
        "L. Ed.",
        "L. Ed. 2d",
        "A.2d",
        "A.3d",
        "N.E.2d",
        "N.E.3d",
        "N.W.2d",
        "S.E.2d",
        "S.W.2d",
        "S.W.3d",
        "P.2d",
        "P.3d",
        "B.R.",
        "Cal. App. 4th",
        "Cal. App. 3d",
        "N.Y.2d",
        "N.Y.3d",
        "So.2d",
        "So.3d",
    ]

    well_formed_reporters: list[str] = []
    for i, rep in enumerate(reporters):
        vol = 100 + i
        page = 400 + i
        year = 1980 + (i % 40)
        party1 = ("Smith", "Jones", "Doe", "Roe")[i % 4]
        party2 = ("Acme", "Beta", "Gamma", "Delta")[i % 4]
        well_formed_reporters.append(f"{party1} v. {party2}, {vol} {rep} {page} ({year}).")
        well_formed_reporters.append(f"See {vol} {rep} {page} (internal pagination omitted).")
        well_formed_reporters.append(f"{vol}   {rep}   {page} (extra spaces).")
        well_formed_reporters.append(f"Cf. {party1} v. {party2}, {vol} {rep} {page}, at 12.")

    well_formed_cases = [
        "Brown v. Board",
        "Miranda v. Arizona",
        "Marbury v. Madison",
        "Gideon v. Wainwright",
        "Roe v. Wade",
        "Dred Scott v. Sandford",
        "Plessy v. Ferguson",
        "McCulloch v. Maryland",
        "Obergefell v. Hodges",
        "United States v. Nixon",
        "Terry v. Ohio",
        "Ernst & Ernst v. Hochfelder",
        "New York Times Co. v. Sullivan",
        "District of Columbia v. Heller",
        "Citizens United v. FEC",
        "Griswold v. Connecticut",
        "Korematsu v. United States",
        "Lawrence v. Texas",
        "Brandenburg v. Ohio",
        "Mapp v. Ohio",
        "Youngstown Sheet & Tube Co. v. Sawyer",
        "Wickard v. Filburn",
        "Loving v. Virginia",
        "Shelby County v. Holder",
        "National Federation of Independent Business v. Sebelius",
        "Masterpiece Cakeshop v. Colorado Civil Rights Commission",
        "Carpenter v. United States",
        "Riley v. California",
        "Boumediene v. Bush",
        "Hamdi v. Rumsfeld",
        "INS v. Chadha",
        "Buckley v. Valeo",
        "Reynolds v. Sims",
        "Baker v. Carr",
        "Heart of Atlanta Motel v. United States",
        "Katzenbach v. McClung",
        "NLRB v. Jones & Laughlin Steel Corp.",
        "Schechter Poultry Corp. v. United States",
        "Humphrey's Executor v. United States",
    ]

    split_and_quoted = [
        "Terry v. Ohio,\n392 U.S. 1 (1968)",
        'The user wrote: "Miranda v. Arizona, 384 U.S. 436 (1966)"',
        "In \nBrown v. Board, 347 U.S. 483 (1954)",
        "Smith v. Jones,\n 123 F. Supp. 2d 456 (2001)",
        "See id.; but see Roe v. Wade, 410 U.S. 113 (1973)",
        "Obergefell v. Hodges (\n576 U.S. 644) established marriage equality.",
        'She asked, "What about Terry v. Ohio, 392 U.S. 1?"',
        "The brief cited\n123 F.3d 789 (1997).",
        "Compare \n456 U.S. 694 (1982).",
        "In re \nSmith, 88 B.R. 12 (Bankr. 1988).",
        "Jones v. Smith,\n200 F. Supp. 3d 50 (D.D.C. 2016).",
        'He wrote: "See 567 So. 2d 890 (Fla. 1990)."',
        "The panel cited\n11 F.4th 321 (5th Cir. 2021).",
        "Marbury v.\nMadison, 5 U.S. 137 (1803).",
        "Gideon v.\nWainwright, 372 U.S. 335 (1963).",
        "As held in\nPlessy v. Ferguson, 163 U.S. 537 (1896).",
        'The complaint quoted "Korematsu v. United States, 323 U.S. 214 (1944)"',
        "See also\n42 A.3d 100 (Md. 2012).",
        "The court noted\n88 N.E.3d 500 (Mass. 2017).",
        "Compare \nMapp v. Ohio, 367 U.S. 643 (1961).",
    ]

    malformed = [
        "392 U.S.",
        "Terry v.",
        "U.S. 1 without volume",
        "F.2d 456 without volume",
        "123 456 (year only)",
        "Smith Jones 123 F.2d 456 (missing v.)",
        "v. Ohio, 392 U.S. 1 (missing first party)",
        "Smith v. 392 U.S. 1 (missing second party)",
    ]

    non_cites = [
        "The U.S. economy grew by 2%.",
        "Return on investment v. risk is a classic tradeoff.",
        "A v. B is a variable naming convention.",
        "F. Supp. is an abbreviation for Federal Supplement.",
        "The 123 F. 2d series is well known.",
        "Id. at 5.",
        "supra note 3",
        "See generally the rule above.",
        "The statute requires scienter.",
        "42 U.S.C. § 1983 is a civil-rights provision.",
        "15 CFR 78j(b) governs securities fraud.",
        "The parties dispute the factual record.",
        "Evidence shows the defendant signed the contract.",
        "Vol. 123, page 456 (no reporter)",
        "Compare apples v. oranges as metaphors.",
        "This proposition is unsupported.",
        "No citation string appears in this sentence.",
        "The holding applies to the facts.",
        "My favorite color is blue.",
        "The quick brown fox jumps over the lazy dog.",
        "Please review the draft before submission.",
        "The model output must avoid citations.",
        "Section 1983 actions are common.",
        "Rule 10b-5 prohibits securities fraud.",
        "The Fourth Amendment protects searches.",
        "Due process requires notice and hearing.",
        "The burden of proof rests with the plaintiff.",
        "Statutory text controls when plain.",
        "Legislative history may illuminate ambiguity.",
        "Canons of construction aid interpretation.",
        "The record lacks sufficient evidence.",
        "Summary judgment is appropriate here.",
        "The motion to dismiss should be denied.",
        "Plaintiff failed to state a claim.",
        "Subject-matter jurisdiction is undisputed.",
        "Personal jurisdiction is contested.",
        "Venue lies in the Northern District.",
        "Service of process was defective.",
        "The complaint pleads fraud with particularity.",
        "Discovery disputes remain unresolved.",
        "The expert report is unreliable.",
        "Daubert governs admissibility of expert testimony.",
        "The jury instruction was erroneous.",
        "The sentence was within the guidelines.",
        "Restitution is not authorized by statute.",
        "The appeal is untimely.",
        "Collateral estoppel bars relitigation.",
        "Res judicata applies to the claim.",
        "The arbitration clause is enforceable.",
        "Class certification was improper.",
        "The settlement is fair and adequate.",
        "Injunctive relief requires irreparable harm.",
        "A preliminary injunction was unwarranted.",
        "The contract is ambiguous on its face.",
        "The implied covenant of good faith applies.",
        "Promissory estoppel requires detrimental reliance.",
        "The statute of limitations has run.",
        "Equitable tolling is unavailable here.",
        "The doctrine of laches bars relief.",
    ]

    positives = well_formed_reporters + well_formed_cases + split_and_quoted
    negatives = non_cites
    total = len(positives) + len(negatives) + len(malformed)

    false_negatives = sum(1 for s in positives if not channel.find_hits(s))
    false_positives = sum(1 for s in negatives if channel.find_hits(s))
    malformed_hits = sum(1 for s in malformed if channel.find_hits(s))

    fn_rate = false_negatives / len(positives) if positives else 0.0
    fp_rate = false_positives / len(negatives) if negatives else 0.0

    print(f"CitationChannel adversarial set: {total} strings")
    print(f"False negatives: {false_negatives}/{len(positives)} ({fn_rate:.2%})")
    print(f"False positives: {false_positives}/{len(negatives)} ({fp_rate:.2%})")
    print(f"Malformed strings flagged: {malformed_hits}/{len(malformed)}")

    assert false_negatives == 0, f"well-formed reporter cite missed: {fn_rate:.2%}"


def test_citation_channel_scans_assistant_but_not_user() -> None:
    channel = CitationChannel()
    with pytest.raises(CitationDetected):
        channel.scan("Terry v. Ohio, 392 U.S. 1", role="assistant")
    channel.scan("Terry v. Ohio, 392 U.S. 1", role="user")  # user text is exempt


def test_citation_channel_detects_split_cite() -> None:
    channel = CitationChannel()
    text = "Terry v. Ohio,\n392 U.S. 1 (1968)"
    with pytest.raises(CitationDetected):
        channel.scan(text, role="assistant")


# --------------------------------------------------------------------------- #
# Canaries C1-C2 (model output containing citation strings)
# --------------------------------------------------------------------------- #
def test_canary_c1_model_output_with_full_cite_is_rejected() -> None:
    channel = CitationChannel()
    model_output = "The stop was justified under Terry v. Ohio, 392 U.S. 1 (1968)."
    with pytest.raises(CitationDetected):
        channel.scan(model_output, role="assistant")


def test_canary_c2_model_output_with_split_newline_cite_is_rejected() -> None:
    channel = CitationChannel()
    model_output = "The stop was justified under Terry v. Ohio,\n392 U.S. 1 (1968)."
    with pytest.raises(CitationDetected):
        channel.scan(model_output, role="assistant")


# --------------------------------------------------------------------------- #
# Stage 1 — Rate budget + CourtListener verification
# --------------------------------------------------------------------------- #
def test_rate_budget_enforces_all_three_windows() -> None:
    budget = RateBudget(per_minute=2, per_hour=5, per_day=10)
    budget.spend()
    budget.spend()
    # Third call inside the minute exceeds per-minute cap.
    with pytest.raises(BudgetExhausted):
        budget.check()


def test_rate_budget_cache_hit_does_not_spend() -> None:
    http = _FakeHTTP(
        status_code=200,
        payload=[
            {
                "citation": "392 U.S. 1",
                "normalized_citations": ["392 U.S. 1"],
                "status": 200,
                "clusters": [{"id": 1}],
            }
        ],
    )
    client = CourtListenerClient(token="token", http=http)

    for _ in range(50):
        result = client.lookup("Terry v. Ohio, 392 U.S. 1")
        assert result[0]["status"] == 200

    assert len(http.calls) == 1
    assert client.budget.counts()["day"] == 1


def test_forced_429_leaves_slots_not_found_not_verified() -> None:
    http = _FakeHTTP(status_code=429, payload=[])
    client = CourtListenerClient(token="token", http=http)
    pos = Position(
        side="thesis",
        model="gpt",
        family="gpt",
        propositions=[
            CitationSlot(
                proposition="The stop was justified.",
                court_hint="stop and frisk",
                weight=Weight.CONTROLLING,
                normalized_cite="392 U.S. 1",
            )
        ],
    )
    result = verify_position(pos, client)
    statuses = {s.status for s in result.citations}
    assert SlotStatus.NOT_FOUND in statuses
    assert SlotStatus.VERIFIED not in statuses


def test_forced_timeout_leaves_slots_not_found_not_verified() -> None:
    http = _FakeHTTP(exception=httpx.TimeoutException)
    client = CourtListenerClient(token="token", http=http)
    pos = Position(
        side="thesis",
        model="gpt",
        family="gpt",
        propositions=[
            CitationSlot(
                proposition="The stop was justified.",
                normalized_cite="392 U.S. 1",
            )
        ],
    )
    result = verify_position(pos, client)
    statuses = {s.status for s in result.citations}
    assert SlotStatus.NOT_FOUND in statuses
    assert SlotStatus.VERIFIED not in statuses


def test_reporter_table_has_no_duplicates() -> None:
    """The list is hand-maintained; duplicates are the first sign of drift."""
    from modules.dialectic.channel import _REPORTERS

    assert len(_REPORTERS) == len(set(_REPORTERS))


def test_rate_budget_prunes_outside_the_day_window() -> None:
    budget = RateBudget(per_minute=2, per_hour=5, per_day=10)
    # Two spends a day and a half ago must fall out of every window.
    stale = 0.0
    budget.spend(now=stale)
    budget.spend(now=stale)
    now = stale + 36 * 60 * 60
    assert budget.counts(now=now) == {"minute": 0, "hour": 0, "day": 0}
    budget.check(now=now)  # must not raise


def test_verification_note_names_the_key_mismatch() -> None:
    """A slot must not fall to NOT_FOUND with an opaque note."""
    # CourtListener answers, but keyed on a different normalization.
    payload = [
        {
            "citation": "392 U. S. 1",
            "normalized_citations": ["392 U. S. 1"],
            "status": 200,
            "clusters": [{"id": 1}],
        }
    ]
    http = _FakeHTTP(status_code=200, payload=payload)
    client = CourtListenerClient(token="token", http=http)
    pos = Position(
        side="thesis",
        model="gpt",
        family="gpt",
        propositions=[
            CitationSlot(proposition="The stop was justified.", normalized_cite="392 U.S. 1")
        ],
    )
    result = verify_position(pos, client)
    slot = result.citations[0]
    assert slot.status == SlotStatus.NOT_FOUND
    assert "392 U.S. 1" in slot.note
    assert "no result keyed on" in slot.note


def test_partially_populated_verified_slot_still_renders_a_marker() -> None:
    """The one state that used to render nothing at all."""
    turn = DialecticTurn(
        question="Q",
        thesis=Position(
            side="thesis",
            model="gpt",
            family="gpt",
            propositions=[
                CitationSlot(
                    proposition="Verified but with no citation.",
                    weight=Weight.CONTROLLING,
                    status=SlotStatus.VERIFIED,
                    normalized_cite="",
                )
            ],
        ),
        antithesis=Position(side="antithesis", model="llama", family="llama"),
    )
    text = copy_position(turn, "thesis")
    assert "[UNSUPPORTED" in text
    assert "treat as unresolved" in text


def test_canary_c3_429_mid_verification_no_slot_verified() -> None:
    http = _FakeHTTP(status_code=429, payload=[])
    client = CourtListenerClient(token="token", http=http)
    pos = Position(
        side="thesis",
        model="gpt",
        family="gpt",
        propositions=[
            CitationSlot(
                proposition="Stop justified.",
                normalized_cite="392 U.S. 1",
            ),
            CitationSlot(
                proposition="Another point.",
                normalized_cite="384 U.S. 436",
            ),
        ],
    )
    result = verify_position(pos, client)
    assert all(s.status == SlotStatus.NOT_FOUND for s in result.citations)


# --------------------------------------------------------------------------- #
# Stage 2 — Model roles / family guard / correlation guard
# --------------------------------------------------------------------------- #
def test_detect_family_maps_models_to_families() -> None:
    assert detect_family("llama3.1:8b") == "llama"
    assert detect_family("gpt-oss:20b") == "gpt"
    assert detect_family("gemma3:4b") == "gemma"
    assert detect_family("nemotron-3-nano:4b") == "nemotron"
    assert detect_family("hermes3:8b") == "hermes"
    assert detect_family("saul:7b-instruct-v1") == "saul"


def test_family_collision_guard_raises_when_debaters_share_family() -> None:
    with pytest.raises(FamilyCollision):
        DialecticChat(
            thesis_client=_FakeLLM("gpt-oss", "{}"),
            antithesis_client=_FakeLLM("gpt-oss", "{}"),
            synthesis_client=_FakeLLM("gemma", "{}"),
        )


def test_canary_c5_same_family_guard_fires() -> None:
    with pytest.raises(FamilyCollision):
        DialecticChat(
            thesis_client=_FakeLLM("llama3.1", "{}"),
            antithesis_client=_FakeLLM("llama3.1:latest", "{}"),
            synthesis_client=_FakeLLM("gemma3", "{}"),
        )


def test_correlation_guard_runs_both_arms_through_the_production_path() -> None:
    """Exercises the harness mechanics only — it asserts no empirical claim.

    The old test supplied byte-identical output for both sides of the
    same-family arm and asserted `distinct_mean > same_mean`. That measured
    "identical text does not contradict itself", which is true and says nothing
    about model families. Whether same-family checkpoints actually correlate is
    an empirical question that a unit test with canned strings cannot answer;
    see REMEDIATION 5. What is testable here is that both arms run, that the
    same-family arm is not silently blocked by the family guard, and that the
    report carries the raw per-question counts rather than a bare verdict.
    """
    thesis_json = _slot_json(
        "The statute requires scienter for liability.",
        "securities fraud scienter",
        Weight.CONTROLLING,
    )
    antithesis_json = _slot_json(
        "The statute does not require scienter for liability.",
        "securities fraud scienter",
        Weight.CONTROLLING,
    )

    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3:8b", thesis_json),
        antithesis_client=_FakeLLM("llama3.1:8b", antithesis_json),
        synthesis_client=_FakeLLM("gemma3:4b", "Synthesis text."),
    )
    questions = ["Does scienter apply?"] * 5

    report = chat.correlation_guard_report(
        questions,
        distinct_pair=(
            _FakeLLM("hermes3:8b", thesis_json),
            _FakeLLM("llama3.1:8b", antithesis_json),
        ),
        # Two genuinely different checkpoints of one family, each with its own
        # output — not one client wearing the other's name.
        same_family_pair=(
            _FakeLLM("llama3.1:8b", thesis_json),
            _FakeLLM("llama3.2:3b", antithesis_json),
        ),
    )

    print(report.summary())
    assert len(report.distinct_counts) == len(questions)
    assert len(report.same_family_counts) == len(questions)
    assert report.same_family_models == ("llama3.1:8b", "llama3.2:3b")
    # The same-family arm must actually run, not be blocked by the family guard.
    assert any(c > 0 for c in report.same_family_counts)


# --------------------------------------------------------------------------- #
# Stage 3 — Crux extraction / NLI / precedence rule
# --------------------------------------------------------------------------- #
def test_nli_derives_negates_not_model() -> None:
    nli = NLIEvaluator()
    assert nli.relation(
        "The statute requires scienter.", "The statute does not require scienter."
    ) == "contradiction"
    assert nli.relation(
        "The statute requires scienter.", "Scienter is required under the statute."
    ) == "entailment"
    assert nli.relation(
        "The statute requires scienter.", "The weather is pleasant today."
    ) == "neutral"


# Pairs drawn from the same legal question but making unrelated claims. Exactly
# one side of each carries a negation word. The old heuristic flagged all three
# as contradictions, because `shared_terms` used raw `.split()` over text whose
# stopwords were never dropped — so the rule reduced to "one side is negated".
_UNRELATED_LEGAL_PAIRS = [
    (
        "The statute of limitations for this claim is four years.",
        "There is no federal question jurisdiction over the dispute.",
    ),
    (
        "The contract was executed by an authorized officer of the company.",
        "Punitive damages are not available under this cause of action.",
    ),
    (
        "Discovery closed on the date set in the scheduling order.",
        "The witness never signed the declaration attached to the motion.",
    ),
]


@pytest.mark.parametrize(("premise", "hypothesis"), _UNRELATED_LEGAL_PAIRS)
def test_nli_does_not_flag_unrelated_legal_prose_as_contradiction(
    premise: str, hypothesis: str
) -> None:
    assert NLIEvaluator().relation(premise, hypothesis) == "neutral"


def test_nli_detects_negation_written_as_a_contraction() -> None:
    """`_normalize` destroyed apostrophes, so "doesn't" read as non-negated."""
    nli = NLIEvaluator()
    assert nli._has_negation("The court doesn't apply Miranda.")
    assert nli._has_negation("The court won't extend the rule.")
    assert (
        nli.relation(
            "The court applies Miranda to custodial interrogation.",
            "The court doesn't apply Miranda to custodial interrogation.",
        )
        == "contradiction"
    )


def test_nli_handles_morphological_antonyms() -> None:
    """"unconstitutional" carries no negation *word*, so polarity looks equal.

    The overlap-based entailment shortcut therefore called a flat contradiction
    an entailment. The antonym check has to run before that shortcut — and it
    has to be word-boundary aware, or two sentences that both say
    "unconstitutional" would match the "constitutional"/"unconstitutional" pair
    by substring and be called a contradiction.
    """
    nli = NLIEvaluator()
    assert (
        nli.relation(
            "A warrantless phone search incident to arrest is constitutional.",
            "A warrantless phone search incident to arrest is unconstitutional.",
        )
        == "contradiction"
    )
    assert (
        nli.relation(
            "The statute is unconstitutional under the Commerce Clause.",
            "The statute is unconstitutional under the Due Process Clause.",
        )
        != "contradiction"
    )


def test_nli_uses_the_model_when_one_is_configured() -> None:
    """The fourth model must actually be consulted, not silently bypassed."""
    calls: list[list[Any]] = []

    class _RecordingNLI:
        name = "nemotron-mini:4b"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            calls.append(messages)
            return '{"label": "contradiction"}'

    nli = NLIEvaluator(client=_RecordingNLI())
    # A pair the heuristic would call neutral, so only the model can explain it.
    label, source = nli.relate(
        "The tribunal retains jurisdiction.", "The weather is pleasant today."
    )
    assert label == "contradiction"
    assert source == "model"
    assert calls, "the configured NLI model was never called"


@pytest.mark.parametrize(
    "bad_response",
    [
        "contradiction",  # bare label, not JSON
        '{"label": "sort of contradictory"}',  # out-of-vocabulary label
        '{"relation": "contradiction"}',  # wrong key
        "not json at all",
        '{"label": "contradiction"',  # truncated
    ],
)
def test_nli_falls_back_to_heuristic_and_records_the_downgrade(bad_response: str) -> None:
    """A silent downgrade to the heuristic must be visible in the output."""

    class _BadNLI:
        name = "nemotron-mini:4b"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            return bad_response

    nli = NLIEvaluator(client=_BadNLI())
    label, source = nli.relate(
        "The statute requires scienter.", "The statute does not require scienter."
    )
    assert label == "contradiction"  # the heuristic still answers
    assert source == "heuristic"  # ...and says it was the one who answered


def test_nli_source_is_recorded_on_the_crux() -> None:
    thesis = Position(
        side="thesis",
        model="hermes3",
        family="hermes",
        propositions=[
            CitationSlot(proposition="The rule applies.", weight=Weight.CONTROLLING)
        ],
    )
    antithesis = Position(
        side="antithesis",
        model="llama3.1",
        family="llama",
        propositions=[
            CitationSlot(proposition="The rule does not apply.", weight=Weight.CONTROLLING)
        ],
    )
    cruxes = CruxExtractor().extract(thesis, antithesis)
    assert cruxes
    assert cruxes[0].nli_source == "heuristic"


def test_nli_client_reaches_the_evaluator() -> None:
    """Constructing an NLI client is not enough; it has to be read."""
    nli_client = _FakeLLM("nemotron-mini:4b", '{"label": "neutral"}')
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json("T.", "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json("A.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
        nli_client=nli_client,
    )
    assert chat.crux_extractor.nli.client is nli_client


def test_family_guard_covers_the_nli_model() -> None:
    """The NLI model adjudicates the debate; it may not share a debater's family."""
    with pytest.raises(FamilyCollision) as exc:
        DialecticChat(
            thesis_client=_FakeLLM("hermes3:8b", "{}"),
            antithesis_client=_FakeLLM("llama3.1:8b", "{}"),
            synthesis_client=_FakeLLM("gemma3:4b", "{}"),
            nli_client=_FakeLLM("llama3.2:3b", "{}"),
        )
    assert "nli" in str(exc.value)
    assert "llama" in str(exc.value)


def test_crux_extractor_finds_contradictions_only_for_outcome_bearing_slots() -> None:
    thesis = Position(
        side="thesis",
        model="gpt",
        family="gpt",
        propositions=[
            CitationSlot(
                proposition="The rule applies to public officials.",
                weight=Weight.CONTROLLING,
            ),
            CitationSlot(
                proposition="A background fact about the statute.",
                weight=Weight.SUPPORTING,
            ),
        ],
    )
    antithesis = Position(
        side="antithesis",
        model="llama",
        family="llama",
        propositions=[
            CitationSlot(
                proposition="The rule does not apply to public officials.",
                weight=Weight.CONTROLLING,
            )
        ],
    )
    extractor = CruxExtractor()
    cruxes = extractor.extract(thesis, antithesis)
    assert len(cruxes) == 1
    assert cruxes[0].negates is True
    assert cruxes[0].thesis_prop.proposition == "The rule applies to public officials."


def test_crux_extraction_is_not_gated_by_self_declared_weight() -> None:
    """A modest model self-assigning `supporting` must not disable the feature.

    Observed live (REMEDIATION 8e): 1 thesis proposition, 2 antithesis
    propositions, 0 cruxes, because the thesis called its own argument
    `supporting`. A contradiction between two `supporting` propositions is still
    a contradiction; it is just not authority-resolvable.
    """
    thesis = Position(
        side="thesis",
        model="hermes3",
        family="hermes",
        propositions=[
            CitationSlot(
                proposition="The rule applies to public officials.",
                weight=Weight.SUPPORTING,
            )
        ],
    )
    antithesis = Position(
        side="antithesis",
        model="llama3.1",
        family="llama",
        propositions=[
            CitationSlot(
                proposition="The rule does not apply to public officials.",
                weight=Weight.SUPPORTING,
            )
        ],
    )
    cruxes = CruxExtractor().extract(thesis, antithesis)
    assert len(cruxes) == 1
    assert cruxes[0].negates is True
    # Extracted, but honestly labelled: neither side carries authority weight.
    assert cruxes[0].outcome_bearing is False
    assert cruxes[0].partition == "open"
    assert cruxes[0].winner == "none"


def test_weight_parsing_is_case_insensitive_and_notes_coercions() -> None:
    """A model emitting "Controlling" was silently stored as `supporting`."""
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3", _slot_json("The rule applies here.", "hint", "Controlling")
        ),
        antithesis_client=_FakeLLM(
            "llama3.1", _slot_json("The rule does not apply here.", "hint", "CONTROLLING")
        ),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Does the rule apply?")
    assert turn.thesis.propositions[0].weight == Weight.CONTROLLING
    assert turn.antithesis.propositions[0].weight == Weight.CONTROLLING


def test_unrecognised_weight_is_coerced_but_recorded_in_the_note() -> None:
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3", _slot_json("The rule applies.", "hint", "extremely important")
        ),
        antithesis_client=_FakeLLM(
            "llama3.1", _slot_json("The rule does not apply.", "hint", Weight.CONTROLLING)
        ),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Does the rule apply?")
    slot = turn.thesis.propositions[0]
    assert slot.weight == Weight.SUPPORTING
    assert "extremely important" in slot.note
    assert "coerced" in slot.note


def test_citation_slot_rejects_a_weight_outside_the_enum() -> None:
    """`class Weight(str)` typed the field as bare `str`, so anything validated."""
    with pytest.raises(ValidationError):
        CitationSlot(proposition="P.", weight="not-a-weight")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CitationSlot(proposition="P.", status="halfway-done")  # type: ignore[arg-type]


def test_empty_crux_table_states_the_reason() -> None:
    """An empty table with no explanation is indistinguishable from a broken extractor."""
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3", _slot_json("The contract was signed in March.", "hint", Weight.CONTROLLING)
        ),
        antithesis_client=_FakeLLM(
            "llama3.1", _slot_json("Venue lies in the Northern District.", "hint", Weight.CONTROLLING)
        ),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Is the claim viable?")
    assert not turn.cruxes
    assert turn.crux_note
    assert turn.crux_note in copy_crux_table(turn)


def test_precedence_rule_weight_gates_authority_classification() -> None:
    rule = PrecedenceRule()
    # Controlling unverified outranks persuasive verified.
    controlling = CitationSlot(
        proposition="Controlling reading.",
        weight=Weight.CONTROLLING,
        status=SlotStatus.PENDING,
    )
    persuasive_verified = CitationSlot(
        proposition="Persuasive reading.",
        weight=Weight.PERSUASIVE,
        status=SlotStatus.VERIFIED,
    )
    crux = Crux(
        thesis_prop=controlling,
        antithesis_prop=persuasive_verified,
        negates=True,
    )
    partition, winner = rule.classify(crux)
    assert partition == "resolvable by authority"
    assert winner == "thesis"


def test_canary_c6_persuasive_verified_does_not_outrank_controlling_unverified() -> None:
    rule = PrecedenceRule()
    thesis_slot = CitationSlot(
        proposition="Persuasive side.",
        weight=Weight.PERSUASIVE,
        status=SlotStatus.VERIFIED,
    )
    antithesis_slot = CitationSlot(
        proposition="Controlling side.",
        weight=Weight.CONTROLLING,
        status=SlotStatus.PENDING,
    )
    crux = Crux(
        thesis_prop=thesis_slot,
        antithesis_prop=antithesis_slot,
        negates=True,
    )
    partition, winner = rule.classify(crux)
    # It may be authority-resolvable, but never for the lower-weight persuasive side.
    assert not (partition == "resolvable by authority" and winner == "thesis")
    assert winner != "thesis"


# --------------------------------------------------------------------------- #
# Stage 4 — Copy parity / unsupported markers
# --------------------------------------------------------------------------- #
def _sample_turn_with_unfilled_slot() -> Any:
    return DialecticChat(
        thesis_client=_FakeLLM("gpt", _slot_json("Thesis point.", "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM(
            "llama", _slot_json("Antithesis point.", "hint", Weight.CONTROLLING)
        ),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
    ).chat("Question?")


def test_copy_exchange_preserves_unsupported_marker() -> None:
    turn = _sample_turn_with_unfilled_slot()
    text = copy_exchange(turn)
    assert "[UNSUPPORTED" in text


def test_copy_position_preserves_unsupported_marker() -> None:
    turn = _sample_turn_with_unfilled_slot()
    text = copy_position(turn, "thesis")
    assert "[UNSUPPORTED" in text


def test_copy_crux_table_preserves_unsupported_marker() -> None:
    slot = CitationSlot(proposition="Unfilled.", weight=Weight.PERSUASIVE)
    crux = Crux(
        thesis_prop=slot,
        antithesis_prop=CitationSlot(proposition="Opposing.", weight=Weight.CONTROLLING),
        negates=True,
        partition="open",
        winner="none",
    )
    turn = DialecticTurn(
        question="Q",
        thesis=Position(side="thesis", model="gpt", family="gpt"),
        antithesis=Position(side="antithesis", model="llama", family="llama"),
        cruxes=[crux],
    )
    text = copy_crux_table(turn)
    assert "[UNSUPPORTED" in text


def test_canary_c4_serializer_preserves_unsupported_marker() -> None:
    turn = _sample_turn_with_unfilled_slot()
    assert "[UNSUPPORTED" in copy_exchange(turn)
    assert "[UNSUPPORTED" in copy_position(turn, "thesis")
    assert "[UNSUPPORTED" in copy_position(turn, "antithesis")


# --------------------------------------------------------------------------- #
# End-to-end dialectic turn
# --------------------------------------------------------------------------- #
def test_citation_in_court_hint_is_rejected_like_one_in_the_proposition() -> None:
    """A model blocked from citing in `proposition` must not route it via `court_hint`.

    Observed live: hermes3:8b emitted a clean proposition but put
    "New York v. Harris" in court_hint, which passed an earlier version of the
    scan that only read `proposition`.
    """

    leaky = _slot_json(
        "Incident-to-arrest searches are an exception to the warrant requirement.",
        "New York v. Harris precedent",
        Weight.CONTROLLING,
    )
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", leaky),
        antithesis_client=_FakeLLM("llama", _slot_json("Antithesis.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
    )
    turn = chat.chat("Was the search lawful?")
    assert (
        turn.thesis.propositions[0].proposition
        == "(generation failed or contained a citation string)"
    )


def test_regeneration_varies_decoding_so_retries_are_not_identical() -> None:
    """Retrying at temperature 0 with a fixed seed reproduces the same rejected text."""

    seen: list[dict] = []

    class _RecordingLLM:
        name = "hermes3"

        def chat(self, messages, config=None):  # noqa: ANN001, ARG002
            seen.append(config)
            return _slot_json("Terry v. Ohio controls.", "hint", Weight.CONTROLLING)

    chat = DialecticChat(
        thesis_client=_RecordingLLM(),
        antithesis_client=_FakeLLM("llama", _slot_json("Antithesis.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
    )
    chat.chat("Was the stop lawful?")

    assert len(seen) == 3, "all regeneration attempts should be spent"
    seeds = [c["seed"] for c in seen]
    assert len(set(seeds)) == 3, f"retries reused the same seed: {seeds}"
    assert seen[0]["temperature"] == 0.0
    assert all(c["temperature"] > 0 for c in seen[1:]), "retries must re-roll, not re-run"


def test_each_side_is_told_which_way_to_argue() -> None:
    """Both sides generated blind from one question frequently agreed.

    Observed live: for "does the exclusionary rule apply to good-faith
    reliance", thesis and antithesis both returned "the exclusionary rule should
    not apply". The NLI pass correctly reported entailment, and the exchange
    produced zero cruxes — the sides simply were not opposed.
    """
    seen: dict[str, str] = {}

    class _RecordingSide:
        def __init__(self, name: str, side: str, response: str) -> None:
            self.name = name
            self._side = side
            self._response = response

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            seen[self._side] = messages[0]["content"]
            return self._response

    chat = DialecticChat(
        thesis_client=_RecordingSide(
            "hermes3", "thesis", _slot_json("T.", "hint", Weight.CONTROLLING)
        ),
        antithesis_client=_RecordingSide(
            "llama3.1", "antithesis", _slot_json("A.", "hint", Weight.CONTROLLING)
        ),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    chat.chat("Does the rule apply?")

    assert "AFFIRMATIVE" in seen["thesis"]
    assert "NEGATIVE" in seen["antithesis"]
    assert "AFFIRMATIVE" not in seen["antithesis"]
    # The weight field is about authority strength, not the arguer's stance.
    assert "not your stance" in seen["thesis"]


def test_antithesis_is_shown_the_thesis_to_contradict() -> None:
    captured: list[str] = []

    class _RecordingAntithesis:
        name = "llama3.1"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            captured.append(messages[1]["content"])
            return _slot_json("The rule does not apply.", "hint", Weight.CONTROLLING)

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3", _slot_json("The rule applies to public officials.", "h", Weight.CONTROLLING)
        ),
        antithesis_client=_RecordingAntithesis(),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    chat.chat("Does the rule apply?")

    assert captured
    user = captured[0]
    assert "Does the rule apply?" in user
    assert "The rule applies to public officials." in user
    # It is shown the thesis in order to build an independent counter-theory,
    # not to negate it: asking for the negation produced mirrors.
    assert "Build your OWN theory" in user
    assert "NOT restatements" in user
    assert "DIFFERENT doctrine" in user


def test_failed_thesis_is_not_fed_back_as_an_argument() -> None:
    """The failure placeholder must never reach the antithesis as a claim."""
    captured: list[str] = []

    class _RecordingAntithesis:
        name = "llama3.1"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            captured.append(messages[1]["content"])
            return _slot_json("Antithesis.", "hint", Weight.CONTROLLING)

    chat = DialecticChat(
        # Thesis emits a citation every time, so it exhausts its budget.
        thesis_client=_FakeLLM(
            "hermes3", _slot_json("The stop was justified under Terry v. Ohio.", "h", Weight.CONTROLLING)
        ),
        antithesis_client=_RecordingAntithesis(),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Was the stop lawful?")

    assert (
        turn.thesis.propositions[0].proposition
        == "(generation failed or contained a citation string)"
    )
    user = captured[0]
    assert "generation failed" not in user
    assert "Build your OWN theory" not in user


# --------------------------------------------------------------------------- #
# Independence guard — the antithesis must argue, not mirror
# --------------------------------------------------------------------------- #

# Mirrors observed live after the stance fix: the antithesis took a thesis
# sentence and inserted "not".
_REAL_MIRRORS = [
    (
        "The Supreme Court has not recognized that modern cell phones are "
        "effectively miniature digital safes that require heightened privacy protection.",
        "The Supreme Court has recognized that modern cell phones are effectively "
        "miniature digital safes that retain highly personal information, and thus "
        "require heightened privacy protection.",
    ),
    (
        "A cell phone does not fall within the scope of the Fourth Amendment's "
        "protections as it contains private information analogous to papers.",
        "The Fourth Amendment protects people from unreasonable searches and seizures "
        "of their persons, houses, papers, and effects. A cell phone falls within this "
        "scope as it contains private information analogous to papers.",
    ),
    (
        "Allowing warrantless searches of cell phones incident to arrest would not "
        "create a significant risk of abuse by law enforcement.",
        "Allowing warrantless searches of cell phones incident to arrest would create "
        "a significant risk of abuse by law enforcement.",
    ),
    (
        "The Supreme Court has not abrogated the physical presence rule, and states "
        "may not require remote sellers to collect sales tax without a clear physical "
        "connection.",
        "The physical presence rule has been abrogated by subsequent Supreme Court "
        "precedent allowing states to require remote sellers to collect sales tax.",
    ),
]

# Genuine counter-theories: incompatible with the thesis, but reasoning from a
# different doctrine rather than flipping its sign.
_INDEPENDENT_THEORIES = [
    (
        "The search-incident-to-arrest exception rests on officer safety and evidence "
        "preservation, neither of which is served by examining stored data after the "
        "device is secured.",
        "The Supreme Court has recognized that modern cell phones are effectively "
        "miniature digital safes that retain highly personal information, and thus "
        "require heightened privacy protection.",
    ),
    (
        "Retailers lacking in-state operations cannot reasonably ascertain thousands of "
        "local tax jurisdictions, so the compliance burden itself violates the Commerce "
        "Clause.",
        "The physical presence rule has been abrogated by subsequent Supreme Court "
        "precedent allowing states to require remote sellers to collect sales tax.",
    ),
    (
        "Congress has occupied this field through express preemption, so the state rule "
        "is void regardless of any nexus analysis.",
        "Many courts have found that remote sellers can have sufficient nexus with a "
        "state through economic activity alone, without physical presence.",
    ),
]


@pytest.mark.parametrize(("candidate", "opposing"), _REAL_MIRRORS)
def test_independence_guard_catches_polarity_flip_mirrors(
    candidate: str, opposing: str
) -> None:
    assert IndependenceGuard().measure(candidate, opposing) is not None


@pytest.mark.parametrize(("candidate", "opposing"), _INDEPENDENT_THEORIES)
def test_independence_guard_allows_a_real_counter_theory(
    candidate: str, opposing: str
) -> None:
    assert IndependenceGuard().measure(candidate, opposing) is None


def test_independence_guard_scan_raises_and_names_every_mirror() -> None:
    guard = IndependenceGuard()
    opposing = [m[1] for m in _REAL_MIRRORS[:2]]
    candidates = [m[0] for m in _REAL_MIRRORS[:2]] + [
        "Congress has occupied this field through express preemption."
    ]
    with pytest.raises(MirrorDetected) as exc:
        guard.scan(candidates, opposing)
    assert len(exc.value.mirrors) == 2, "the independent proposition was flagged too"
    assert "merely negate" in str(exc.value)


def test_independence_guard_is_inert_without_an_opposing_side() -> None:
    IndependenceGuard().scan(["Anything at all, negated or not."], [])


def test_independence_guard_needs_both_shared_words_and_flipped_polarity() -> None:
    guard = IndependenceGuard()
    thesis = (
        "The Supreme Court has recognized that modern cell phones require heightened "
        "privacy protection under the Fourth Amendment."
    )
    # Same vocabulary, same polarity: an agreeing restatement, not a mirror.
    agreeing = (
        "The Supreme Court has recognized that modern cell phones require heightened "
        "privacy protection under the Fourth Amendment doctrine."
    )
    assert guard.measure(agreeing, thesis) is None
    # Flipped polarity, different vocabulary: a real disagreement.
    different = "Officer safety does not extend to data already secured in an evidence locker."
    assert guard.measure(different, thesis) is None


def test_independence_guard_ignores_propositions_too_short_to_judge() -> None:
    """A terse claim shares nearly all its vocabulary with its own negation."""
    guard = IndependenceGuard()
    assert guard.measure("No warrant is required.", "A warrant is required.") is None


def test_antithesis_that_mirrors_is_regenerated_with_feedback() -> None:
    """Prompting alone did not stop mirroring, so the guard enforces it."""
    prompts: list[str] = []

    class _MirrorThenArgue:
        name = "llama3.1"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            prompts.append(messages[1]["content"])
            if len(prompts) == 1:
                return _slot_json(
                    "The Supreme Court has not recognized that modern cell phones "
                    "require heightened privacy protection under the Fourth Amendment.",
                    "hint",
                    Weight.CONTROLLING,
                )
            return _slot_json(
                "The search-incident-to-arrest exception rests on officer safety and "
                "evidence preservation, neither of which is served by examining stored "
                "data after the device is secured.",
                "hint",
                Weight.CONTROLLING,
            )

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3",
            _slot_json(
                "The Supreme Court has recognized that modern cell phones require "
                "heightened privacy protection under the Fourth Amendment.",
                "hint",
                Weight.CONTROLLING,
            ),
        ),
        antithesis_client=_MirrorThenArgue(),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Was the phone search lawful?")

    assert len(prompts) == 2, "the mirrored draft was not rejected"
    # The re-roll names the offending proposition rather than just re-rolling.
    assert "REJECTED" in prompts[1]
    assert "merely negated" in prompts[1]
    assert "The Supreme Court has not recognized" in prompts[1]
    # The accepted answer is the independent one.
    assert "officer safety" in turn.antithesis.propositions[0].proposition
    assert turn.regenerated >= 1


def test_persistent_mirroring_degrades_visibly_rather_than_failing_closed() -> None:
    """Independence is a quality property, not a safety one.

    A mirrored antithesis is worth more than no antithesis, so the least-bad
    draft is kept — with the concession flagged on the slot, because a reader
    must know the antithesis conceded the thesis's framing.
    """
    thesis_text = (
        "The Supreme Court has recognized that modern cell phones require heightened "
        "privacy protection under the Fourth Amendment."
    )
    mirror_text = (
        "The Supreme Court has not recognized that modern cell phones require "
        "heightened privacy protection under the Fourth Amendment."
    )
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json(thesis_text, "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json(mirror_text, "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Was the phone search lawful?")

    slot = turn.antithesis.propositions[0]
    # Kept, not discarded.
    assert slot.proposition == mirror_text
    assert "independence guard" in slot.note
    assert "merely negates" in slot.note
    # And the flag survives into the copy payloads.
    assert "independence guard" in copy_exchange(turn)


def test_thesis_is_not_subject_to_the_independence_guard() -> None:
    """The thesis has nothing to mirror; it is generated first."""
    thesis_text = "The rule applies to public officials under settled doctrine."
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json(thesis_text, "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM(
            "llama3.1",
            _slot_json(
                "Sovereign immunity bars the claim before any merits question arises.",
                "hint",
                Weight.CONTROLLING,
            ),
        ),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
    )
    turn = chat.chat("Does the rule apply?")
    assert turn.thesis.propositions[0].proposition == thesis_text
    assert "independence guard" not in turn.thesis.propositions[0].note


def test_synthesis_receives_the_crux_table_and_no_duplicated_thesis() -> None:
    """The synthesis prompt says "name the decisive crux", so it must see one.

    Cruxes used to be extracted *after* the synthesis was generated, and the
    prompt pasted the thesis into the antithesis slot as well, so the model saw
    the thesis three times and the antithesis once.
    """
    captured: list[list[Any]] = []

    class _RecordingSynthesis:
        name = "gemma3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            captured.append(messages)
            return "The dispute turns on whether the rule reaches public officials."

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3",
            _slot_json("The rule applies to public officials.", "hint", Weight.CONTROLLING),
        ),
        antithesis_client=_FakeLLM(
            "llama3.1",
            _slot_json("The rule does not apply to public officials.", "hint", Weight.CONTROLLING),
        ),
        synthesis_client=_RecordingSynthesis(),
    )
    turn = chat.chat("Does the rule reach public officials?")
    assert turn.cruxes, "precondition: this exchange produces a crux"

    assert captured, "the synthesis model was never called"
    messages = captured[0]
    system = messages[0]["content"]
    user = messages[1]["content"]

    # The derived crux table reaches the prompt.
    assert "CRUX TABLE" in user
    assert "resolvable by authority" in user or "open" in user

    # The system prompt is not also pasted into the user message.
    assert system not in user

    # The thesis is not rendered a second time into the antithesis slot. The old
    # `model_copy(update={"antithesis": turn.thesis})` produced a block labelled
    # ANTITHESIS but carrying the thesis's model name and text; that signature
    # must be absent, and the antithesis section must hold the real antithesis.
    assert "ANTITHESIS (hermes3)" not in user
    antithesis_section = user.split("ANTITHESIS:", 1)[1]
    assert "The rule does not apply to public officials." in antithesis_section
    assert '"side":"antithesis"' in antithesis_section


def test_synthesis_retries_on_citation_instead_of_failing_once() -> None:
    """Synthesis got a single shot; positions got `max_regenerations`."""
    seen: list[Any] = []

    class _LeakyThenCleanSynthesis:
        name = "gemma3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            seen.append(config)
            if len(seen) == 1:
                return "This follows from Roe v. Wade."
            return "The dispute turns on the scope of the rule."

    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json("T.", "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json("A.", "hint", Weight.CONTROLLING)),
        synthesis_client=_LeakyThenCleanSynthesis(),
    )
    turn = chat.chat("Q?")

    assert len(seen) == 2, "the rejected synthesis was not retried"
    assert turn.synthesis == "The dispute turns on the scope of the rule."
    # Re-rolled, not re-run.
    assert seen[0]["temperature"] == 0.0
    assert seen[1]["temperature"] > 0
    assert seen[0]["seed"] != seen[1]["seed"]
    assert turn.regenerated == 1


def test_synthesis_failure_records_the_reason() -> None:
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json("T.", "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json("A.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma3", "This follows from Roe v. Wade."),
    )
    turn = chat.chat("Q?")
    assert "citation" in turn.synthesis
    assert "Roe v. Wade" in turn.synthesis, "the offending hit must be recorded"


def test_end_to_end_turn_rejects_citation_in_proposition() -> None:
    bad_json = _slot_json(
        "The stop was justified under Terry v. Ohio.", "hint", Weight.CONTROLLING
    )
    chat = DialecticChat(
        thesis_client=_FakeLLM("gpt", bad_json),
        antithesis_client=_FakeLLM("llama", _slot_json("Antithesis.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
    )
    turn = chat.chat("Was the stop lawful?")
    # Regeneration limit exhausted; the thesis position carries the failure marker.
    assert turn.thesis.propositions[0].proposition == "(generation failed or contained a citation string)"


def test_citation_in_normalized_cite_is_rejected_like_one_in_the_proposition() -> None:
    """`normalized_cite` is an output of verification, never an input from a model.

    The parser used to read the field straight off the model's JSON, so a live
    reporter cite placed there reached the user unscanned — the same defect class
    as the `court_hint` leak (REMEDIATION 8b), in a third field.
    """

    leaky = _slot_json(
        "Officers must obtain a warrant before searching a phone.",
        "digital privacy precedent",
        Weight.CONTROLLING,
        normalized_cite="573 U.S. 373",
    )
    # The control: the same string is a citation when scanned directly.
    assert CitationChannel().find_hits("573 U.S. 373")

    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", leaky),
        antithesis_client=_FakeLLM("llama", _slot_json("Antithesis.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
    )
    turn = chat.chat("Was the phone search lawful?")

    assert (
        turn.thesis.propositions[0].proposition
        == "(generation failed or contained a citation string)"
    )
    assert all(not s.normalized_cite for s in turn.thesis.propositions)


def test_spec_compliant_exchange_spends_no_verification_calls() -> None:
    """With no retriever configured, a clean exchange verifies nothing — explicitly.

    This is the honest current behaviour of the pipeline: the debaters emit
    propositions and plain-English `court_hint`s, never citations, so there is
    no candidate for CourtListener to look up unless a retrieval stage supplies
    one. Every slot must therefore stay unverified and *say so*, rather than
    resting in a `pending` state that reads as "in progress".

    The end-to-end test that legitimately spends verification calls is
    `test_retrieval_stage_feeds_courtlistener_verification`, which configures a
    retriever.
    """

    http = _FakeHTTP(status_code=200, payload=[])
    client = CourtListenerClient(token="token", http=http)

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "gpt", _slot_json("The stop was justified.", "stop frisk precedent", Weight.CONTROLLING)
        ),
        antithesis_client=_FakeLLM(
            "llama",
            _slot_json("The stop was not justified.", "stop frisk precedent", Weight.CONTROLLING),
        ),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
        courtlistener=client,
    )
    turn = chat.chat("Was the stop lawful?")

    assert turn.calls_spent == 0
    assert not http.calls, "no retrieval stage means nothing to look up"
    slots = [*turn.thesis.propositions, *turn.antithesis.propositions]
    assert slots
    assert all(s.status == SlotStatus.PENDING for s in slots)
    assert all(not s.normalized_cite for s in slots)
    # `pending` alone reads as "in progress"; the reason must be stated.
    assert all("no retrieval stage configured" in s.note for s in slots)
    assert "no retrieval stage configured" in copy_exchange(turn)
    # Nothing unverified may ever render clean.
    assert "[UNSUPPORTED" in copy_exchange(turn)


def test_retrieval_stage_feeds_courtlistener_verification() -> None:
    """The path that legitimately spends verification calls.

    Retrieval supplies the candidate cites that the debaters are forbidden to
    emit; CourtListener then confirms them. This is the only route from a
    spec-compliant exchange to a `verified` slot.
    """
    payload = [
        {
            "citation": "392 U.S. 1",
            "normalized_citations": ["392 U.S. 1"],
            "status": 200,
            "clusters": [{"id": 123}],
        },
        {
            "citation": "384 U.S. 436",
            "normalized_citations": ["384 U.S. 436"],
            "status": 200,
            "clusters": [{"id": 124}],
        },
    ]
    http = _FakeHTTP(status_code=200, payload=payload)
    client = CourtListenerClient(token="token", http=http)
    retriever = StubCiteRetriever(
        {
            "stop and frisk": ["392 U.S. 1"],
            "custodial interrogation": ["384 U.S. 436"],
        }
    )

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "gpt",
            _slot_json("The stop was justified.", "stop and frisk precedent", Weight.CONTROLLING),
        ),
        antithesis_client=_FakeLLM(
            "llama",
            _slot_json(
                "The stop was not justified.",
                "custodial interrogation precedent",
                Weight.CONTROLLING,
            ),
        ),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
        courtlistener=client,
        retriever=retriever,
    )
    turn = chat.chat("Was the stop lawful?")

    assert turn.calls_spent > 0
    assert http.calls, "verification never reached the network"
    thesis_slot = turn.thesis.propositions[0]
    antithesis_slot = turn.antithesis.propositions[0]
    assert thesis_slot.status == SlotStatus.VERIFIED
    assert antithesis_slot.status == SlotStatus.VERIFIED
    assert thesis_slot.normalized_cite == "392 U.S. 1"
    assert thesis_slot.cluster_id == "123"
    assert turn.cruxes


def test_retrieved_candidate_is_never_clean_until_verified() -> None:
    """A `proposed` slot is a candidate, not authority: it keeps its marker."""
    retriever = StubCiteRetriever({"stop and frisk": ["392 U.S. 1"]})
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "gpt",
            _slot_json("The stop was justified.", "stop and frisk precedent", Weight.CONTROLLING),
        ),
        antithesis_client=_FakeLLM(
            "llama", _slot_json("The stop was not justified.", "hint", Weight.CONTROLLING)
        ),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
        retriever=retriever,
        # No CourtListener client: nothing can promote the candidate.
    )
    turn = chat.chat("Was the stop lawful?")

    slot = turn.thesis.propositions[0]
    assert slot.status == SlotStatus.PROPOSED
    assert slot.normalized_cite == "392 U.S. 1"
    text = copy_exchange(turn)
    assert "[UNSUPPORTED" in text
    assert "proposed, unverified" in text


def test_retrieval_miss_leaves_the_slot_pending_and_says_so() -> None:
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "gpt", _slot_json("A novel proposition.", "no such authority", Weight.CONTROLLING)
        ),
        antithesis_client=_FakeLLM("llama", _slot_json("Antithesis.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma", "Synthesis."),
        retriever=StubCiteRetriever({"stop and frisk": ["392 U.S. 1"]}),
    )
    turn = chat.chat("Anything?")
    slot = turn.thesis.propositions[0]
    assert slot.status == SlotStatus.PENDING
    assert not slot.normalized_cite
    assert "no candidate" in slot.note


def test_offline_retrievers_satisfy_the_protocol() -> None:
    assert isinstance(StubCiteRetriever(), CiteRetriever)
    assert isinstance(NullCiteRetriever(), CiteRetriever)
    assert NullCiteRetriever().propose("any hint", "any proposition") == []


def test_stub_retriever_prefers_the_more_specific_hint() -> None:
    retriever = StubCiteRetriever(
        {"privacy": ["1 A.2d 1"], "digital privacy in the home": ["2 A.2d 2"]}
    )
    assert retriever.propose("digital privacy in the home", "P.") == ["2 A.2d 2"]
    assert retriever.propose("privacy generally", "P.") == ["1 A.2d 1"]
    assert retriever.propose("unrelated", "P.") == []


def test_non_operative_status_survives_verification_and_reaches_the_synthesis() -> None:
    """`verify_position` rewrites `slot.note` on every branch.

    Annotating during retrieval meant the status was overwritten before the
    synthesis — the only role running after retrieval — could act on it, so the
    temporal gate failed on a response the module had no way to get right.
    """
    captured: list[str] = []

    class _RecordingSynthesis:
        name = "gemma3"

        def chat(self, messages: list[Any], config: Any | None = None) -> str:
            captured.append(messages[1]["content"])
            return "The framework is no longer operative."

    class _AnnotatingRetriever:
        def propose(self, court_hint: str, proposition: str) -> list[str]:
            return ["90 Fed. Reg. 4544"]

        def annotate(self, cite: str) -> str:
            return "NOT CURRENTLY OPERATIVE (rescinded): repealed May 2025"

    http = _FakeHTTP(
        status_code=200,
        payload=[{
            "citation": "90 Fed. Reg. 4544",
            "normalized_citations": ["90 Fed. Reg. 4544"],
            "status": 200,
            "clusters": [{"id": 1}],
        }],
    )
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json("The rule controls.", "hint", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json("Sovereign immunity bars it.", "hint", Weight.CONTROLLING)),
        synthesis_client=_RecordingSynthesis(),
        courtlistener=CourtListenerClient(token="t", http=http),
        retriever=_AnnotatingRetriever(),
    )
    turn = chat.chat("Does the rule control?")

    slot = turn.thesis.propositions[0]
    # Verified by CourtListener, yet the status survived the note rewrite.
    assert slot.status == SlotStatus.VERIFIED
    assert "NOT CURRENTLY OPERATIVE" in slot.note
    # And the synthesis was actually shown it.
    assert "NOT CURRENTLY OPERATIVE" in captured[0]
    assert "NOT CURRENTLY OPERATIVE" in copy_exchange(turn)


def test_annotation_is_skipped_when_the_retriever_cannot_supply_one() -> None:
    """Most retrievers have no notion of good-law status and must still work."""
    chat = DialecticChat(
        thesis_client=_FakeLLM("hermes3", _slot_json("T.", "stop and frisk", Weight.CONTROLLING)),
        antithesis_client=_FakeLLM("llama3.1", _slot_json("A.", "hint", Weight.CONTROLLING)),
        synthesis_client=_FakeLLM("gemma3", "Synthesis."),
        retriever=StubCiteRetriever({"stop and frisk": ["392 U.S. 1"]}),
    )
    turn = chat.chat("Q?")
    assert turn.thesis.propositions[0].normalized_cite == "392 U.S. 1"
    assert "NOT CURRENTLY OPERATIVE" not in turn.thesis.propositions[0].note
