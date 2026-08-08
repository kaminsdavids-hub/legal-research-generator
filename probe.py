#!/usr/bin/env python3
"""Reproduction harness for the ``modules/dialectic/`` correctness pass.

Each probe executes the *shipped* code and prints ``CONFIRMED`` when the defect
it describes is still present, or ``not reproduced`` when it is gone.

Item 8 is inverted on purpose: it checks documented *correct* behaviour, so it
must read ``CONFIRMED`` both before and after the pass. A regression there is a
failure even though every other item flipping to ``not reproduced`` is a pass.

Run from the repository root::

    python probe.py

No network and no GPU: every model and HTTP client below is a local fake.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

sys.path.insert(0, "src")

from modules.dialectic.channel import CitationChannel  # noqa: E402
from modules.dialectic.crux import CruxExtractor, PrecedenceRule  # noqa: E402
from modules.dialectic.engine import DialecticChat  # noqa: E402
from modules.dialectic.models import (  # noqa: E402
    CitationSlot,
    Crux,
    Position,
    SlotStatus,
    Weight,
)
from modules.dialectic.nli import NLIEvaluator  # noqa: E402

# --------------------------------------------------------------------------- #
# Local fakes
# --------------------------------------------------------------------------- #


class _FakeLLM:
    """Deterministic chat client. Returns one canned string."""

    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response = response

    def chat(self, messages: list[Any], config: Any | None = None) -> str:
        return self.response


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeHTTP:
    """In-memory stand-in for httpx.Client."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, "kwargs": kwargs})
        return _FakeResponse(200, self.payload)


def _slots_json(*items: dict[str, str]) -> str:
    return json.dumps({"propositions": list(items)})


def _prop(
    proposition: str,
    court_hint: str = "hint",
    weight: str = "controlling",
    **extra: str,
) -> dict[str, str]:
    out = {"proposition": proposition, "court_hint": court_hint, "weight": weight}
    out.update(extra)
    return out


def _clean_chat(**kwargs: Any) -> DialecticChat:
    """A spec-compliant three-role chat: no citations anywhere, distinct families."""
    return DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3:8b",
            _slots_json(_prop("Warrantless entry requires exigent circumstances.")),
        ),
        antithesis_client=_FakeLLM(
            "llama3.1:8b",
            _slots_json(
                _prop("Warrantless entry does not require exigent circumstances.")
            ),
        ),
        synthesis_client=_FakeLLM("gemma3:4b", "The dispute turns on exigency."),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def probe_1_normalized_cite_bypass() -> tuple[bool, str]:
    """W1: a live reporter cite placed in `normalized_cite` passes the channel."""
    channel = CitationChannel()
    live_cite = "573 U.S. 373"

    # Sanity: the channel does flag this string when scanned directly.
    direct_hits = channel.find_hits(live_cite)
    if not direct_hits:
        return False, "channel no longer flags the control string; probe is stale"

    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3:8b",
            _slots_json(
                _prop(
                    "Officers must obtain a warrant before searching a phone.",
                    normalized_cite=live_cite,
                )
            ),
        ),
        antithesis_client=_FakeLLM(
            "llama3.1:8b", _slots_json(_prop("No warrant is required."))
        ),
        synthesis_client=_FakeLLM("gemma3:4b", "Synthesis."),
    )
    turn = chat.chat("Was the phone search lawful?")

    survivors = [
        s.normalized_cite
        for s in turn.thesis.propositions
        if s.normalized_cite and channel.find_hits(s.normalized_cite)
    ]
    if survivors:
        return True, f"model-supplied cite survived the channel: {survivors!r}"
    return False, "no model-supplied citation survived in normalized_cite"


def probe_2_no_retrieval_stage() -> tuple[bool, str]:
    """W3: nothing populates `normalized_cite`, so verification never runs."""
    payload = [
        {
            "citation": "392 U.S. 1",
            "normalized_citations": ["392 U.S. 1"],
            "status": 200,
            "clusters": [{"id": 123}],
        }
    ]

    try:
        from modules.dialectic.retrieval import StubCiteRetriever
    except ImportError as exc:
        return True, f"no retrieval stage exists ({exc})"

    from modules.dialectic.verification import CourtListenerClient

    http = _FakeHTTP(payload)
    client = CourtListenerClient(token="probe-token", http=http)  # type: ignore[arg-type]
    retriever = StubCiteRetriever({"exigent": ["392 U.S. 1"]})
    chat = _clean_chat(courtlistener=client, retriever=retriever)
    turn = chat.chat("Was the entry lawful?")

    verified = [
        s
        for s in (*turn.thesis.propositions, *turn.antithesis.propositions)
        if s.status == SlotStatus.VERIFIED
    ]
    if turn.calls_spent == 0 or not verified:
        return True, (
            f"spec-compliant exchange still cannot verify: "
            f"calls_spent={turn.calls_spent}, verified={len(verified)}"
        )
    return False, (
        f"retrieval stage carried the exchange to verification: "
        f"calls_spent={turn.calls_spent}, verified={len(verified)}"
    )


def probe_3_nli_client_never_read() -> tuple[bool, str]:
    """W4: the constructed NLI client is stored but never reaches NLIEvaluator."""
    nli_client = _FakeLLM("nemotron-mini:4b", '{"label": "neutral"}')
    chat = _clean_chat(nli_client=nli_client)

    engine_client = getattr(chat, "nli_client", None)
    evaluator_client = getattr(chat.crux_extractor.nli, "client", None)

    if engine_client is not None and evaluator_client is None:
        return True, (
            f"engine.nli_client={getattr(engine_client, 'name', engine_client)!r} "
            f"but crux_extractor.nli.client={evaluator_client!r}"
        )
    return False, (
        f"NLI client is wired through: crux_extractor.nli.client="
        f"{getattr(evaluator_client, 'name', evaluator_client)!r}"
    )


# Unrelated legal-prose pairs. Exactly one side of each carries a negation word;
# none of them is a contradiction of the other.
_UNRELATED_PAIRS = [
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


def probe_4_heuristic_flags_unrelated_pairs() -> tuple[bool, str]:
    """W5A: shared-token + one-sided negation reduces to 'any two sentences'."""
    nli = NLIEvaluator()
    bad = [
        (p, h)
        for p, h in _UNRELATED_PAIRS
        if nli.relation(p, h) == "contradiction"
    ]
    if bad:
        return True, (
            f"{len(bad)}/{len(_UNRELATED_PAIRS)} unrelated pairs flagged as "
            f"contradiction, e.g. {bad[0][0]!r} vs {bad[0][1]!r}"
        )
    return False, f"0/{len(_UNRELATED_PAIRS)} unrelated pairs flagged as contradiction"


def probe_5_contraction_negation_unreachable() -> tuple[bool, str]:
    """W5B: contractions read as non-negated, so a flat contradiction is missed.

    Stated behaviourally rather than against `_has_negation`'s arguments: the
    original code destroyed apostrophes in `_normalize` before `_has_negation`
    ever saw them, so half its alternation was unreachable. The observable
    consequence is the thing that matters — a contraction-negated proposition
    is not recognised as contradicting its affirmative twin.
    """
    nli = NLIEvaluator()
    premise = "The court applies Miranda to custodial interrogation."
    hypothesis = "The court doesn't apply Miranda to custodial interrogation."

    label = nli.relation(premise, hypothesis)
    negated = nli._has_negation(hypothesis)  # noqa: SLF001 - probing internals on purpose

    if label != "contradiction" or not negated:
        return True, (
            f"contraction negation missed: {hypothesis!r} reads as "
            f"{'negated' if negated else 'non-negated'} and the pair classifies "
            f"as {label!r}, not 'contradiction'"
        )
    return False, "contraction-negated propositions are recognised as contradictions"


def probe_6_weight_gate_disables_crux_engine() -> tuple[bool, str]:
    """W6: case-sensitive weight parsing plus the outcome-bearing filter."""
    reasons: list[str] = []

    # Defect B: "Controlling" is silently coerced to "supporting".
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3:8b",
            _slots_json(_prop("The rule applies here.", weight="Controlling")),
        ),
        antithesis_client=_FakeLLM(
            "llama3.1:8b",
            _slots_json(_prop("The rule does not apply here.", weight="Controlling")),
        ),
        synthesis_client=_FakeLLM("gemma3:4b", "Synthesis."),
    )
    turn = chat.chat("Does the rule apply?")
    got = turn.thesis.propositions[0].weight
    if str(got) != str(Weight.CONTROLLING):
        reasons.append(f'weight "Controlling" parsed as {str(got)!r} with no note')

    # Defect A: two `supporting` propositions that flatly contradict yield 0 cruxes.
    thesis = Position(
        side="thesis",
        model="hermes3:8b",
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
        model="llama3.1:8b",
        family="llama",
        propositions=[
            CitationSlot(
                proposition="The rule does not apply to public officials.",
                weight=Weight.SUPPORTING,
            )
        ],
    )
    cruxes = CruxExtractor().extract(thesis, antithesis)
    if not cruxes:
        reasons.append(
            "a flat contradiction between two `supporting` propositions "
            "produced 0 cruxes"
        )

    if reasons:
        return True, "; ".join(reasons)
    return False, "weights parse case-insensitively and non-outcome-bearing pairs extract"


def probe_7_failure_diagnostics_lost() -> tuple[bool, str]:
    """W8: a position rejected for citations returns an empty `note`."""
    chat = DialecticChat(
        thesis_client=_FakeLLM(
            "hermes3:8b",
            _slots_json(_prop("The stop was justified under Terry v. Ohio.")),
        ),
        antithesis_client=_FakeLLM(
            "llama3.1:8b", _slots_json(_prop("The stop was not justified."))
        ),
        synthesis_client=_FakeLLM("gemma3:4b", "Synthesis."),
    )
    turn = chat.chat("Was the stop lawful?")
    slot = turn.thesis.propositions[0]

    failed = slot.proposition == "(generation failed or contained a citation string)"
    if not failed:
        return False, "citation-bearing position was not rejected; probe is stale"

    if not slot.note:
        return True, (
            "position failed on CitationDetected but note is empty: "
            "indistinguishable from a parse failure"
        )
    if getattr(turn, "regenerated", 0) == 0:
        return True, f"note={slot.note!r} but DialecticTurn.regenerated is still 0"
    return False, (
        f"note={slot.note!r}, regenerated={getattr(turn, 'regenerated', None)}"
    )


def probe_8_precedence_rule_holds() -> tuple[bool, str]:
    """Control: CONFIRMED means the documented precedence rule still holds.

    Weight always outranks verification; verification only breaks ties within
    the same weight. This must read CONFIRMED before and after the pass.
    """
    rule = PrecedenceRule()

    controlling_unverified = CitationSlot(
        proposition="Controlling reading.",
        weight=Weight.CONTROLLING,
        status=SlotStatus.PENDING,
    )
    persuasive_verified = CitationSlot(
        proposition="Persuasive reading.",
        weight=Weight.PERSUASIVE,
        status=SlotStatus.VERIFIED,
    )

    # Controlling-unverified beats persuasive-verified, whichever side it sits on.
    partition, winner = rule.classify(
        Crux(
            thesis_prop=controlling_unverified,
            antithesis_prop=persuasive_verified,
            negates=True,
        )
    )
    if partition != "resolvable by authority" or winner != "thesis":
        return False, f"controlling/unverified lost to persuasive/verified: {partition}/{winner}"

    partition, winner = rule.classify(
        Crux(
            thesis_prop=persuasive_verified,
            antithesis_prop=controlling_unverified,
            negates=True,
        )
    )
    if winner == "thesis":
        return False, "persuasive/verified outranked controlling/unverified"

    # Verification breaks ties only within the same weight.
    same_weight_verified = CitationSlot(
        proposition="Same weight, verified.",
        weight=Weight.CONTROLLING,
        status=SlotStatus.VERIFIED,
    )
    partition, winner = rule.classify(
        Crux(
            thesis_prop=same_weight_verified,
            antithesis_prop=controlling_unverified,
            negates=True,
        )
    )
    if partition != "resolvable by authority" or winner != "thesis":
        return False, f"verification failed to break a same-weight tie: {partition}/{winner}"

    return True, "weight outranks verification; verification breaks same-weight ties only"


PROBES = [
    ("W1  citation bypass via normalized_cite", probe_1_normalized_cite_bypass),
    ("W3  no retrieval stage; verification unreachable", probe_2_no_retrieval_stage),
    ("W4  NLI client constructed but never read", probe_3_nli_client_never_read),
    ("W5A NLI heuristic flags unrelated pairs", probe_4_heuristic_flags_unrelated_pairs),
    ("W5B contraction negation unreachable", probe_5_contraction_negation_unreachable),
    ("W6  weight gate silently disables cruxes", probe_6_weight_gate_disables_crux_engine),
    ("W8  failure diagnostics discarded", probe_7_failure_diagnostics_lost),
    ("--  documented precedence rule holds", probe_8_precedence_rule_holds),
]


def main() -> int:
    print("modules/dialectic — defect reproduction probe")
    print("=" * 78)
    print("Items 1-7 are defects: CONFIRMED means still broken.")
    print("Item 8 is a control: CONFIRMED means correct behaviour is intact.")
    print("=" * 78)

    confirmed: list[int] = []
    for index, (title, probe) in enumerate(PROBES, start=1):
        try:
            reproduced, detail = probe()
        except Exception:  # noqa: BLE001 - a crashing probe is a probe result
            print(f"{index}. {title}\n   ERROR    {traceback.format_exc().strip().splitlines()[-1]}")
            continue
        label = "CONFIRMED" if reproduced else "not reproduced"
        if reproduced:
            confirmed.append(index)
        print(f"{index}. {title}\n   {label:<14} {detail}")

    print("=" * 78)
    defects = [i for i in confirmed if i != len(PROBES)]
    control_ok = len(PROBES) in confirmed
    print(f"defects still reproducing: {defects or 'none'}")
    print(f"control (item {len(PROBES)}) intact: {control_ok}")
    return 0 if not defects and control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
