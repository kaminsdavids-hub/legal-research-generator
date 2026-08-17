"""The mechanism gate: what fires, what must not, and what may never be softened."""

from __future__ import annotations

import pytest

from legal_research.blackboard import Blackboard
from legal_research.citations.report import build_verification_report
from legal_research.mechanism import GateReport, run_gate
from legal_research.mechanism.blackbox import Severity, scan_text
from legal_research.mechanism.gate import (
    MechanismCritic,
    combine,
    lexical_findings,
    repair_paragraph,
)
from legal_research.models import MechanismFinding


class _ScriptedClient:
    """Returns canned replies in order; records what it was asked."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    def chat(self, messages, config=None) -> str:  # noqa: ANN001 - test double
        self.calls.append(messages[-1].content)
        return self._replies.pop(0) if self._replies else ""


# --------------------------------------------------------------------------- #
# The adaptation that separates this from the patent detector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "The EU AI Act regulates machine-learning systems placed on the Union market.",
        "Courts have struggled with artificial intelligence as a subject of regulation.",
        "AI-driven regulation has become a scholarly preoccupation.",
        "Scholars disagree about whether neural networks are patentable subject matter.",
        # The vocabulary is present and the noun is not the head the verb
        # attaches to; an intervening qualifier is exactly what carries meaning.
        "The model of cooperative federalism determines the allocation of authority.",
    ],
)
def test_topical_mention_does_not_fire(text: str) -> None:
    """A paper about AI says 'AI'. Mention is not the defect; position is."""

    assert scan_text(text) == []


@pytest.mark.parametrize(
    ("text", "term"),
    [
        ("The algorithm determines which applicants advance.", "The algorithm determines"),
        ("Benefits were denied by an algorithm the agency never described.", "by an algorithm"),
        ("The disparity is attributable to the model used by the county.", "attributable to the model"),
        ("The algorithm is biased against older applicants.", "The algorithm is biased"),
        ("The system intelligently prioritizes high-risk cases.", "intelligently prioritizes"),
        ("The AI-driven denial was upheld on review.", "AI-driven denial"),
    ],
)
def test_mechanism_position_fires_as_black_box(text: str, term: str) -> None:
    findings = scan_text(text)
    assert [f.severity for f in findings] == [Severity.BLACK_BOX]
    assert findings[0].term == term


def test_adjacent_structure_downgrades_to_under_specified() -> None:
    text = (
        "A gradient-boosted classifier over 42 application features scores each file, "
        "and the algorithm flags any score above 0.65."
    )
    findings = scan_text(text)
    assert [f.severity for f in findings] == [Severity.UNDER_SPECIFIED]


def test_structure_in_the_next_sentence_counts() -> None:
    """The fix an author would actually write must not still be a BLACK_BOX."""

    text = (
        "The model determines eligibility. It is a logistic regression over six "
        "application variables, thresholded at 0.4."
    )
    assert [f.severity for f in scan_text(text)] == [Severity.UNDER_SPECIFIED]


def test_structure_two_paragraphs_later_does_not_rescue() -> None:
    prose = (
        "The algorithm determines eligibility.\n\n"
        "Unrelated doctrinal discussion follows here at some length.\n\n"
        "The county's tool is a logistic regression over six variables."
    )
    findings = lexical_findings("sec-001", "Analysis", prose)
    assert [f.severity for f in findings] == [Severity.BLACK_BOX.value]
    assert findings[0].paragraph_index == 0


def test_overlapping_patterns_report_one_finding() -> None:
    findings = scan_text("Applicants were screened using an algorithm that ranks them.")
    assert len(findings) == 1


# --------------------------------------------------------------------------- #
# Precedence: the semantic layer may add, never soften
# --------------------------------------------------------------------------- #
def _finding(severity: str, source: str, index: int = 0) -> MechanismFinding:
    return MechanismFinding(
        section_id="sec-001",
        section_title="Analysis",
        paragraph_index=index,
        severity=severity,
        term="the algorithm determines",
        reason="reason",
        source=source,
    )


def test_semantic_cannot_demote_a_lexical_black_box() -> None:
    merged = combine(
        [_finding("BLACK_BOX", "lexical")],
        [_finding("UNDER_SPECIFIED", "semantic")],
    )
    assert {f.severity for f in merged} == {"BLACK_BOX"}
    assert {f.source for f in merged} == {"lexical", "semantic"}


def test_semantic_finding_on_a_clean_paragraph_is_kept() -> None:
    merged = combine([], [_finding("UNDER_SPECIFIED", "semantic", index=2)])
    assert [(f.source, f.severity) for f in merged] == [("semantic", "UNDER_SPECIFIED")]


def test_the_critic_asks_the_server_to_constrain_decoding() -> None:
    """Prompting for JSON is not the same as getting it."""

    class _Recorder:
        def __init__(self) -> None:
            self.config = None

        def chat(self, messages, config=None):  # noqa: ANN001, ANN202 - test double
            self.config = config
            return '{"findings": []}'

    recorder = _Recorder()
    MechanismCritic(recorder).findings("sec-001", "Analysis", ["text"])
    assert recorder.config.response_format == "json_object"


def test_a_malformed_reply_is_retried_once_with_the_error_quoted_back() -> None:
    client = _ScriptedClient(
        '{"findings": [{"term": "the tool decides", "reason": "he said "yes" here"}]}',
        '{"findings": [{"paragraph_index": 0, "severity": "BLACK_BOX", '
        '"term": "the tool decides", "reason": "no operation"}]}',
    )
    findings = MechanismCritic(client).findings("sec-001", "Analysis", ["the tool decides fast"])

    assert [f.term for f in findings] == ["the tool decides"]
    assert "could not be parsed" in client.calls[-1]


def test_a_reply_that_breaks_partway_salvages_the_findings_before_the_break() -> None:
    """Three good objects then a broken one should cost one finding, not four."""

    good = ('{"paragraph_index": 0, "severity": "BLACK_BOX", "term": "%s", '
            '"reason": "no operation"}')
    raw = (
        '{"findings": ['
        + ",".join(good % t for t in ("the tool decides", "by the model", "the system scores"))
        + ', {"term": "broken", "reason": "unescaped "quote" here"}]}'
    )
    findings = MechanismCritic(_ScriptedClient(raw, raw)).findings(
        "sec-001",
        "Analysis",
        ["the tool decides and by the model the system scores each file"],
    )
    assert [f.term for f in findings] == ["the tool decides", "by the model", "the system scores"]


def test_salvage_cannot_turn_an_unparseable_reply_into_no_findings() -> None:
    """A critic that said nothing usable is unavailable, not clean."""

    broken = "{ this is not json at all, no objects here }"
    with pytest.raises(ValueError):
        MechanismCritic(_ScriptedClient(broken, broken)).findings("sec-001", "A", ["text"])


def test_critic_output_that_is_not_json_raises() -> None:
    """An unusable critic must not read as 'nothing found' (REMEDIATION §21)."""

    critic = MechanismCritic(_ScriptedClient("Looks fine to me!"))
    with pytest.raises(ValueError, match="no JSON"):
        critic.findings("sec-001", "Analysis", ["The algorithm determines eligibility."])


def test_critic_findings_are_parsed_and_clamped() -> None:
    critic = MechanismCritic(
        _ScriptedClient('{"findings": [{"paragraph_index": 9, "severity": "BLACK_BOX", '
                        '"term": "the tool", "reason": "no operation stated"}]}')
    )
    findings = critic.findings("sec-001", "Analysis", ["the tool decides, in one paragraph"])
    assert len(findings) == 1
    assert findings[0].paragraph_index == 0  # clamped rather than misattributed
    assert findings[0].source == "semantic"
    assert "the tool" in findings[0].excerpt


def test_a_critic_finding_that_is_not_in_the_text_is_dropped() -> None:
    """Observed on a real run: a 4B critic returned the prompt's own examples,
    and reasoned about landlord-tenant law in a paper on export control."""

    critic = MechanismCritic(
        _ScriptedClient(
            '{"findings": ['
            '{"paragraph_index": 0, "severity": "BLACK_BOX", "term": "the algorithm '
            'determines", "reason": "echoed from the prompt"},'
            '{"paragraph_index": 0, "severity": "BLACK_BOX", "term": "weights are '
            'numerical", "reason": "actually in the text"}]}'
        )
    )
    findings = critic.findings(
        "sec-001", "Analysis", ["Model weights are numerical values with no author."]
    )
    assert [f.term for f in findings] == ["weights are numerical"]


def test_a_quote_from_the_wrong_paragraph_is_repointed_not_dropped() -> None:
    critic = MechanismCritic(
        _ScriptedClient(
            '{"findings": [{"paragraph_index": 0, "severity": "BLACK_BOX", '
            '"term": "the screening tool", "reason": "no operation"}]}'
        )
    )
    findings = critic.findings(
        "sec-001", "Analysis", ["First paragraph, unrelated.", "The screening tool ranks files."]
    )
    assert [(f.paragraph_index, f.term) for f in findings] == [(1, "the screening tool")]


def test_the_critic_prompt_carries_no_example_phrases() -> None:
    """The examples were what the weak critic returned as findings."""

    from legal_research.mechanism.gate import CRITIC_SYSTEM

    for echoed in ("the algorithm determines", "denied by an AI system", "the model is biased"):
        assert echoed not in CRITIC_SYSTEM


# --------------------------------------------------------------------------- #
# Repair: accepted only if a re-scan says it worked
# --------------------------------------------------------------------------- #
def test_repair_is_rejected_when_the_rewrite_fixes_nothing() -> None:
    paragraph = "The algorithm determines which applicants advance."
    client = _ScriptedClient("The algorithm determines which applicants advance to interview.")
    text, improved = repair_paragraph(client, paragraph, [])
    assert (text, improved) == (paragraph, False)


def test_repair_is_refused_when_the_rewrite_deletes_the_argument() -> None:
    """From a real run: a 450-word section came back as 130 words. Removing the
    finding by removing the passage is the author's decision, not the gate's."""

    paragraph = (
        "Model weights are numerical values generated through machine learning "
        "algorithms, which do not possess the requisite originality for copyright. "
        "Copyright protection generally applies to original works of authorship "
        "fixed in a tangible medium, and the weights result from computation "
        "rather than authorship. Trade secret protection fails for a parallel "
        "reason: the weights are published, so no competitive advantage is "
        "maintained by keeping them secret, and the doctrine of reverse "
        "engineering would in any event permit their analysis."
    )
    client = _ScriptedClient("The weights are numerical values. Nothing follows.")

    text, improved = repair_paragraph(client, paragraph, [])
    assert (text, improved) == (paragraph, False)


def test_repair_is_refused_when_the_rewrite_invents_the_mechanism() -> None:
    """Also from a real run. Supplying the missing operation out of the model's
    weights is the one way of satisfying this gate that must never work."""

    paragraph = (
        "People may not fully understand the decisions being made by AI algorithms, "
        "and that opacity undermines confidence in the courts that rely on them."
    )
    client = _ScriptedClient(
        "The systems read case facts, applicable statutes, precedent and the "
        "training corpus; they perform statistical inference over those features; "
        "and they emit a risk score that a judge may adopt at sentencing."
    )

    text, improved = repair_paragraph(client, paragraph, [])
    assert (text, improved) == (paragraph, False)


def test_repair_is_refused_when_it_supplies_a_mechanism_that_sounds_right() -> None:
    """The tempting failure: a plausible, well-written mechanism that appears
    nowhere in the passage. Plausibility is not provenance."""

    paragraph = "The algorithm determines which applicants advance to interview."
    client = _ScriptedClient(
        "A logistic regression over six application variables scores each file, and "
        "applicants above the 0.4 cut-off advance to interview."
    )
    text, improved = repair_paragraph(client, paragraph, [])
    assert (text, improved) == (paragraph, False)


def test_repair_is_accepted_when_it_confines_the_claim() -> None:
    """What an LLM repair legitimately does: say the operation is unstated and
    narrow the assertion, introducing no structure the passage lacked."""

    paragraph = "The algorithm determines which applicants advance to interview."
    client = _ScriptedClient(
        "The county has not stated how applications are sorted, so on this record "
        "the claim is confined to the fact that some applicants advanced to "
        "interview and others did not."
    )
    text, improved = repair_paragraph(client, paragraph, [])
    assert improved is True
    assert scan_text(text) == []


def test_repair_is_accepted_when_it_moves_an_operation_the_passage_already_stated() -> None:
    paragraph = (
        "The algorithm determines which applicants advance. Elsewhere the county "
        "describes a logistic regression over six application variables."
    )
    client = _ScriptedClient(
        "A logistic regression over six application variables, which the county "
        "describes, is what sorts the applicants who advance."
    )
    text, improved = repair_paragraph(client, paragraph, [])
    assert improved is True
    assert scan_text(text) == []


def test_repair_touches_only_the_sentences_around_the_finding() -> None:
    """A long block must not hand the model licence over the whole argument."""

    untouched_head = (
        "The Ninth Circuit held that source code is expressive because a "
        "programmer reads it. That holding governs this case in every respect "
        "that matters to the parties here."
    )
    untouched_tail = (
        "The remedy therefore follows from the classification rather than from "
        "any balancing the agency now proposes."
    )
    paragraph = f"{untouched_head} The algorithm determines eligibility. {untouched_tail}"

    client = _ScriptedClient(
        "The county has not stated how applications are sorted, so the claim is "
        "confined to the fact that some applications were sorted."
    )
    text, improved = repair_paragraph(client, paragraph, [])

    assert improved is True
    assert untouched_head.split(".")[0] in text
    assert "balancing the agency now proposes" in text
    assert "The algorithm determines" not in text
    # The model was shown the sentence, not the section.
    assert "Rewrite ONLY this passage" in client.calls[0]
    assert client.calls[0].count("The Ninth Circuit held") == 1  # context copy only


# --------------------------------------------------------------------------- #
# End to end over a blackboard
# --------------------------------------------------------------------------- #
def _blackboard_with(content: str) -> Blackboard:
    bb = Blackboard(session_id="session-test")
    section = bb.add_section("Analysis")
    section.content = content
    return bb


def test_run_gate_repairs_and_preserves_citation_tokens() -> None:
    bb = _blackboard_with("The algorithm determines which applicants advance.{{cite:c-001}}")
    client = _ScriptedClient(
        "The county has not stated how applications are sorted, so on this record "
        "the claim is confined to the fact that some applicants advanced and "
        "others did not."
    )

    report = run_gate(bb, repair_client=client)

    assert report.repaired == 1
    assert report.unresolved == []
    assert "{{cite:c-001}}" in bb.outline[0].content
    assert bb.edits and bb.edits[0].author == "Mechanism Gate"


def test_unrepairable_findings_survive_and_reach_the_report() -> None:
    bb = _blackboard_with("Benefits were denied by an algorithm the agency never described.")
    report = run_gate(bb, repair_client=_ScriptedClient("Benefits were denied by an algorithm."))

    assert report.repaired == 0
    assert len(report.unresolved) == 1
    assert bb.mechanism_findings == report.findings

    rendered = build_verification_report([], [], _EmptyCorpus(), bb.mechanism_findings)
    assert "Mechanism findings" in rendered
    assert "BLACK_BOX" in rendered


def test_a_rerun_replaces_findings_rather_than_accumulating() -> None:
    bb = _blackboard_with("The algorithm determines eligibility.")
    run_gate(bb)
    run_gate(bb)
    assert len(bb.mechanism_findings) == 1


def test_black_box_findings_block_shipping_only_when_asked() -> None:
    bb = _blackboard_with("The algorithm determines eligibility.")
    run_gate(bb)
    assert bb.is_shippable() is True
    assert bb.is_shippable(block_on_mechanism=True) is False


def test_critic_failure_is_reported_not_swallowed() -> None:
    class _Broken:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202 - test double
            raise RuntimeError("endpoint down")

    bb = _blackboard_with("The algorithm determines eligibility.")
    report = run_gate(bb, critic=MechanismCritic(_Broken()), critic_fail_open=True)

    assert report.critic_ran is False
    assert "RuntimeError" in report.critic_error
    assert "critic unavailable" in report.summary()
    # The deterministic layer still ran.
    assert len(report.findings) == 1


def test_critic_failure_is_fatal_when_fail_open_is_off() -> None:
    class _Broken:
        def chat(self, messages, config=None):  # noqa: ANN001, ANN202 - test double
            raise RuntimeError("endpoint down")

    bb = _blackboard_with("The algorithm determines eligibility.")
    with pytest.raises(RuntimeError):
        run_gate(bb, critic=MechanismCritic(_Broken()), critic_fail_open=False)


def test_gate_report_summary_separates_repaired_from_unresolved() -> None:
    report = GateReport(
        findings=[_finding("BLACK_BOX", "lexical"), _finding("UNDER_SPECIFIED", "lexical", 1)],
        repaired=1,
    )
    summary = report.summary()
    assert "2 finding(s)" in summary
    assert "1 repaired" in summary
    assert "2 unresolved" in summary


class _EmptyCorpus:
    def get(self, record_id: str):  # noqa: ANN201 - minimal stand-in
        return None
