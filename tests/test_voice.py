"""Anti-"AI voice" lint and anti-template check (spec §5, §9.2)."""

from __future__ import annotations

from legal_research.voice.anti_ai_voice import Severity, lint_ai_voice
from legal_research.voice.anti_template import structural_overlap, template_violations

CLEAN = (
    "The rule is narrow. Courts have wrestled for decades with the precise boundary "
    "between a duty to speak and mere silence, and the answer has never been tidy. "
    "Text matters. When the statute and the equities diverge, the honest move is to "
    "say so plainly rather than pretend the conflict away. That is the wager."
)


def _kinds(text: str) -> set[str]:
    return {f.kind for f in lint_ai_voice(text).failures()}


def test_clean_prose_passes() -> None:
    report = lint_ai_voice(CLEAN)
    assert report.passed
    assert report.score == 1.0


def test_over_hedging_flagged() -> None:
    text = "Arguably this is somewhat relatively unclear. Perhaps it seems so."
    assert "over_hedging" in _kinds(text)


def test_transition_clustering_flagged() -> None:
    text = "Moreover, the rule applies. Furthermore, it extends. Additionally, it binds."
    assert "transition_clustering" in _kinds(text)


def test_empty_summary_flagged() -> None:
    text = "In conclusion, the argument holds together and everyone should agree."
    assert "empty_summary" in _kinds(text)


def test_uniform_cadence_flagged() -> None:
    text = (
        "The cat sat on mats. The dog ran to parks. A bird flew through trees. "
        "Some fish swam in streams. Many ants marched over floors."
    )
    assert "uniform_cadence" in _kinds(text)


def test_failures_have_fail_severity() -> None:
    report = lint_ai_voice("In summary, arguably it seems somewhat relatively obvious.")
    assert report.failures()
    assert all(f.severity is Severity.FAIL for f in report.failures())


def test_anti_template_flags_shared_boilerplate() -> None:
    boiler = (
        "This Article proceeds in four Parts. Part I lays out the doctrinal background. "
        "Part II identifies the problem. Part III proposes a solution and Part IV concludes."
    )
    violations = template_violations({"paper_a": boiler, "paper_b": boiler})
    assert len(violations) == 1
    assert violations[0].kind == "shared_template"


def test_anti_template_allows_distinct_papers() -> None:
    a = "The scienter requirement polices the boundary between fraud and mere negligence."
    b = "Leverage ratios constrain bank balance sheets independent of internal risk models."
    assert structural_overlap(a, b) < 0.35
    assert template_violations({"a": a, "b": b}) == []
