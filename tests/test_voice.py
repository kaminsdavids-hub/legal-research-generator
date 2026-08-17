"""Anti-"AI voice" lint and anti-template check (spec §5, §9.2)."""

from __future__ import annotations

from legal_research.voice.anti_ai_voice import Severity, lint_ai_voice, rewrite_for_voice
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


def test_rewrite_preserves_paragraph_breaks() -> None:
    """The blanket whitespace collapse flattened every section it touched: a
    2,647-word manuscript reached the renderer as six 400-word paragraphs, and
    every paragraph-indexed stage downstream was really operating on sections."""

    text = (
        "Moreover, the first paragraph makes its point. It does so at length, "
        "with some variety in how the sentences run.\n\n"
        "Furthermore, the second paragraph answers it. In conclusion, the two "
        "together state the argument the section was drafted to carry."
    )
    out = rewrite_for_voice(text)

    assert out.count("\n\n") == 1
    assert len(_split(out)) == 2
    assert "second paragraph answers it" in out


def test_cadence_fix_does_not_merge_across_a_paragraph_break() -> None:
    metronome = " ".join(f"The court ruled on point number {i} today." for i in range(1, 6))
    out = rewrite_for_voice(f"{metronome}\n\n{metronome}")

    assert out.count("\n\n") == 1
    assert all(part.strip() for part in _split(out))


def _split(text: str) -> list[str]:
    import re

    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]
