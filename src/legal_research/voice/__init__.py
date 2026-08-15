"""Human-like, unique, novel voice checks (spec §5).

These are explicit, testable guards:
* :mod:`anti_template` — fails if two papers share boilerplate scaffolding.
* :mod:`anti_ai_voice` — flags formulaic LLM tells and scores cadence variety.
* :mod:`novelty` — grounds the paper's original contribution in retrieved literature.
"""

from __future__ import annotations

from .anti_ai_voice import AiVoiceReport, Finding, Severity, lint_ai_voice, rewrite_for_voice
from .anti_template import structural_overlap, template_violations
from .novelty import assess_novelty

__all__ = [
    "AiVoiceReport",
    "Finding",
    "Severity",
    "lint_ai_voice",
    "rewrite_for_voice",
    "structural_overlap",
    "template_violations",
    "assess_novelty",
]
