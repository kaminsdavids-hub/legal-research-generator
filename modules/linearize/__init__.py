"""Manuscript ordering: a reading order for the argument graph, and its sections.

The renderer emits a cross-reference exactly when it crosses a dependency edge
into a different section, so the order decides how many references a reader
meets and how far each one reaches back. Choosing it well is a
precedence-constrained minimum linear arrangement.

Two solvers behind one shape (``problem`` -> ``order``): simulated annealing for
graphs up to ~300 nodes, a biased random-key GA above that. Segmentation is not
searched at all -- once the order is fixed it is an exact O(n²k) dynamic
program, and using search there would trade determinism for nothing.
"""

from __future__ import annotations

from .brkga import BRKGAConfig, evolve
from .problem import (
    LinearizeProblem,
    PrecedenceViolation,
    Weights,
    build_problem,
)
from .report import (
    OrderingReport,
    ablation,
    build_report,
    compare_baselines,
    reading_plan,
)
from .sa import SAConfig, anneal
from .segment import Segmentation, SegmentConfig, segment, sweep_k

__all__ = [
    "BRKGAConfig",
    "LinearizeProblem",
    "OrderingReport",
    "PrecedenceViolation",
    "SAConfig",
    "SegmentConfig",
    "Segmentation",
    "Weights",
    "ablation",
    "anneal",
    "build_problem",
    "build_report",
    "compare_baselines",
    "reading_plan",
    "evolve",
    "segment",
    "sweep_k",
]
