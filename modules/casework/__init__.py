"""Shared infrastructure for the two search modules.

`schema` (feature axes + verdict vocabulary), `corpus` (coded real cases and how
far to trust each coding), `search` (pymoo NSGA-II and pyribs MAP-Elites
wrappers, archives keyed by evaluator hash), `graph_adapter` (emission into the
argument graph).

The property that makes both modules safe to search: their fitness functions are
executed, not judged. See `schema` for why that is the load-bearing constraint.
"""

from __future__ import annotations

from .corpus import CaseCorpus, CodedCase, Coding, load_cases
from .graph_adapter import Emission, emit_precedents, propositions_for_portfolio
from .report import Report, divergence_report, holdout_notice, portfolio_report
from .schema import (
    Axis,
    AxisKind,
    EvaluatorStamp,
    FeatureSchema,
    Outcome,
    Scrutiny,
    Verdict,
    load_schema,
)
from .search import ArchiveMismatch, Elite, EliteArchive

__all__ = [
    "ArchiveMismatch",
    "Axis",
    "AxisKind",
    "CaseCorpus",
    "CodedCase",
    "Coding",
    "Elite",
    "Emission",
    "EliteArchive",
    "EvaluatorStamp",
    "FeatureSchema",
    "Outcome",
    "Scrutiny",
    "Report",
    "Verdict",
    "divergence_report",
    "emit_precedents",
    "holdout_notice",
    "load_cases",
    "portfolio_report",
    "propositions_for_portfolio",
    "load_schema",
]
