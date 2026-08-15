"""Citation portfolio selection: which of the verified authorities to cite.

Constrained subset selection with genuinely competing objectives — footnote
economy against binding weight against recency against concentration. No weight
vector represents an author's real preference between those, so the module
returns the Pareto front and a few named strategies, and the author's pick is
recorded as a `[stated]` editorial decision.

Hard invariant: the candidate pool is the verified-authority set. Portfolio
chooses among citations the grounding gate already accepted and can never
introduce one it did not.
"""

from __future__ import annotations

from .model import (
    Authority,
    CitatorFlag,
    PortfolioProblem,
    Precedential,
    Proposition,
    UnverifiedAuthority,
)
from .nsga import Portfolio, Strategy, named_strategies, select
from .objectives import PortfolioScores, score_selection, thin_coverage

__all__ = [
    "Authority",
    "CitatorFlag",
    "Portfolio",
    "PortfolioProblem",
    "PortfolioScores",
    "Precedential",
    "Proposition",
    "Strategy",
    "UnverifiedAuthority",
    "named_strategies",
    "score_selection",
    "select",
    "thin_coverage",
]
