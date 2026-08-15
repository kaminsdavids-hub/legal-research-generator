"""Adversarial hypothetical search: where do formalised readings disagree?

Finds the fact patterns a referee would raise, by executing frozen rule
predicates over a feature space and looking for the points where they split.
The archive is the output, not a winner — coverage of the space of hard cases is
the deliverable.

Its most valuable finding is not a hypothetical at all: when a high-disagreement
pattern matches a *coded real case*, the module emits a `MishandledPrecedent` —
your framework classifies a decided case inconsistently.
"""

from __future__ import annotations

from .genotype import FactVector, FeasibilityMask, decode, encode
from .mapelites import DivergenceConfig, run_map_elites, run_nsga2
from .objectives import Scores, brittleness, disagreement, realism_prior, score
from .precedent import Collision, Hypothetical, MishandledPrecedent, match_precedents
from .render import RenderError, render_hypothetical, template_prose
from .rules import RuleSet, load_rules

__all__ = [
    "Collision",
    "DivergenceConfig",
    "FactVector",
    "FeasibilityMask",
    "Hypothetical",
    "MishandledPrecedent",
    "RenderError",
    "RuleSet",
    "Scores",
    "brittleness",
    "decode",
    "disagreement",
    "encode",
    "load_rules",
    "match_precedents",
    "realism_prior",
    "render_hypothetical",
    "run_map_elites",
    "run_nsga2",
    "score",
    "template_prose",
]
