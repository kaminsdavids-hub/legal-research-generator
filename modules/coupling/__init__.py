"""How the two modules feed each other without oscillating.

Two directed edges through the argument graph: a MishandledPrecedent creates a
proposition needing authority; a ThinCoverage finding raises the search prior on
the feature region that proposition occupies. That is a cycle, and an undamped
one would spend every epoch elaborating a single corner of the space.

The damping is epoch discipline: a run reads signals only from epochs strictly
before its own, so nothing a run produces can change what that same run explores.
"""

from __future__ import annotations

from .events import EpochManager, Event, EventKind, EventLog, regions_from_priors

__all__ = ["EpochManager", "Event", "EventKind", "EventLog", "regions_from_priors"]
