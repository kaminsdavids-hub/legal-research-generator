"""Dialectic chat module: adversarial legal argument with citation discipline."""

from .channel import CitationChannel, CitationDetected
from .copy import copy_crux_table, copy_exchange, copy_position
from .crux import CruxExtractor, PrecedenceRule
from .engine import CorrelationGuardReport, DialecticChat, FamilyCollision
from .independence import IndependenceGuard, Mirror, MirrorDetected
from .models import (
    BudgetLedger,
    CitationSlot,
    Crux,
    DialecticTurn,
    Position,
    SlotStatus,
    Weight,
)
from .nli import NLIEvaluator
from .retrieval import CiteRetriever, NullCiteRetriever, StubCiteRetriever
from .roles import detect_family
from .verification import (
    BudgetExhausted,
    ContentCache,
    CourtListenerClient,
    PersistentCiteCache,
    RateBudget,
    VerificationResult,
    verify_position,
)

__all__ = [
    "BudgetExhausted",
    "BudgetLedger",
    "CitationChannel",
    "CitationDetected",
    "CitationSlot",
    "CiteRetriever",
    "ContentCache",
    "CorrelationGuardReport",
    "CourtListenerClient",
    "Crux",
    "CruxExtractor",
    "DialecticTurn",
    "DialecticChat",
    "FamilyCollision",
    "IndependenceGuard",
    "Mirror",
    "MirrorDetected",
    "NLIEvaluator",
    "NullCiteRetriever",
    "PersistentCiteCache",
    "Position",
    "PrecedenceRule",
    "RateBudget",
    "SlotStatus",
    "StubCiteRetriever",
    "VerificationResult",
    "Weight",
    "copy_crux_table",
    "copy_exchange",
    "copy_position",
    "detect_family",
    "verify_position",
]
