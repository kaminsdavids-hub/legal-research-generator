"""legal research generator — local multi-agent studio for grounded legal papers.

Everything in this package runs locally. No cloud LLMs. The default configuration
uses a deterministic mock LLM and an in-memory retriever so the full pipeline and
test suite run with no GPU and no network access.
"""

from __future__ import annotations

__version__ = "0.1.0"

DISCLAIMER = (
    "AI-assisted draft — NOT legal advice. The author is responsible for "
    "verifying all authorities and analysis before use or submission. Citations "
    "are machine-checked against the provided corpus but must be confirmed by the "
    "author."
)

__all__ = ["__version__", "DISCLAIMER"]
