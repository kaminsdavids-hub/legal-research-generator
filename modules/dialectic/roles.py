"""Model-family detection and role configuration for the dialectic module."""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_FAMILIES = ("gpt", "gemma", "llama", "nemotron", "hermes", "saul", "apertus", "unknown")


@dataclass(frozen=True)
class RoleSpec:
    """Configures one debate role: name, model identifier, and temperature policy."""

    role: str
    model: str
    family: str
    base_url: str


def detect_family(model: str) -> str:
    """Map a model string to its base family.

    Same-family instances have correlated errors; the dialectic module requires
    three distinct families for thesis, antithesis, and synthesis.
    """
    m = model.lower().replace("-", " ").replace("_", " ")
    # Ordered checks: more specific tokens before generic ones.
    if "saul" in m:
        return "saul"
    if "nemotron" in m:
        return "nemotron"
    if "hermes" in m:
        return "hermes"
    if "apertus" in m:
        return "apertus"
    if "gemma" in m:
        return "gemma"
    if "llama" in m:
        return "llama"
    if "gpt" in m or "o1" in m.split() or "o3" in m.split():
        return "gpt"
    # Frontier families. Without these every hosted model mapped to "unknown",
    # and because two unknowns are treated as colliding, no frontier
    # configuration could satisfy the distinctness guard at all.
    if "claude" in m or "anthropic" in m:
        return "claude"
    if "gemini" in m or "palm" in m:
        return "gemini"
    if "mistral" in m or "mixtral" in m or "magistral" in m:
        return "mistral"
    if "qwen" in m:
        return "qwen"
    if "deepseek" in m:
        return "deepseek"
    if "grok" in m:
        return "grok"
    if "command" in m or "cohere" in m:
        return "cohere"
    if "phi" in m:
        return "phi"
    # Deliberately conservative: an unrecognised model is "unknown", and two
    # unknowns collide. Guessing that two unfamiliar names are different
    # families would let correlated models debate each other, which is the one
    # thing this function exists to prevent. Add a case rather than loosening it.
    return "unknown"
