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
    if "gpt" in m:
        return "gpt"
    return "unknown"
