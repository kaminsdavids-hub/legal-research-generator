"""Model-family detection and role configuration for the dialectic module."""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_FAMILIES = (
    "gpt",
    "gemma",
    "llama",
    "nemotron",
    "hermes",
    "saul",
    "apertus",
    # Hosted families. They are here for the control arm: to ask whether this
    # pipeline's limits come from the local models or from the pipeline, one
    # role is pointed at a frontier API while the rest stay on Ollama — and an
    # unrecognised family fails the independence check, so a control run could
    # not be configured without these entries.
    "claude",
    "deepseek",
    "glm",
    "qwen",
    "mistral",
    "unknown",
)


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
    if "claude" in m or "anthropic" in m:
        return "claude"
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
    # Hosted families last, deliberately. A distilled model carries the base
    # weights' failure modes, so "deepseek-r1-distill-llama-70b" is llama for
    # the purpose this check serves and must have matched above; only a model
    # named for the house alone reaches here.
    if "deepseek" in m:
        return "deepseek"
    if "glm" in m:
        return "glm"
    if "qwen" in m:
        return "qwen"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    return "unknown"
