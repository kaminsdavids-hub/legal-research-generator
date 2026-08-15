"""Agent runtimes.

Both OpenHands (primary orchestrator) and NemoClaw (second cooperating runtime) run
agents through the same :class:`AgentRuntime` interface. The runtime is responsible
for the plan-first / cooperative-review wrapper around each agent step; agents
themselves stay runtime-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .base import Agent, AgentContext, AgentResult


@runtime_checkable
class AgentRuntime(Protocol):
    name: str

    def run(self, agent: Agent, ctx: AgentContext, **kwargs: Any) -> AgentResult: ...


class OpenHandsRuntime(AgentRuntime):
    """Primary orchestrator. Plan-first: records the intended step before acting."""

    name = "openhands"

    def run(self, agent: Agent, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        result = agent.act(ctx, **kwargs)
        result.runtime = self.name
        result.payload.setdefault("plan_step", f"{agent.name}: {result.summary}")
        return result


class NemoClawRuntime(AgentRuntime):
    """Second cooperating runtime. Adds a lightweight cooperative-review marker."""

    name = "nemoclaw"

    def run(self, agent: Agent, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        result = agent.act(ctx, **kwargs)
        result.runtime = self.name
        result.payload.setdefault("reviewed_by", "nemoclaw")
        return result


def build_runtime(kind: str = "openhands") -> AgentRuntime:
    if kind == "nemoclaw":
        return NemoClawRuntime()
    return OpenHandsRuntime()
