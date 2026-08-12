"""Check a dialectic lineup before spending money on it.

A frontier run costs real money per call and the loop makes four model calls per
answer, with retries. Discovering a typo'd model name or an unreachable endpoint
after ten minutes of paid calls is avoidable, so this validates the configuration
first: distinct families, reachable endpoints, and a one-token completion from
each role.

It prints which endpoint and which key *source* each role resolved to. It never
prints a key. Reading a credential back to a terminal is how credentials end up
in scrollback and screenshots.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.dialectic.roles import detect_family  # noqa: E402

ROLES = ("thesis", "antithesis", "synthesis", "nli")


@dataclass
class RoleCheck:
    role: str
    model: str
    family: str
    endpoint: str
    key_source: str
    reachable: bool | None = None
    detail: str = ""


def resolve(settings: Any, role: str) -> RoleCheck:
    model = str(getattr(settings, f"dialectic_{role}_model", "") or "")
    override_url = str(getattr(settings, f"dialectic_{role}_base_url", "") or "").strip()
    override_key = str(getattr(settings, f"dialectic_{role}_api_key", "") or "").strip()
    return RoleCheck(
        role=role,
        model=model,
        family=detect_family(model),
        endpoint=override_url or settings.dialectic_base_url,
        key_source=(
            f"LRG_DIALECTIC_{role.upper()}_API_KEY" if override_key else "LRG_LLM_API_KEY"
        ),
    )


def families_ok(checks: list[RoleCheck]) -> list[str]:
    """Report every family collision, including 'unknown' against 'unknown'.

    Two unrecognised models collide by design: guessing that two unfamiliar
    names are different families would let correlated models debate each other.
    A frontier model reported as `unknown` needs a case added to `detect_family`,
    not a looser guard.
    """
    problems = []
    for i, a in enumerate(checks):
        for b in checks[i + 1 :]:
            if a.family == b.family:
                problems.append(
                    f"{a.role} ({a.model}) and {b.role} ({b.model}) "
                    f"share family '{a.family}'"
                )
    return problems


def probe(check: RoleCheck, settings: Any) -> RoleCheck:
    """One minimal completion, to prove the endpoint answers for this model."""
    from legal_research.llm.base import ChatMessage
    from legal_research.llm.openai_compat import OpenAICompatLLM

    key = str(getattr(settings, f"dialectic_{check.role}_api_key", "") or "").strip()
    try:
        client = OpenAICompatLLM(
            name=check.model,
            base_url=check.endpoint,
            model=check.model,
            api_key=key or settings.llm_api_key,
            timeout=30.0,
        )
        reply = client.chat([ChatMessage(role="user", content="Reply with: ok")])
    except Exception as exc:  # noqa: BLE001 - an unreachable endpoint is the finding
        check.reachable = False
        check.detail = f"{type(exc).__name__}: {str(exc)[:120]}"
        return check

    check.reachable = bool(str(reply).strip())
    check.detail = "" if check.reachable else "endpoint returned empty content"
    return check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_models",
        description="Validate a dialectic lineup before running it.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="check families and endpoints only; make no model calls",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from legal_research.config import get_settings

    settings = get_settings()
    checks = [resolve(settings, role) for role in ROLES]

    print(f"{'role':12s} {'model':28s} {'family':10s} endpoint")
    for check in checks:
        print(f"{check.role:12s} {check.model:28s} {check.family:10s} {check.endpoint}")
        print(f"{'':12s} key from {check.key_source}")

    problems = families_ok(checks)
    print()
    if problems:
        for problem in problems:
            print(f"FAIL family collision: {problem}")
        if any(c.family == "unknown" for c in checks):
            print(
                "\nA model reported as 'unknown' needs a case in "
                "modules/dialectic/roles.py::detect_family. Two unknowns collide "
                "deliberately; do not loosen the guard to get past this."
            )
        return 1
    print("families: distinct across all four roles")

    if args.no_probe:
        print("skipped endpoint probes (--no-probe)")
        return 0

    failed = 0
    for check in checks:
        probe(check, settings)
        state = "ok" if check.reachable else f"UNREACHABLE — {check.detail}"
        print(f"probe {check.role:12s} {state}")
        failed += 0 if check.reachable else 1

    if failed:
        print(f"\n{failed} role(s) unreachable; fix before running the loop")
        return 1
    print("\nlineup is ready")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
