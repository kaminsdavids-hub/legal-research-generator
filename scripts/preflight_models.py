#!/usr/bin/env python3
"""Validate a dialectic model lineup before a run spends anything on it.

Two checks, in this order:

1. **Family independence.** Thesis, antithesis, synthesis and the NLI pass must
   come from four distinct base families, and the eval judge — a fifth role —
   from a fifth. Same-family instances have correlated errors and produce
   agreement dressed as debate; a judge that is a sibling of a debater grades
   its own family's habits. This check makes no calls and needs no key, which is
   why it runs first and why ``--no-probe`` exists: a lineup that cannot be
   valid should not cost a token to reject.

2. **Reachability.** One tiny completion per role, reporting latency. A cold
   model load on the Spark costs ~40s, so a probe that answers slowly is
   information, not a failure.

Exit codes: ``0`` valid, ``2`` invalid lineup, ``3`` a role was unreachable.

    python scripts/preflight_models.py --no-probe
    python scripts/preflight_models.py --thesis claude-sonnet-4-5 \\
        --thesis-base-url https://api.anthropic.com/v1 \\
        --thesis-api-key-env LRG_DIALECTIC_THESIS_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.dialectic.roles import RoleSpec, detect_family  # noqa: E402
from modules.dialectic.service import is_local_endpoint  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
#: Roles in the order they are reported. ``judge`` is the eval's fifth model and
#: is included because the same discipline applies to it.
ROLES = ("thesis", "antithesis", "synthesis", "nli", "judge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="shared endpoint for roles without an override")
    parser.add_argument("--thesis", default="saul:7b-instruct-v1")
    parser.add_argument("--antithesis", default="llama3.1:8b")
    parser.add_argument("--synthesis", default="gemma3:4b")
    parser.add_argument("--nli", default="nemotron-3-nano:4b")
    parser.add_argument("--judge", default="hermes3:8b", help="eval judge; pass '' to skip")
    for role in ROLES:
        parser.add_argument(f"--{role}-base-url", default="", help=f"endpoint override for {role}")
        parser.add_argument(
            f"--{role}-api-key-env",
            default="",
            help=f"environment variable holding {role}'s key; required for a remote endpoint",
        )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="check the lineup only: no calls, no key needed, no cost",
    )
    return parser


def _specs(args: argparse.Namespace) -> list[RoleSpec]:
    specs: list[RoleSpec] = []
    for role in ROLES:
        model = str(getattr(args, role) or "").strip()
        if not model:
            continue
        override = str(getattr(args, f"{role}_base_url") or "").strip()
        specs.append(
            RoleSpec(
                role=role,
                model=model,
                family=detect_family(model),
                base_url=override or args.base_url,
            )
        )
    return specs


def check_families(specs: list[RoleSpec]) -> list[str]:
    """Complaints about the lineup; empty means it is valid.

    ``unknown`` is reported per role rather than collapsed: two roles both
    detected as ``unknown`` may or may not be siblings, and the check cannot
    tell. Saying so is more useful than either passing them or failing them.
    """

    complaints: list[str] = []
    seen: dict[str, str] = {}
    for spec in specs:
        if spec.family == "unknown":
            complaints.append(
                f"{spec.role}: family of {spec.model!r} is unrecognised, so independence "
                "from the other roles cannot be established — add it to KNOWN_FAMILIES "
                "in modules/dialectic/roles.py"
            )
            continue
        if spec.family in seen:
            complaints.append(
                f"{spec.role} ({spec.model}) shares family {spec.family!r} with "
                f"{seen[spec.family]} — correlated errors, not a debate"
            )
            continue
        seen[spec.family] = spec.role
    return complaints


def _probe(spec: RoleSpec, api_key: str, timeout: float) -> tuple[bool, str]:
    from legal_research.llm.base import ChatMessage, DecodingPolicy
    from legal_research.llm.openai_compat import OpenAICompatLLM

    client = OpenAICompatLLM(
        name=spec.model, base_url=spec.base_url, model=spec.model, api_key=api_key, timeout=timeout
    )
    started = time.monotonic()
    try:
        reply = client.chat(
            [ChatMessage("user", "Reply with the single word: ready.")],
            DecodingPolicy.COLD.config,
        )
    except Exception as exc:  # noqa: BLE001 - the point of the probe is to report this
        # Not truncated to a status line: this probe exists so a paid run is not
        # started against a role that cannot answer, and "400 Bad Request" with
        # the provider's explanation cut off tells the operator nothing they can
        # fix. See _raise_for_status in llm/openai_compat.py.
        detail = f"{type(exc).__name__}: {str(exc)[:400]}"
        if "400" in str(exc):
            # A 400 means the credential was accepted and the *payload* was
            # rejected -- the decoding parameters this repo sends to Ollama are
            # not all accepted by every hosted provider. Rather than report that
            # and wait for a human to guess which field, narrow it here: each
            # retry is a handful of tokens, and the alternative is another round
            # trip before a paid run can start.
            culprit = _narrow_payload(client, timeout)
            if culprit:
                detail = f"{detail}\n               → {culprit}"
        return False, detail
    elapsed = time.monotonic() - started
    if not str(reply).strip():
        # An empty content channel is what reasoning-only models return through
        # the OpenAI-compatible endpoint, and every turn would then fail to
        # parse. Reachable is not the same as usable.
        return False, f"answered in {elapsed:.1f}s with an empty content channel"
    return True, f"{elapsed:.1f}s"


#: Tried in order, each dropping one more field from the COLD policy. The first
#: that succeeds names what the provider would not accept.
_NARROWING = (
    ("seed", {"seed": None}),
    ("top_p", {"seed": None, "top_p": 1.0}),
    ("temperature", {"seed": None, "top_p": 1.0, "temperature": 1.0}),
)


def _narrow_payload(client: Any, timeout: float) -> str:
    """Which decoding field the provider rejects, or "" if none of them."""

    from dataclasses import replace

    from legal_research.llm.base import ChatMessage, DecodingPolicy

    message = [ChatMessage("user", "Reply with the single word: ready.")]
    dropped: list[str] = []
    for field, overrides in _NARROWING:
        dropped.append(field)
        try:
            client.chat(message, replace(DecodingPolicy.COLD.config, **overrides))
        except Exception:  # noqa: BLE001 - still refused; keep narrowing
            continue
        return (
            f"accepted once {', '.join(dropped)} was dropped — this provider "
            f"rejects it in the COLD policy; set the role's decoding accordingly"
        )
    # Deliberately names no cause. An earlier version concluded "the model name
    # or endpoint path is wrong", and the actual 400 was an empty credit
    # balance -- a confident wrong diagnosis printed directly beneath the
    # provider's correct one. What this function can establish is that dropping
    # decoding fields does not help; the reason is in the message above it.
    return ("still refused with every decoding field dropped: no decoding parameter "
            "is to blame — read the provider's message above")


def main() -> int:
    args = build_parser().parse_args()
    specs = _specs(args)

    print("lineup:")
    for spec in specs:
        where = "local" if is_local_endpoint(spec.base_url) else f"REMOTE {spec.base_url}"
        print(f"  {spec.role:11s} {spec.model:28s} family={spec.family:9s} {where}")

    complaints = check_families(specs)
    if complaints:
        print("\nFATAL: lineup is not independent:", file=sys.stderr)
        for complaint in complaints:
            print(f"  - {complaint}", file=sys.stderr)
        return 2
    print(f"\nfamilies distinct across {len(specs)} role(s): OK")

    if args.no_probe:
        print("--no-probe: no calls made, nothing spent")
        return 0

    failures = 0
    print("\nprobing:")
    for spec in specs:
        key_env = str(getattr(args, f"{spec.role}_api_key_env") or "").strip()
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not api_key and not is_local_endpoint(spec.base_url):
            print(
                f"  {spec.role:11s} FATAL: remote endpoint with no key "
                f"({key_env or f'no --{spec.role}-api-key-env given'})",
                file=sys.stderr,
            )
            failures += 1
            continue
        ok, detail = _probe(spec, api_key or "ollama", args.timeout)
        print(f"  {spec.role:11s} {'OK  ' if ok else 'FAIL'} {detail}")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} role(s) unusable", file=sys.stderr)
        return 3
    print("\nall roles reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
