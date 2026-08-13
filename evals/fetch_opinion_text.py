"""Replace the corpus's headnote labels with real opinion text.

REMEDIATION §15 found the bottleneck: `openweights.jsonl` stores headnote-style
topic labels — 68 passages, median 12 words — and a label cannot *support* a
proposition, only share a subject with it. Every support check over it degenerates
into topic matching, which is why three different scorers behaved alike and why
neither a scorer swap nor a prompt change moved the survival rate.

This fetches each case's opinion text from CourtListener and chunks it into
quotable passages.

**The cluster id comes from the record's own URL**, not from a citation lookup.
Every case record already carries `courtlistener.com/opinion/<cluster>/…` from
when it was verified, so this costs one request per case instead of two, and the
125/day budget is never the reason a corpus rebuild fails.

**The original file is never overwritten.** The curated headnotes are the record
of what each case was admitted to the corpus *for*, and they move to a
`headnotes` field rather than being discarded — a rebuilt corpus that lost them
would lose the curation, and there would be no way to tell a fetch that returned
nothing from a case that was never about anything in particular.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from legal_research.config import get_settings  # noqa: E402
from legal_research.ingest.base import chunk_passages  # noqa: E402

CLUSTER_RE = re.compile(r"/opinion/(\d+)/")
API = "https://www.courtlistener.com/api/rest/v4/opinions/"

#: Passages shorter than this assert too little to support anything. The old
#: corpus's median was 12 words, which is what this exists to stop recurring.
MIN_WORDS = 25

#: Cap per record. A whole opinion would swamp the scorer and make every claim
#: match something; the point is quotable passages, not a full-text index.
MAX_PASSAGES = 12

#: Seconds between requests. CourtListener allows 5/minute for an authenticated
#: user -- the same budget `RateBudget` enforces for verification -- so anything
#: faster earns 429s. A first run at 1.5s filled 5 records of 34 and was refused
#: for the rest, which is what set this number.
POLITE_DELAY = 13.0

#: A 429 is a pacing problem, not a dead record, so it is retried rather than
#: recorded as a failure that quietly shrinks the corpus.
RETRIES = 3


@dataclass
class FetchOutcome:
    record_id: str
    cluster: str = ""
    passages: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.passages)


def cluster_id(record: dict[str, Any]) -> str:
    match = CLUSTER_RE.search(str(record.get("url", "")))
    return match.group(1) if match else ""


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?i)</p\s*>|<br\s*/?>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#8217;", "'").replace("&lt;", "<")
    return text


def opinion_text(payload: dict[str, Any]) -> str:
    """Prefer plain text; fall back through the HTML variants CourtListener has."""
    for key in ("plain_text", "html_with_citations", "html", "html_lawbox", "xml_harvard"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value if key == "plain_text" else _strip_html(value)
    return ""


def select(text: str, headnotes: list[str], limit: int = MAX_PASSAGES) -> list[str]:
    """Chunk an opinion and keep the passages closest to its headnotes.

    The headnotes record what this case was admitted to the corpus *for*, so
    ranking against them keeps the curation instead of taking whatever appears
    first — an opinion's opening paragraphs are usually procedural history.
    """
    candidates = [
        p for p in chunk_passages(text) if len(p.split()) >= MIN_WORDS
    ]
    if not candidates or not headnotes:
        return candidates[:limit]

    wanted: set[str] = set()
    for note in headnotes:
        wanted |= {w for w in re.findall(r"[a-z]{4,}", note.lower())}

    def overlap(passage: str) -> int:
        return len({w for w in re.findall(r"[a-z]{4,}", passage.lower())} & wanted)

    ranked = sorted(candidates, key=lambda p: (-overlap(p), candidates.index(p)))
    kept = [p for p in ranked if overlap(p) > 0][:limit]
    # A case whose opinion shares nothing with its headnotes is worth keeping
    # something from rather than nothing: the headnotes may simply be terse.
    return kept or candidates[:limit]


def fetch(record: dict[str, Any], client: httpx.Client, token: str) -> FetchOutcome:
    cluster = cluster_id(record)
    outcome = FetchOutcome(record_id=str(record.get("id", "")), cluster=cluster)
    if not cluster:
        outcome.error = "no cluster id in the record url"
        return outcome

    response = None
    for attempt in range(RETRIES):
        try:
            response = client.get(
                API,
                params={"cluster": cluster},
                headers={"Authorization": f"Token {token}"},
                timeout=60.0,
            )
        except Exception as exc:  # noqa: BLE001 - a network failure is data
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome

        if response.status_code != 429:
            break
        # Honour Retry-After when the server sends one; otherwise back off.
        wait = float(response.headers.get("retry-after") or POLITE_DELAY * (attempt + 2))
        time.sleep(wait)

    if response is None or response.status_code != 200:
        code = response.status_code if response is not None else "no response"
        outcome.error = f"HTTP {code}"
        return outcome

    results = response.json().get("results", [])
    if not results:
        outcome.error = "cluster returned no opinions"
        return outcome

    text = ""
    for payload in results:
        text = opinion_text(payload)
        if text:
            break
    if not text:
        outcome.error = "opinion carried no text in any field"
        return outcome

    outcome.passages = select(text, list(record.get("passages", [])))
    if not outcome.passages:
        outcome.error = "no passage survived the length filter"
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_opinion_text",
        description="Rebuild a corpus with real opinion passages instead of headnotes.",
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/openweights.jsonl"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus/openweights_fulltext.jsonl"),
        help="written fresh; the input corpus is never modified",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N case records")
    parser.add_argument(
        "--delay",
        type=float,
        default=POLITE_DELAY,
        help=f"seconds between requests (default {POLITE_DELAY}, i.e. 5/minute)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip records the output file already filled. The daily budget is "
            "125 requests, so re-fetching what is already there can be the "
            "reason a rebuild never finishes."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = str(get_settings().courtlistener_token or "").strip()
    if not token:
        print("no CourtListener token configured", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    cases = [r for r in records if r.get("type") == "case"]

    already: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if "headnotes" in prior:
                already[prior["id"]] = prior
        cases = [r for r in cases if r["id"] not in already]
        print(f"resuming: {len(already)} record(s) already filled, {len(cases)} to go")

    targets = {r["id"] for r in cases}

    outcomes: list[FetchOutcome] = []
    with httpx.Client(follow_redirects=True) as client:
        for index, record in enumerate(cases):
            outcome = fetch(record, client, token)
            outcomes.append(outcome)
            state = f"{len(outcome.passages)} passage(s)" if outcome.ok else outcome.error
            print(f"  [{index + 1}/{len(cases)}] {record['id']}: {state}")
            if index + 1 < len(cases):
                time.sleep(args.delay)

    by_id = {o.record_id: o for o in outcomes if o.ok}
    rebuilt = []
    for record in records:
        fresh = dict(record)
        prior = already.get(record.get("id", ""))
        if prior is not None:
            rebuilt.append(prior)
            continue
        fetched = by_id.get(record.get("id", ""))
        if fetched:
            # Headnotes are the curation record, not scratch. Keeping them means
            # a fetch that returned nothing is distinguishable from a case that
            # was never about anything in particular.
            fresh["headnotes"] = list(record.get("passages", []))
            fresh["passages"] = fetched.passages
        rebuilt.append(fresh)

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rebuilt) + "\n",
        encoding="utf-8",
    )

    filled = len(by_id)
    words = [len(p.split()) for o in by_id.values() for p in o.passages]
    print(f"\n{filled}/{len(targets)} case record(s) fetched this run")
    if words:
        words.sort()
        print(
            f"passages: {len(words)}, median {words[len(words) // 2]} words, "
            f"min {words[0]}, max {words[-1]}"
        )
    for outcome in outcomes:
        if not outcome.ok:
            print(f"  unfilled: {outcome.record_id} — {outcome.error}")
    print(f"\nwrote {args.out} ({args.corpus} unchanged)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
