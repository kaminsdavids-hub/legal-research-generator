#!/usr/bin/env python3
"""Resolve the eval set's case authorities against CourtListener and emit corpus records.

Citations here are *proposed* from the eval set's doctrinal anchors and then
adjudicated by CourtListener's citation-lookup endpoint. A record is written
only when the endpoint returns HTTP 200 with a cluster whose case name matches
the expected name. Anything that does not match is reported for manual
resolution rather than written.

That ordering matters: the corpus is the ground truth a citation-integrity gate
is measured against, so a cite recalled from memory must be checked by the API
before it becomes that truth, not after.

Batches citations into few requests to respect the 5/min, 50/hr, 125/day budget.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# (record id, expected case name fragment, proposed citation, court, topical passages)
AUTHORITIES: list[tuple[str, str, str, str, list[str]]] = [
    ("bernstein-9th-1999", "Bernstein", "176 F.3d 1132", "9th Cir.",
     ["Encryption source code as expression protected by the First Amendment.",
      "Licensing requirement for publishing cryptographic source code treated as a prior restraint.",
      "Scientific communication among researchers as the expressive interest at stake."]),
    ("junger-6th-2000", "Junger", "209 F.3d 481", "6th Cir.",
     ["Computer source code as protected speech notwithstanding its functional capacity.",
      "Export regulation of encryption software and the expressive/functional line."]),
    ("sorrell-2011", "Sorrell", "564 U.S. 552", "U.S.",
     ["Restrictions on the sale, disclosure, and use of data as speech regulation.",
      "Content- and speaker-based burdens on information subject to heightened scrutiny."]),
    ("brown-ema-2011", "Brown", "564 U.S. 786", "U.S.",
     ["Courts do not create new categories of unprotected speech by balancing costs and benefits.",
      "Interactive works receive full First Amendment protection."]),
    ("obrien-1968", "O'Brien", "391 U.S. 367", "U.S.",
     ["Intermediate scrutiny for regulation of conduct with an incidental burden on expression.",
      "Government interest unrelated to the suppression of free expression."]),
    ("spence-1974", "Spence", "418 U.S. 405", "U.S.",
     ["Expressive conduct test: intent to convey a particularized message likely to be understood."]),
    ("hurley-1995", "Hurley", "515 U.S. 557", "U.S.",
     ["Expression need not carry a narrow, succinctly articulable message to be protected."]),
    ("rice-paladin-1997", "Rice", "128 F.3d 233", "4th Cir.",
     ["Aiding and abetting liability for instructional material as the outer bound of protection.",
      "Mere instructions distinguished from speech integral to criminal conduct."]),
    ("feist-1991", "Feist", "499 U.S. 340", "U.S.",
     ["Selection, coordination, and arrangement as the locus of originality in a compilation."]),
    ("reed-gilbert-2015", "Reed", "576 U.S. 155", "U.S.",
     ["Content neutrality determined on the face of the regulation, not by government purpose."]),
    ("city-of-austin-2022", "Austin", "596 U.S. 61", "U.S.",
     ["On-premises/off-premises distinction as content neutral; Reed refined."]),
    ("freedman-1965", "Freedman", "380 U.S. 51", "U.S.",
     ["Procedural safeguards required of any licensing scheme touching expression.",
      "Prompt judicial determination and the burden on the censor."]),
    ("near-1931", "Near", "283 U.S. 697", "U.S.",
     ["Prior restraint doctrine; the heavy presumption against restraining publication."]),
    ("wv-epa-2022", "West Virginia", "597 U.S. 697", "U.S.",
     ["Major questions doctrine: clear congressional authorization for decisions of vast economic and political significance."]),
    ("lamont-1965", "Lamont", "381 U.S. 301", "U.S.",
     ["The right to receive information and ideas from abroad.",
      "Affirmative government obstacle to receipt of foreign material struck down."]),
    ("stanley-1969", "Stanley", "394 U.S. 557", "U.S.",
     ["Right to receive information and ideas regardless of their social worth."]),
    ("kleindienst-mandel-1972", "Mandel", "408 U.S. 753", "U.S.",
     ["Listener interests asserted against an exclusion of a foreign speaker.",
      "Facially legitimate and bona fide reason standard."]),
    ("aosi-i-2013", "Alliance for Open Society", "570 U.S. 205", "U.S.",
     ["Compelled adoption of the government's viewpoint as a condition on funding.",
      "Conditions defining the limits of the program versus conditions reaching outside it."]),
    ("aosi-ii-2020", "Alliance for Open Soc", "591 U.S. 430", "U.S.",
     ["Foreign organizations operating abroad hold no First Amendment rights."]),
    ("bridges-wixon-1945", "Bridges", "326 U.S. 135", "U.S.",
     ["Freedom of speech and press are accorded to aliens residing in this country."]),
    ("verdugo-urquidez-1990", "Verdugo-Urquidez", "494 U.S. 259", "U.S.",
     ["Constitutional protections and the significant voluntary connection to the United States."]),
    ("sweezy-1957", "Sweezy", "354 U.S. 234", "U.S.",
     ["Academic freedom and political expression; the essentiality of freedom in universities."]),
    ("keyishian-1967", "Keyishian", "385 U.S. 589", "U.S.",
     ["Academic freedom as a special concern of the First Amendment."]),
    ("hlp-2010", "Humanitarian Law Project", "561 U.S. 1", "U.S.",
     ["Material support ban applied to speech coordinated with designated organizations.",
      "Independent advocacy distinguished from coordinated support.",
      "Deference to executive predictive judgments in national security."]),
    ("ward-1989", "Ward", "491 U.S. 781", "U.S.",
     ["Narrow tailoring in the content-neutral context does not require the least restrictive means."]),
    ("mccullen-2014", "McCullen", "134 S. Ct. 2518", "U.S.",
     ["Government must show it seriously undertook less intrusive alternatives."]),
    ("ashcroft-aclu-2004", "Ashcroft", "542 U.S. 656", "U.S.",
     ["Less restrictive alternatives and the government's burden to prove their inadequacy."]),
    ("minneapolis-star-1983", "Minneapolis Star", "460 U.S. 575", "U.S.",
     ["Differential burdens on the means of producing speech are themselves suspect."]),
    ("zauderer-1985", "Zauderer", "471 U.S. 626", "U.S.",
     ["Compelled disclosure of purely factual and uncontroversial commercial information."]),
    ("naacp-alabama-1958", "Advancement of Colored People", "357 U.S. 449", "U.S.",
     ["Compelled disclosure of membership and the freedom of association."]),
    ("talley-1960", "Talley", "362 U.S. 60", "U.S.",
     ["Anonymous publication and the identification requirement's chilling effect."]),
    ("lukumi-1993", "Lukumi", "508 U.S. 520", "U.S.",
     ["Underinclusiveness as evidence that a law does not serve its asserted interest."]),
    ("williams-yulee-2015", "Williams-Yulee", "575 U.S. 433", "U.S.",
     ["Underinclusiveness tolerated where the law addresses the problem it identifies."]),
    ("defense-distributed-5th-2016", "Defense Distributed", "838 F.3d 451", "5th Cir.",
     ["Export control applied to computer files for manufacturing; preliminary injunction posture."]),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data/corpus/openweights_cases.jsonl"))
    parser.add_argument("--batch", type=int, default=9, help="citations per request")
    parser.add_argument("--sleep", type=float, default=13.0, help="seconds between requests (5/min budget)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import httpx
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import os

    token = os.environ.get("LRG_COURTLISTENER_TOKEN", "").strip()
    if not token:
        print("FATAL: LRG_COURTLISTENER_TOKEN is not set", file=sys.stderr)
        return 2

    batches = [AUTHORITIES[i : i + args.batch] for i in range(0, len(AUTHORITIES), args.batch)]
    print(f"{len(AUTHORITIES)} authorities in {len(batches)} request(s)")
    if args.dry_run:
        return 0

    resolved: dict[str, dict] = {}
    for n, batch in enumerate(batches, start=1):
        block = "\n".join(f"{a[2]}" for a in batch)
        resp = httpx.post(
            "https://www.courtlistener.com/api/rest/v4/citation-lookup/",
            headers={"Authorization": f"Token {token}"},
            data={"text": block},
            timeout=60.0,
        )
        print(f"  request {n}/{len(batches)}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"    body: {resp.text[:200]}", file=sys.stderr)
            continue
        for item in resp.json():
            resolved[item.get("citation", "")] = item
        if n < len(batches):
            time.sleep(args.sleep)

    records, mismatches = [], []
    for rec_id, expected, cite, court, passages in AUTHORITIES:
        item = resolved.get(cite)
        clusters = (item or {}).get("clusters") or []
        status = (item or {}).get("status")
        # 200 = one match; 300 = several clusters for the same opinion, which is a
        # CourtListener data duplicate rather than an ambiguous citation. Accept
        # either, but only via a cluster whose name matches what we expected.
        if not item or status not in (200, 300) or not clusters:
            mismatches.append((rec_id, cite, f"unresolved (status={status})"))
            continue
        cluster = next(
            (c for c in clusters if _norm(expected) in _norm(c.get("case_name") or "")), None
        )
        if cluster is None:
            got = [c.get("case_name") for c in clusters]
            mismatches.append((rec_id, cite, f"name mismatch: expected ~{expected!r}, got {got}"))
            continue
        name = cluster.get("case_name") or ""
        volume, reporter, page = cite.split(" ", 1)[0], cite.rsplit(" ", 1)[0].split(" ", 1)[1], cite.rsplit(" ", 1)[1]
        date = cluster.get("date_filed") or ""
        records.append({
            "id": rec_id,
            "type": "case",
            "title": name,
            "reporter": reporter,
            "volume": int(volume),
            "page": int(page),
            "court": court,
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "url": f"https://www.courtlistener.com{cluster.get('absolute_url', '')}",
            "passages": passages,
            "status": "in_force",
            "status_note": f"Resolved via CourtListener citation-lookup; cluster id {cluster.get('id')}.",
            "unverified": False,
        })

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"\nwrote {len(records)} verified record(s) -> {out}")
    if mismatches:
        print(f"\n{len(mismatches)} NOT written (resolve manually):")
        for rec_id, cite, why in mismatches:
            print(f"  {rec_id:<32} {cite:<18} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
