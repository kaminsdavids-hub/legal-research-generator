#!/usr/bin/env python3
"""Run the dialectic eval set against live local models.

This is the only file in ``evals/`` that touches ``legal_research.*`` or reads
application settings; ``harness.py`` stays transport-agnostic.

The judge is a **fifth** model. It must not share a family with any of the four
debate roles, for the same reason the debaters must not share one with each
other: a judge that is a sibling of a debater grades its own family's habits.

    python evals/run_eval.py --corpus data/corpus/openweights.jsonl

Holdouts are excluded unless ``--include-holdout`` is passed. Do not pass it
during optimization; it is for final validation only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.harness import (  # noqa: E402
    EvalSet,
    JudgeVerdict,
    LoggingRetriever,
    QuestionResult,
    RunReport,
    gate_citation_integrity,
    gate_temporal_validity,
    has_plateaued,
    judge_response,
)
from modules.dialectic.engine import DialecticChat  # noqa: E402
from modules.dialectic.roles import detect_family  # noqa: E402
from modules.dialectic.service import (  # noqa: E402
    _ClientAdapter,
    _CorpusCiteRetriever,
    is_local_endpoint,
)


def _corpus_index(corpus_path: str) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Return (status by cite, cites CourtListener cannot adjudicate).

    The second set is statutes, regulations and secondary material: the corpus
    is their verifier, because a citation-lookup call for a C.F.R. section
    returns nothing no matter how correct the section is.
    """
    from legal_research.citations.corpus import load_corpus

    index: dict[str, tuple[str, str]] = {}
    corpus_verified: set[str] = set()
    for record in load_corpus(corpus_path).records:
        entry = (record.status.value, record.status_note)
        if record.volume is not None and record.page is not None and record.reporter:
            index[f"{record.volume} {record.reporter} {record.page}"] = entry
            continue
        if record.code and record.section:
            cite = f"{record.code} {record.section}"
            index[cite] = entry
            corpus_verified.add(cite)
    return index, corpus_verified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=str(ROOT / "evals/openweights_first_amendment.json"))
    parser.add_argument("--corpus", default="data/corpus/sample_corpus.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-call timeout. Ollama swaps models between the five roles and a "
        "cold load alone costs ~40s, so 300s is not enough headroom.",
    )
    parser.add_argument("--thesis", default="saul:7b-instruct-v1")
    parser.add_argument("--antithesis", default="llama3.1:8b")
    parser.add_argument("--synthesis", default="gemma3:4b")
    parser.add_argument("--nli", default="nemotron-3-nano:4b")
    parser.add_argument(
        "--judge",
        default="hermes3:8b",
        help="fifth model; must be a distinct family. Calibrate with calibrate_judge.py "
        "before trusting a new one: saul rated pure failure 8.0/10.",
    )
    # Per-role endpoint overrides. Without these the eval can only measure a
    # lineup served from one place, so the question "is the bottleneck the local
    # models or the pipeline?" could not be asked: answering it means running
    # the identical harness with one role on a frontier API. An empty override
    # means --base-url, so an all-local run is configured exactly as before.
    for role in ("thesis", "antithesis", "synthesis", "nli", "judge"):
        parser.add_argument(f"--{role}-base-url", default="", help=f"endpoint override for {role}")
        parser.add_argument(
            f"--{role}-api-key-env",
            default="",
            help=f"environment variable holding {role}'s key; required if its endpoint is remote",
        )
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions")
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="FINAL VALIDATION ONLY - never during optimization",
    )
    parser.add_argument("--out", default=str(ROOT / "evals/results"))
    parser.add_argument(
        "--round-id",
        default="",
        help="tag this run as part of a round. Runs sharing an id are averaged "
        "before the plateau rule reads them, so one citation-gate flip cannot "
        "move the figure the rule sees.",
    )
    parser.add_argument(
        "--cite-cache",
        default=str(ROOT / "evals/.cache/courtlistener.json"),
        help="persistent citation-lookup cache shared across runs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # The judge is a fifth role and gets the same family discipline.
    families = {
        role: detect_family(model)
        for role, model in (
            ("thesis", args.thesis),
            ("antithesis", args.antithesis),
            ("synthesis", args.synthesis),
            ("nli", args.nli),
            ("judge", args.judge),
        )
    }
    if len(set(families.values())) != len(families):
        print(f"FATAL: roles share a model family: {families}", file=sys.stderr)
        return 2

    from legal_research.citations.corpus import load_corpus
    from legal_research.citations.retriever import MockRetriever
    from legal_research.config import get_settings
    from legal_research.llm.openai_compat import OpenAICompatLLM

    def make(role: str, model: str) -> _ClientAdapter:
        base_url = str(getattr(args, f"{role}_base_url", "") or "").strip() or args.base_url
        key_env = str(getattr(args, f"{role}_api_key_env", "") or "").strip()
        api_key = os.environ.get(key_env, "").strip() if key_env else ""
        if not api_key:
            if not is_local_endpoint(base_url):
                # Falling back to "ollama" here would send a run at a hosted
                # provider with a placeholder credential and report whatever
                # came back as this lineup's result. The whole point of a
                # control arm is which model produced the numbers.
                raise SystemExit(
                    f"FATAL: role {role!r} is served from {base_url} but "
                    f"{key_env or f'--{role}-api-key-env'} is empty"
                )
            api_key = "ollama"
        if not is_local_endpoint(base_url):
            print(f"NOTE: {role} runs on {base_url} — prompt text leaves this machine")
        return _ClientAdapter(
            model,
            OpenAICompatLLM(
                name=model,
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=args.timeout,
            ),
        )

    corpus = load_corpus(args.corpus)
    unverified = [r.id for r in corpus.records if r.unverified]
    if unverified:
        print(
            f"WARNING: {len(unverified)} corpus record(s) are marked unverified and are "
            f"the ground truth the gates measure against: {unverified}",
            file=sys.stderr,
        )

    retriever = LoggingRetriever(_CorpusCiteRetriever(MockRetriever(corpus), corpus))
    settings = get_settings()
    courtlistener = None
    token = str(getattr(settings, "courtlistener_token", "") or "").strip()
    from modules.dialectic.verification import (  # noqa: E402
        CourtListenerClient,
        PersistentCiteCache,
        RateBudget,
    )

    # Cross-run cache. A full run costs ~64 lookups against a 125/day quota, so
    # without this a repeat run cannot be done on the same day at all.
    #
    # Built whether or not a token is configured. It used to live inside the
    # `if token:` branch, so a tokenless run ignored every answer it had already
    # paid for: all 34 case cites in data/corpus/openweights.jsonl were cached,
    # and the run still failed the citation gate on every question because no
    # client existed to consult them.
    cite_cache = PersistentCiteCache(args.cite_cache)
    courtlistener = CourtListenerClient(
        token=token, budget=RateBudget(), cite_cache=cite_cache
    )
    if token:
        print(f"cite cache: {args.cite_cache} ({len(cite_cache)} citation(s) known)")
    else:
        print(
            f"CACHE-ONLY: no LRG_COURTLISTENER_TOKEN. {len(cite_cache)} citation(s) can be "
            f"resolved from {args.cite_cache}; any cite not already cached stays "
            "unverified and fails the citation-integrity gate. Cached answers are real "
            "prior lookups, so nothing is marked VERIFIED on an unverified path.",
            file=sys.stderr,
        )

    chat = DialecticChat(
        thesis_client=make("thesis", args.thesis),
        antithesis_client=make("antithesis", args.antithesis),
        synthesis_client=make("synthesis", args.synthesis),
        nli_client=make("nli", args.nli),
        courtlistener=courtlistener,
        retriever=retriever,
    )
    judge = make("judge", args.judge)

    eval_set = EvalSet.load(args.eval_set)
    questions = eval_set.select(include_holdout=args.include_holdout)
    if args.limit:
        questions = questions[: args.limit]

    if args.include_holdout:
        print("!! HOLDOUT SET INCLUDED - final validation only, never optimization\n")

    status_by_cite, corpus_verified = _corpus_index(args.corpus)
    report = RunReport(eval_set=eval_set.name, round_id=args.round_id)
    started = time.time()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
    out_path = out_dir / f"{eval_set.name}-{stamp}.json"

    for i, question in enumerate(questions, start=1):
        retriever.reset()
        print(f"[{i}/{len(questions)}] {question.id} ({question.cluster})", flush=True)
        # One slow call must not discard hours of completed work. A question that
        # fails is recorded as unmeasured and the run continues.
        try:
            turn = chat.chat(question.prompt())
            gates = [
                gate_citation_integrity(turn, retriever.retrieved_cites, corpus_verified),
                gate_temporal_validity(turn, status_by_cite),
            ]
            verdict = judge_response(judge, question, turn)
            result = QuestionResult(
                id=question.id,
                cluster=question.cluster,
                holdout=question.holdout,
                gates=gates,
                verdict=verdict,
                regenerated=turn.regenerated,
                crux_count=len(turn.cruxes),
                synthesis_note=turn.synthesis_note,
            )
            cruxes = len(turn.cruxes)
        except Exception as exc:  # noqa: BLE001 - a dead question is data, not a crash
            result = QuestionResult(
                id=question.id,
                cluster=question.cluster,
                holdout=question.holdout,
                gates=[],
                verdict=JudgeVerdict({}, "", parsed=False),
                error=f"{type(exc).__name__}: {exc}",
            )
            cruxes = 0
        report.results.append(result)
        # Checkpoint after every question: a crash at question 30 previously lost
        # everything, including 4.4 hours of completed work.
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        if result.error:
            flag, shown = "RUN ERROR", "unmeasured"
        elif not result.gates_passed:
            flag, shown = "GATE FAIL", "0.00"
        elif result.score is None:
            flag, shown = "JUDGE FAILED", "unscored"
        else:
            flag, shown = "PASS", f"{result.score:.2f}"
        extra = f"  regen={result.regenerated}" if result.regenerated else ""
        print(f"     {flag}  mean={shown}  cruxes={cruxes}{extra}", flush=True)

    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"overall mean : {report.overall_mean:.3f}  "
          f"(over {len(report._scored)}/{len(report.results)} scored)")
    print(f"per cluster  : {report.cluster_means()}")
    print(f"gate failures: {report.gate_failures() or 'none'}")
    print(f"retry cost   : {report.retry_cost()}")
    print(f"judge failed : {report.unscored() or 'none'}  (excluded from the mean, not scored 0)")
    if report.errors():
        print(f"run errors   : {report.errors()}")
    print(f"written      : {out_path}")

    history = sorted(out_dir.glob(f"{eval_set.name}-*.json"))
    means = [json.loads(p.read_text())["overall_mean"] for p in history]
    if has_plateaued(means):
        print(f"PLATEAU: overall mean moved < 0.2 across the last 3 rounds {means[-4:]}")
    print(f"elapsed      : {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
