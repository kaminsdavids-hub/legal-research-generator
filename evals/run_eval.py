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
from modules.dialectic.service import _ClientAdapter, _CorpusCiteRetriever  # noqa: E402


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

    def make(model: str) -> _ClientAdapter:
        return _ClientAdapter(
            model,
            OpenAICompatLLM(
                name=model,
                base_url=args.base_url,
                model=model,
                api_key="ollama",
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
    if token:
        from modules.dialectic.verification import (
            CourtListenerClient,
            PersistentCiteCache,
            RateBudget,
        )

        # Cross-run cache. A full run costs ~64 lookups against a 125/day quota,
        # so without this a repeat run cannot be done on the same day at all.
        cite_cache = PersistentCiteCache(args.cite_cache)
        print(f"cite cache: {args.cite_cache} ({len(cite_cache)} citation(s) known)")
        courtlistener = CourtListenerClient(
            token=token, budget=RateBudget(), cite_cache=cite_cache
        )
    else:
        print(
            "WARNING: no LRG_COURTLISTENER_TOKEN; no slot can reach VERIFIED, so the "
            "citation-integrity gate will fail every question that resolves a cite.",
            file=sys.stderr,
        )

    chat = DialecticChat(
        thesis_client=make(args.thesis),
        antithesis_client=make(args.antithesis),
        synthesis_client=make(args.synthesis),
        nli_client=make(args.nli),
        courtlistener=courtlistener,
        retriever=retriever,
    )
    judge = make(args.judge)

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
        print(f"     {flag}  mean={shown}  cruxes={cruxes}", flush=True)

    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"overall mean : {report.overall_mean:.3f}  "
          f"(over {len(report._scored)}/{len(report.results)} scored)")
    print(f"per cluster  : {report.cluster_means()}")
    print(f"gate failures: {report.gate_failures() or 'none'}")
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
