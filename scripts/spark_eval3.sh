#!/usr/bin/env bash
# Run the dialectic eval N times on the Spark.
#
# 121 GB RAM holds all five roles resident (~35 GB), so unlike the 16 GB laptop
# OLLAMA_KEEP_ALIVE actually helps here: models load once instead of swapping
# on every question.
set -uo pipefail
cd "$HOME/legal-research-generator"

RUNS="${1:-3}"
LOG="$HOME/legal-research-generator/logs/eval_spark3.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

set -a; . "$HOME/lrg-eval.env"; set +a
export OLLAMA_KEEP_ALIVE=4h

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: ollama not answering on 11434"; exit 1
fi

say "starting ${RUNS} run(s) on $(hostname); keep_alive=${OLLAMA_KEEP_ALIVE}"
for i in $(seq 1 "$RUNS"); do
  say "===== run ${i}/${RUNS} starting ====="
  .venv/bin/python -u evals/run_eval.py \
      --corpus data/corpus/openweights.jsonl \
      --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1
  say "===== run ${i}/${RUNS} finished (exit $?) ====="
done
say "all runs complete; variance report follows"
.venv/bin/python evals/variance_report.py >>"$LOG" 2>&1
say "done"
