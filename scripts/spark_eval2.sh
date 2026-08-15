#!/usr/bin/env bash
# Wait for the CourtListener quota, warm the cache in a handful of calls, then
# run the eval N times. Every run after the warm is served from disk, so the
# quota cannot run out mid-set -- which is what invalidated the previous three.
set -uo pipefail
cd "$HOME/legal-research-generator"

RESET_EPOCH="${1:-0}"
RUNS="${2:-3}"
LOG="$HOME/legal-research-generator/logs/eval_spark2.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

set -a; . "$HOME/lrg-eval.env"; set +a
export OLLAMA_KEEP_ALIVE=4h

say "waiting for quota reset at epoch ${RESET_EPOCH}"
while [ "$(date +%s)" -lt "$RESET_EPOCH" ]; do
  r=$(( RESET_EPOCH - $(date +%s) ))
  say "waiting ${r}s (~$(( r / 60 ))m)"
  sleep $(( r > 600 ? 600 : r + 5 ))
done

say "warming the citation cache"
if ! .venv/bin/python -u evals/warm_cite_cache.py \
      --corpus data/corpus/openweights.jsonl \
      --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1; then
  say "ABORT: cache warm did not complete; runs would spend quota and fail like last time"
  exit 1
fi
say "cache warm complete"

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: ollama not answering"; exit 1
fi

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
