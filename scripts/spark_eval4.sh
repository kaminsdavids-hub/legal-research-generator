#!/usr/bin/env bash
# Three eval runs, gated on proving the cache covers the workload first.
#
# Three previous attempts were invalidated by quota exhaustion. Each time the
# cache looked complete and the runs still went to the network. This refuses to
# start unless a dry probe shows zero live calls are needed.
set -uo pipefail
cd "$HOME/legal-research-generator"

START_EPOCH="${1:-0}"
RUNS="${2:-3}"
LOG="$HOME/legal-research-generator/logs/eval_spark4.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

set -a; . "$HOME/lrg-eval.env"; set +a
export OLLAMA_KEEP_ALIVE=4h

while [ "$(date +%s)" -lt "$START_EPOCH" ]; do
  r=$(( START_EPOCH - $(date +%s) ))
  say "waiting ${r}s (~$(( r / 3600 ))h $(( (r % 3600) / 60 ))m) until start"
  sleep $(( r > 900 ? 900 : r + 5 ))
done

say "topping up the citation cache (should need 0 calls)"
.venv/bin/python -u evals/warm_cite_cache.py --corpus data/corpus/openweights.jsonl \
    --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1 || {
  say "ABORT: cache incomplete and could not be filled"; exit 1; }

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: ollama not answering"; exit 1
fi

for i in $(seq 1 "$RUNS"); do
  say "===== run ${i}/${RUNS} starting ====="
  .venv/bin/python -u evals/run_eval.py --corpus data/corpus/openweights.jsonl \
      --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1
  say "===== run ${i}/${RUNS} finished (exit $?) ====="
done
say "all runs complete; variance report follows"
.venv/bin/python evals/variance_report.py >>"$LOG" 2>&1
say "done"
