#!/usr/bin/env bash
# Run one optimization ROUND: N runs sharing a round id, averaged before the
# plateau rule reads them.
#
# A citation gate flip moves a single run's mean by ~0.24, over the 0.2
# threshold, and four different questions have flipped across four sets. Fixing
# them individually looked like convergence and was not. Averaging removes the
# assumption that none lands rather than tolerating it.
set -uo pipefail
cd "$HOME/legal-research-generator"

RUNS="${1:-3}"
ROUND_ID="${2:-round-$(date +%Y%m%d-%H%M%S)}"
LOG="$HOME/legal-research-generator/logs/eval_round.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

set -a; . "$HOME/lrg-eval.env"; set +a
export OLLAMA_KEEP_ALIVE=4h

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: ollama not answering"; exit 1
fi
.venv/bin/python evals/warm_cite_cache.py --corpus data/corpus/openweights.jsonl \
    --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1 || {
  say "ABORT: citation cache incomplete; runs would spend quota"; exit 1; }

say "round ${ROUND_ID}: ${RUNS} run(s)"
for i in $(seq 1 "$RUNS"); do
  say "===== run ${i}/${RUNS} starting ====="
  .venv/bin/python -u evals/run_eval.py --corpus data/corpus/openweights.jsonl \
      --cite-cache evals/.cache/courtlistener.json \
      --round-id "$ROUND_ID" >>"$LOG" 2>&1
  say "===== run ${i}/${RUNS} finished (exit $?) ====="
done
say "round complete; round report follows"
.venv/bin/python evals/round_report.py --per-round "$RUNS" >>"$LOG" 2>&1
say "done"
