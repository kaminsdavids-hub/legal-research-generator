#!/usr/bin/env bash
# Two rounds of three runs on IDENTICAL code, to measure the round-level noise
# floor. Nothing may change between them: that is the whole point.
set -uo pipefail
cd "$HOME/legal-research-generator"
LOG="$HOME/legal-research-generator/logs/eval_tworounds.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

set -a; . "$HOME/lrg-eval.env"; set +a
export OLLAMA_KEEP_ALIVE=4h

FP=$(find modules/dialectic evals -name '*.py' -not -path '*__pycache__*' | sort | xargs md5sum | md5sum | cut -c1-12)
say "code fingerprint at start: ${FP}"

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: ollama not answering"; exit 1
fi
.venv/bin/python evals/warm_cite_cache.py --corpus data/corpus/openweights.jsonl \
    --cite-cache evals/.cache/courtlistener.json >>"$LOG" 2>&1 || {
  say "ABORT: citation cache incomplete"; exit 1; }

for r in A B; do
  RID="floor-${r}"
  say "########## round ${RID} ##########"
  for i in 1 2 3; do
    say "===== ${RID} run ${i}/3 starting ====="
    .venv/bin/python -u evals/run_eval.py --corpus data/corpus/openweights.jsonl \
        --cite-cache evals/.cache/courtlistener.json --round-id "$RID" >>"$LOG" 2>&1
    say "===== ${RID} run ${i}/3 finished (exit $?) ====="
  done
done

FP2=$(find modules/dialectic evals -name '*.py' -not -path '*__pycache__*' | sort | xargs md5sum | md5sum | cut -c1-12)
say "code fingerprint at end: ${FP2}"
if [ "$FP" != "$FP2" ]; then
  say "WARNING: code changed mid-measurement (${FP} -> ${FP2}); the floor is not valid"
fi
say "both rounds complete; round report follows"
.venv/bin/python evals/round_report.py --last 6 --per-round 3 >>"$LOG" 2>&1
say "done"
