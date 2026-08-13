#!/usr/bin/env bash
# Wait for the CourtListener daily quota to reset, then run the eval N times.
#
# A full run costs ~64 lookups against a 125/day quota, so only the first run
# spends quota: it populates the persistent citation cache and the runs after it
# are served from disk. That is what makes a variance estimate possible in one
# day rather than three.
#
#   scripts/run_eval_after_quota.sh [RESET_EPOCH] [RUNS]
#
# Detached and safe to leave: writes to logs/eval_scheduled.log and survives the
# terminal closing. Aborts rather than starting a doomed run if the quota is
# still refused or Ollama is not answering.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESET_EPOCH="${1:-0}"
RUNS="${2:-3}"
LOG="$ROOT/logs/eval_scheduled.log"
mkdir -p "$ROOT/logs"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "scheduled eval: ${RUNS} run(s); waiting for quota reset at epoch ${RESET_EPOCH}"

# Sleep until the reset, in chunks so the log shows progress.
while [ "$(date +%s)" -lt "$RESET_EPOCH" ]; do
  remaining=$(( RESET_EPOCH - $(date +%s) ))
  say "waiting ${remaining}s (~$(( remaining / 60 ))m) for CourtListener quota"
  sleep $(( remaining > 600 ? 600 : remaining + 5 ))
done

# Confirm the quota really is back before spending hours of GPU on it. The
# reset estimate comes from a retry-after header and can be optimistic.
for attempt in 1 2 3 4 5 6; do
  code=$(.venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import httpx
tok = os.environ.get("LRG_COURTLISTENER_TOKEN", "").strip()
try:
    r = httpx.post(
        "https://www.courtlistener.com/api/rest/v4/citation-lookup/",
        headers={"Authorization": f"Token {tok}"},
        data={"text": "392 U.S. 1"}, timeout=30.0,
    )
    print(r.status_code)
except Exception:
    print("000")
PY
)
  say "quota probe ${attempt}: HTTP ${code}"
  [ "$code" = "200" ] && break
  [ "$attempt" = "6" ] && { say "ABORT: quota still refused after 6 probes"; exit 1; }
  sleep 600
done

if ! curl -s --max-time 8 http://127.0.0.1:11434/api/tags >/dev/null; then
  say "ABORT: Ollama is not answering on 11434; start it before scheduling"
  exit 1
fi

for i in $(seq 1 "$RUNS"); do
  say "===== run ${i}/${RUNS} starting ====="
  .venv/bin/python -u evals/run_eval.py --corpus data/corpus/openweights.jsonl >>"$LOG" 2>&1
  say "===== run ${i}/${RUNS} finished (exit $?) ====="
done

say "all runs complete; variance report follows"
.venv/bin/python evals/variance_report.py >>"$LOG" 2>&1
say "done"
