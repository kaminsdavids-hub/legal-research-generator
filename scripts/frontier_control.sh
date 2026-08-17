#!/usr/bin/env bash
# Control arm: run the eval with the THESIS role on a frontier API and every
# other role unchanged.
#
# The question it answers is the one no local-only run can: when an authority
# proposed by the thesis arm does not survive the support gate, is that the
# generator or the pipeline? Holding retrieval, the corpus, the gates, the judge
# and the two other debate roles fixed, and changing only which model drafts the
# thesis, is the one configuration where the answer is attributable.
#
# The prediction is recorded before the run, not after: if the generator is the
# bottleneck, thesis-side authority survival should rise materially above the
# all-local baseline of 1/14 printed below. If it lands at 1/14 again, the
# limit is retrieval and the corpus, and no better drafter fixes it.
#
# It refuses to start without a key rather than falling back to a local model.
# A "frontier" number produced by llama3.1 is worse than no number: it is the
# silent-degradation shape that has already cost this repository three separate
# investigations (REMEDIATION §11.6, §14, §21), and here it would corrupt the
# only comparison the run exists to make.
#
#   export LRG_DIALECTIC_THESIS_API_KEY='sk-...' && ./scripts/frontier_control.sh
#
# Cost: 4 tiny preflight calls plus one thesis call per question, before retries.
set -uo pipefail
cd "$HOME/legal-research-generator" || exit 1

: "${FRONTIER_MODEL:=claude-sonnet-4-5}"
: "${FRONTIER_BASE_URL:=https://api.anthropic.com/v1}"
: "${CORPUS:=data/corpus/openweights.jsonl}"
: "${LIMIT:=0}"

LOG="$HOME/legal-research-generator/logs/frontier_control.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -z "${LRG_DIALECTIC_THESIS_API_KEY:-}" ]; then
  say "ABORT: LRG_DIALECTIC_THESIS_API_KEY is empty."
  say "       This script will not fall back to a local model: the entire point"
  say "       of the comparison is which model produced the numbers."
  exit 1
fi

say "preflight: lineup and reachability (thesis=${FRONTIER_MODEL} on ${FRONTIER_BASE_URL})"
.venv/bin/python scripts/preflight_models.py \
    --thesis "$FRONTIER_MODEL" \
    --thesis-base-url "$FRONTIER_BASE_URL" \
    --thesis-api-key-env LRG_DIALECTIC_THESIS_API_KEY 2>&1 | tee -a "$LOG"
# shellcheck disable=SC2181  # PIPESTATUS is what we want, not $?
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  say "ABORT: preflight failed; nothing was spent on the run itself"
  exit 1
fi

say "baseline to compare against: all-local claim-arm authority survival was 1/14 (REMEDIATION §14)"

.venv/bin/python -u evals/run_eval.py \
    --corpus "$CORPUS" \
    --limit "$LIMIT" \
    --thesis "$FRONTIER_MODEL" \
    --thesis-base-url "$FRONTIER_BASE_URL" \
    --thesis-api-key-env LRG_DIALECTIC_THESIS_API_KEY \
    --round-id "frontier-control-$(date +%Y%m%d-%H%M%S)" 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"

say "run finished (exit ${status}); compare thesis-arm survival against 1/14 above"
exit "$status"
