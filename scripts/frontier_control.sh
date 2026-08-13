#!/usr/bin/env bash
# The §17 frontier control. Set the key in YOUR shell first -- never in a chat
# transcript, and never committed:
#
#   export LRG_DIALECTIC_THESIS_API_KEY='sk-ant-...'
#
# Then run this script. It refuses to start without the key rather than falling
# back to a local model and reporting a "frontier" result that is not one.
set -euo pipefail

: "${LRG_DIALECTIC_THESIS_API_KEY:?set LRG_DIALECTIC_THESIS_API_KEY in your shell first}"

export LRG_DIALECTIC_THESIS_BASE_URL="https://api.anthropic.com/v1"
export LRG_DIALECTIC_THESIS_MODEL="${LRG_DIALECTIC_THESIS_MODEL:-claude-sonnet-4-5}"
export LRG_CORPUS_PATH="data/corpus/openweights.jsonl"
export OLLAMA_KEEP_ALIVE="45m"

echo "== preflight (4 tiny calls, ~1c) =="
python evals/preflight_models.py

echo
echo "== live loop, thesis on the frontier model =="
python evals/run_loop_eval.py --live --only S2-expressive-conduct --out frontier.json

echo
echo "== scorer comparison over what it produced =="
python evals/compare_support_scorers.py frontier.json

echo
echo "== against the §14 baseline: claim-arm authority survival was 1/14 =="
