#!/usr/bin/env bash
# Reclaim disk from Hugging Face hub model caches. SAFE BY DEFAULT: dry-run unless
# APPLY=1. Refuses any path that does not live directly under the hub cache dir.
#
# Usage (stream this file over SSH stdin; args after --):
#   ssh spark-node 'bash -s -- meta-llama/Llama-3.1-70B-Instruct' < scripts/reclaim_hf_cache.sh
#   ssh spark-node 'APPLY=1 bash -s -- meta-llama/Llama-3.1-70B-Instruct' < scripts/reclaim_hf_cache.sh
set -uo pipefail
HUB="${HF_HUB:-$HOME/.cache/huggingface/hub}"
APPLY="${APPLY:-0}"        # APPLY=1 to actually delete; default = dry-run
echo "hub: $HUB"; echo "--- disk before ---"; df -h "$HOME" | tail -1
for name in "$@"; do
  d="$HUB/models--${name//\//--}"
  case "$d" in "$HUB"/models--?*) : ;;   # guard: must live under the hub cache
    *) echo "REFUSING unsafe path: '$d'"; continue ;; esac
  [ -d "$d" ] || { echo "already gone: $d"; continue; }
  sz="$(du -sh "$d" 2>/dev/null | cut -f1)"
  if [ "$APPLY" = 1 ]; then rm -rf "$d" && echo "deleted ($sz): $d"
  else echo "[dry-run] would delete ($sz): $d  (set APPLY=1 to remove)"; fi
done
echo "--- disk after ---"; df -h "$HOME" | tail -1
