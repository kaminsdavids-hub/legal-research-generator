#!/usr/bin/env bash
# Smoke-test Ollama models through the OpenAI-compatible endpoint. Asserts a real
# response (JSON must contain "content"), caps each call with curl -m 120 (no hang),
# and unloads each model afterward. Exits 0 only if EVERY model responded.
#
# Usage (stream this file over SSH stdin; args after --):
#   ssh spark-node 'bash -s -- llama3.1:8b saul:7b-instruct-v1 gemma3:4b' < scripts/model_smoke.sh
set -uo pipefail
API="${OLLAMA_API:-http://127.0.0.1:11434}"
fail=0; echo "--- models present ---"; ollama list || true
for m in "$@"; do
  echo "=== $m ==="
  resp="$(curl -s -m 120 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply in one short sentence: what is 2+2, and the capital of France?\"}],\"max_tokens\":60,\"temperature\":0.2}")"
  if printf '%s' "$resp" | grep -q '"content"'; then
    echo "  OK: $(printf '%s' "$resp" | sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | head -c 120)"
  else echo "  FAIL: $m -> $resp"; fail=1; fi
  ollama stop "$m" 2>/dev/null && echo "  (unloaded $m)"
done
exit $fail
