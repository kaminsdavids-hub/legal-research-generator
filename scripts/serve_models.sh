#!/usr/bin/env bash
# Launch local, OpenAI-compatible model servers for the four experts on a DGX Spark
# using vLLM. Each expert gets its own port; the backend is pointed at them via the
# LRG_*_BASE_URL / LRG_*_MODEL environment variables (see .env.example).
#
# Nothing here calls the cloud. Adjust the model IDs to the checkpoints you have
# pulled locally (Hugging Face cache or a local path).
set -euo pipefail

# Expert -> (port, model). Override any of these via the environment.
SAUL_MODEL="${LRG_SAUL_MODEL:-Equall/Saul-7B-Instruct-v1}"
FINANCE_MODEL="${LRG_FINANCE_MODEL:-AdaptLLM/finance-chat}"
WRITER_MODEL="${LRG_WRITER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
ROUTER_MODEL="${LRG_ROUTER_MODEL:-google/gemma-3-1b-it}"

SAUL_PORT="${SAUL_PORT:-8101}"
FINANCE_PORT="${FINANCE_PORT:-8102}"
WRITER_PORT="${WRITER_PORT:-8103}"
ROUTER_PORT="${ROUTER_PORT:-8104}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "!! vllm not found. Install with: pip install vllm" >&2
  echo "   (On the DGX Spark, use the CUDA build appropriate for your driver.)" >&2
  exit 1
fi

serve() {
  local name="$1" model="$2" port="$3" frac="$4"
  echo "==> Serving ${name}: ${model} on :${port}"
  vllm serve "$model" \
    --port "$port" \
    --gpu-memory-utilization "$frac" \
    --served-model-name "$name" \
    >"logs/${name}.log" 2>&1 &
}

mkdir -p logs
serve saul    "$SAUL_MODEL"    "$SAUL_PORT"    "${SAUL_FRAC:-0.30}"
serve finance "$FINANCE_MODEL" "$FINANCE_PORT" "${FINANCE_FRAC:-0.20}"
serve writer  "$WRITER_MODEL"  "$WRITER_PORT"  "${WRITER_FRAC:-0.40}"
serve router  "$ROUTER_MODEL"  "$ROUTER_PORT"  "${ROUTER_FRAC:-0.10}"

cat <<EOF

All four expert servers launching. Point the backend at them, e.g.:

  export LRG_LLM_MODE=openai
  export LRG_SAUL_BASE_URL=http://127.0.0.1:${SAUL_PORT}/v1
  export LRG_FINANCE_BASE_URL=http://127.0.0.1:${FINANCE_PORT}/v1
  export LRG_WRITER_BASE_URL=http://127.0.0.1:${WRITER_PORT}/v1
  export LRG_ROUTER_BASE_URL=http://127.0.0.1:${ROUTER_PORT}/v1

Tail logs in ./logs/. Stop with: pkill -f 'vllm serve'
EOF

wait
