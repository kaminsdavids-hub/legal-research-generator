#!/usr/bin/env bash
# Launch local, OpenAI-compatible model servers for the expert pool on a DGX Spark
# using vLLM. Each expert gets its own port; the backend is pointed at them via the
# LRG_*_BASE_URL / LRG_*_MODEL environment variables (see .env.example).
#
# Nothing here calls the cloud. Adjust the model IDs to the checkpoints you have
# pulled locally (Hugging Face cache or a local path).
set -euo pipefail

# Expert -> (port, model). Override any of these via the environment.
SAUL_MODEL="${LRG_SAUL_MODEL:-Equall/Saul-7B-Instruct-v1}"
WRITER_MODEL="${LRG_WRITER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GEMMA_MODEL="${LRG_GEMMA_MODEL:-google/gemma-3-4b-it}"
HERMES_MODEL="${LRG_HERMES_MODEL:-nvidia/Nemotron-3-Nano-4B-v1}"
HERMES3_MODEL="${LRG_HERMES3_MODEL:-NousResearch/Hermes-3-Llama-3.1-8B}"

SAUL_PORT="${SAUL_PORT:-8101}"
WRITER_PORT="${WRITER_PORT:-8103}"
GEMMA_PORT="${GEMMA_PORT:-8105}"
HERMES_PORT="${HERMES_PORT:-8106}"
HERMES3_PORT="${HERMES3_PORT:-8107}"

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
serve saul   "$SAUL_MODEL"   "$SAUL_PORT"   "${SAUL_FRAC:-0.20}"
serve writer "$WRITER_MODEL" "$WRITER_PORT" "${WRITER_FRAC:-0.22}"
serve gemma  "$GEMMA_MODEL"  "$GEMMA_PORT"  "${GEMMA_FRAC:-0.14}"
serve hermes "$HERMES_MODEL" "$HERMES_PORT" "${HERMES_FRAC:-0.17}"
serve hermes3 "$HERMES3_MODEL" "$HERMES3_PORT" "${HERMES3_FRAC:-0.20}"

cat <<EOF

All expert servers launching. Point the backend at them, e.g.:

  export LRG_LLM_MODE=openai
  export LRG_SAUL_BASE_URL=http://127.0.0.1:${SAUL_PORT}/v1
  export LRG_WRITER_BASE_URL=http://127.0.0.1:${WRITER_PORT}/v1
  export LRG_GEMMA_BASE_URL=http://127.0.0.1:${GEMMA_PORT}/v1
  export LRG_HERMES_BASE_URL=http://127.0.0.1:${HERMES_PORT}/v1
  export LRG_HERMES3_BASE_URL=http://127.0.0.1:${HERMES3_PORT}/v1

Tail logs in ./logs/. Stop with: pkill -f 'vllm serve'
EOF

wait
