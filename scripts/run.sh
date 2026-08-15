#!/usr/bin/env bash
# Run the backend API and the frontend dev server together. Ctrl-C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LRG_HOST="${LRG_HOST:-127.0.0.1}"
export LRG_PORT="${LRG_PORT:-8000}"

echo "==> Starting backend on http://${LRG_HOST}:${LRG_PORT}"
./.venv/bin/lrg-api &
BACKEND_PID=$!

cleanup() {
  echo "\n==> Stopping..."
  kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting frontend on http://127.0.0.1:3000"
(cd frontend && NEXT_PUBLIC_API_URL="http://${LRG_HOST}:${LRG_PORT}" npm run dev) &
FRONTEND_PID=$!

wait
