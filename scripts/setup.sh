#!/usr/bin/env bash
# Set up the backend (Python) and frontend (Node) for local development or the
# DGX Spark. Idempotent: safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- backend -------------------------------------------------------------------
PY="${PYTHON:-python3.12}"
EXTRAS="${EXTRAS:-dev}"   # e.g. EXTRAS="dev,gpu,pdf" on the Spark

echo "==> Creating virtualenv (.venv) with $PY"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY" .venv
  uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"
else
  "$PY" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -e ".[${EXTRAS}]"
fi

# ---- frontend ------------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  echo "==> Installing frontend dependencies"
  (cd frontend && npm install)
else
  echo "!! npm not found; skipping frontend install"
fi

echo "==> Done. Activate with: source .venv/bin/activate"
