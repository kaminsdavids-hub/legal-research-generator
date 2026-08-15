#!/usr/bin/env bash
# Serve the studio on the local network. Auto-detects this machine's LAN IP and
# launches the backend (0.0.0.0:8000) and frontend (0.0.0.0:3000) wired together so
# any device on the same Wi-Fi/LAN can use it. Ctrl-C stops both.
#
# Note: macOS may block incoming connections via its Application Firewall. If remote
# devices can't connect, allow Node + Python (System Settings > Network > Firewall),
# or run: sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${LRG_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# ---- detect LAN IP -------------------------------------------------------------
detect_ip() {
  if [[ -n "${LAN_IP:-}" ]]; then echo "$LAN_IP"; return; fi
  local ip=""
  if [[ "$(uname)" == "Darwin" ]]; then
    for iface in en0 en1 en2; do
      ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
      [[ -n "$ip" ]] && break
    done
  fi
  # Fallback: first non-loopback IPv4 (Linux or unusual macOS setups).
  if [[ -z "$ip" ]]; then
    ip="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -n1 || true)"
  fi
  echo "$ip"
}

IP="$(detect_ip)"
if [[ -z "$IP" ]]; then
  echo "!! Could not detect a LAN IP. Set it manually: LAN_IP=192.168.x.y $0" >&2
  exit 1
fi

# ---- free the ports if something is already listening --------------------------
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "==> Port ${port} in use; stopping $(echo "$pids" | tr '\n' ' ')"
    echo "$pids" | xargs kill 2>/dev/null || true
  fi
done
sleep 1

API_URL="http://${IP}:${BACKEND_PORT}"

cleanup() {
  echo ""
  echo "==> Stopping..."
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on http://0.0.0.0:${BACKEND_PORT}"
LRG_HOST=0.0.0.0 LRG_PORT="$BACKEND_PORT" ./.venv/bin/lrg-api &
BACKEND_PID=$!

echo "==> Starting frontend on http://0.0.0.0:${FRONTEND_PORT} (API -> ${API_URL})"
(cd frontend && NEXT_PUBLIC_API_URL="$API_URL" npm run dev -- -H 0.0.0.0 -p "$FRONTEND_PORT") &
FRONTEND_PID=$!

cat <<EOF

────────────────────────────────────────────────────────────
  Studio is live on your network:

    UI   ->  http://${IP}:${FRONTEND_PORT}
    API  ->  ${API_URL}

  Share the UI link with anyone on the same Wi-Fi/LAN.
  (Ctrl-C to stop both servers.)
────────────────────────────────────────────────────────────
EOF

wait
