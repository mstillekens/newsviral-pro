#!/bin/bash
# VOZ DEL PUEBLO — webapp launcher
# Starts uvicorn bound to all interfaces and prints how to reach it from
# your phone (same Wi-Fi) and from anywhere (via Cloudflare Tunnel).

set -e
cd "$(dirname "$0")/.."

# Load .env (REPLICATE_API_TOKEN, ANTHROPIC_API_KEY, optional MINIMAX_VOICE_ID)
if [ -f .env ]; then
  set -a; . .env; set +a
fi

PORT="${PORT:-8000}"
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '?.?.?.?')"

cat <<INFO

╭─ VOZ DEL PUEBLO ────────────────────────────────────────────╮
│                                                              │
│  Mac local         http://localhost:${PORT}                      │
│  Misma Wi-Fi       http://${LAN_IP}:${PORT}                       │
│                                                              │
│  Para acceder DESDE FUERA de tu Wi-Fi:                       │
│  En otra terminal:                                           │
│      brew install cloudflared        (una sola vez)          │
│      cloudflared tunnel --url http://localhost:${PORT}            │
│  Te dará un URL .trycloudflare.com pegable en el celular.    │
│                                                              │
╰──────────────────────────────────────────────────────────────╯

INFO

exec .venv/bin/uvicorn webapp.server:app --host 0.0.0.0 --port "$PORT" --reload
