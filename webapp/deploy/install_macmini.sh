#!/bin/bash
# install_macmini.sh — bootstrap script that turns a fresh Mac mini into a
# running VOZ DEL PUEBLO host.
#
# Usage on the Mac mini:
#   git clone https://github.com/mstillekens/newsviral-pro.git
#   cd newsviral-pro
#   ./webapp/deploy/install_macmini.sh
#
# After it finishes you'll be guided through:
#   1) Filling in .env with your API keys + WEBAPP_PASSWORD
#   2) Authenticating Cloudflare Tunnel
#   3) Loading the launchd services

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
HOME_DIR="$HOME"
PLIST_DIR="$HOME_DIR/Library/LaunchAgents"

cd "$PROJECT_DIR"
echo "📂 project dir: $PROJECT_DIR"

# ---------- 1. System deps ----------
echo ""
echo "━━━ 1/5  Homebrew packages ━━━"
if ! command -v brew >/dev/null; then
    echo "Homebrew not found. Install it first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
brew list --formula | grep -q "^ffmpeg-full$" || brew install ffmpeg-full
brew list --formula | grep -q "^cloudflared$" || brew install cloudflared
brew list --formula | grep -q "^python@3.12$" || brew install python@3.12 || true
echo "✓ system deps installed"

# ---------- 2. Python venv ----------
echo ""
echo "━━━ 2/5  Python venv + deps ━━━"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "✓ python venv ready"

# ---------- 3. .env scaffold ----------
echo ""
echo "━━━ 3/5  .env scaffold ━━━"
if [ ! -f .env ]; then
    cat > .env <<'ENV'
# Required: keys for the AI services
REPLICATE_API_TOKEN=
ANTHROPIC_API_KEY=

# Optional: cloned MiniMax voice id (set after running clone_voice.py)
MINIMAX_VOICE_ID=

# Required for public hosting: HTTP Basic auth credentials for the webapp
WEBAPP_USERNAME=voz
WEBAPP_PASSWORD=
ENV
    echo "✓ .env created from template — EDIT IT NOW and fill in the values"
    echo "   path: $PROJECT_DIR/.env"
else
    echo "✓ .env already exists (not overwriting)"
fi

# ---------- 4. launchd plists ----------
echo ""
echo "━━━ 4/5  launchd services ━━━"
mkdir -p "$PLIST_DIR" logs

WEBAPP_PLIST="$PLIST_DIR/com.vozdelpueblo.webapp.plist"
TUNNEL_PLIST="$PLIST_DIR/com.vozdelpueblo.tunnel.plist"

sed "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    webapp/deploy/com.vozdelpueblo.webapp.plist.template \
    > "$WEBAPP_PLIST"
echo "✓ wrote $WEBAPP_PLIST"

sed "s|{{HOME}}|$HOME_DIR|g" \
    webapp/deploy/com.vozdelpueblo.tunnel.plist.template \
    > "$TUNNEL_PLIST"
echo "✓ wrote $TUNNEL_PLIST"

# ---------- 5. Final guidance ----------
echo ""
echo "━━━ 5/5  Next manual steps ━━━"
cat <<NEXT

  1) Fill in .env with your secrets:
     open -t "$PROJECT_DIR/.env"

  2) Cloudflare Tunnel — one-time auth and tunnel creation:
     cloudflared tunnel login
        (opens a browser, pick the domain you own)
     cloudflared tunnel create voz-del-pueblo
     cloudflared tunnel route dns voz-del-pueblo voz.<your-domain>

     Then write the tunnel routing config:
     cat > ~/.cloudflared/config.yml <<EOF
     tunnel: voz-del-pueblo
     credentials-file: ~/.cloudflared/<TUNNEL-UUID>.json
     ingress:
       - hostname: voz.<your-domain>
         service: http://localhost:8000
       - service: http_status:404
     EOF

     IF you DON'T own a domain: use the quick-tunnel variant. Skip steps
     2 entirely, run the webapp service from the next step, and start the
     tunnel manually each time with:
        cloudflared tunnel --url http://localhost:8000
     It'll print a *.trycloudflare.com URL. New URL each restart.

  3) Load the launchd services (will start now AND on every boot):
     launchctl unload "$WEBAPP_PLIST" 2>/dev/null || true
     launchctl load   "$WEBAPP_PLIST"
     launchctl unload "$TUNNEL_PLIST" 2>/dev/null || true
     launchctl load   "$TUNNEL_PLIST"

  4) Confirm they're up:
     launchctl list | grep vozdelpueblo
     curl -s http://localhost:8000/health
     tail -f logs/webapp.stdout.log

  5) Open in your phone:
     If named tunnel: https://voz.<your-domain>
     If quick tunnel: the *.trycloudflare.com URL it printed.
     Username + password from .env.

NEXT
