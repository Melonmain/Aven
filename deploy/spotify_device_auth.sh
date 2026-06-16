#!/usr/bin/env bash
#
# One-time Spotify sign-in for the Aven Connect device — WITHOUT the phone app.
#
# librespot (Raspotify) normally only appears in Spotify once a phone activates
# it, and that's lost on every reboot. This uses librespot's browser OAuth flow
# to cache real credentials (credentials.json), after which librespot logs into
# your account on every boot and shows up in the Web API for play_music — no
# phone, survives reboots.
#
# Run it (as root) on the board:   sudo bash deploy/spotify_device_auth.sh
# It prints a URL — open it in ANY browser, log in, authorize. The redirect goes
# to http://127.0.0.1:5588; if you're on another machine, forward the port first:
#     ssh -L 5588:localhost:5588 <user>@<board>
# then open the URL there. Once credentials.json is written, raspotify restarts.
set -euo pipefail

CACHE=/var/cache/raspotify
PORT=5588
NAME="Aven"

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo (cache dir is root-owned)."; exit 1; }

echo "Stopping raspotify…"
systemctl stop raspotify
mkdir -p "$CACHE"

echo "Starting OAuth sign-in for device '$NAME' (redirect port $PORT)…"
echo "──────────────────────────────────────────────────────────────"
# librespot prints the authorize URL, runs a local server on $PORT to catch the
# redirect, then writes credentials.json into the system cache and keeps running.
librespot --name "$NAME" --enable-oauth --oauth-port "$PORT" \
  --cache "$CACHE" --system-cache "$CACHE" \
  --backend alsa --device default --disable-audio-cache &
LPID=$!

# Wait (up to 3 min) for credentials to land, then hand back to raspotify.
for _ in $(seq 1 180); do
  [ -f "$CACHE/credentials.json" ] && break
  kill -0 "$LPID" 2>/dev/null || break
  sleep 1
done
kill "$LPID" 2>/dev/null || true
wait "$LPID" 2>/dev/null || true

if [ -f "$CACHE/credentials.json" ]; then
  echo "──────────────────────────────────────────────────────────────"
  echo "✓ credentials cached at $CACHE/credentials.json"
  systemctl start raspotify
  echo "✓ raspotify restarted — 'Aven' will now auto-connect on every boot."
else
  echo "✗ No credentials.json written (sign-in not completed)."
  systemctl start raspotify
  exit 1
fi
