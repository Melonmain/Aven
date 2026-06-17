#!/usr/bin/env bash
# Prime the USB speaker + ALSA dmix mixer at boot.
#
# Why: the Creative Pebble V3 (USB card "V3") can take up to ~2 minutes after a
# cold boot to enumerate and become openable. If the coordinator or Raspotify
# opens the shared dmix `default` device before then, the mixer comes up in a
# bad state and stays silent until something re-opens it (the manual
# `speaker-test` unstick). This waits until `default` actually opens, then does
# one short SILENT open to initialise dmix cleanly — so the first real playback
# just works.
#
# Always exits 0: a missing speaker must never block the rest of the stack
# (the coordinator self-heals / drops audio when no speaker is present).

CARD="${1:-V3}"
TIMEOUT="${2:-180}"   # seconds to wait for the speaker to become ready

log() { echo "$(date '+%H:%M:%S') prime_speaker: $*"; }

# 1. Wait for the card to appear in the kernel's list.
for i in $(seq 1 "$TIMEOUT"); do
    grep -q "\[\s*${CARD}\s*\]" /proc/asound/cards 2>/dev/null && break
    sleep 1
done
if ! grep -q "\[\s*${CARD}\s*\]" /proc/asound/cards 2>/dev/null; then
    log "card '${CARD}' not present after ${TIMEOUT}s — skipping (no speaker)"
    exit 0
fi
log "card '${CARD}' present; priming dmix on 'default'…"

# 2. Once present, the device may still be settling. Retry a short SILENT open
#    of the shared `default` (dmix) device until it succeeds or we hit timeout.
deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if aplay -D default -q -f S16_LE -r 48000 -c 2 -d 1 /dev/zero >/dev/null 2>&1; then
        log "dmix primed OK"
        exit 0
    fi
    sleep 2
done

log "could not open 'default' within ${TIMEOUT}s — leaving for runtime self-heal"
exit 0
