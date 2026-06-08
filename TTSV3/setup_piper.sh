#!/usr/bin/env bash
#
# setup_piper.sh — set up the TTSV3 node (vanilla Piper, CPU/in-process).
# No apt, no C++, no NPU: piper-tts is a pip package that bundles its own
# onnxruntime + espeak-ng data. Just sync the venv and download a voice.
#
# Usage:
#   bash TTSV3/setup_piper.sh                 # default voice from config.yaml
#   VOICE=en_GB-alba-medium bash TTSV3/setup_piper.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICES_DIR="$SCRIPT_DIR/voices"

bold() { printf "\033[1m== %s\033[0m\n" "$*"; }
ok()   { printf "\033[32m   %s\033[0m\n" "$*"; }

bold "1/2 sync uv project (piper-tts + websockets)"
( cd "$SCRIPT_DIR" && uv sync )
ok "venv ready"

# Voice: $VOICE env overrides config.yaml's ttsv3.voice.
VOICE="${VOICE:-$(cd "$SCRIPT_DIR" && uv run python -c \
  'import sys,pathlib; sys.path.insert(0,str(pathlib.Path().resolve().parent)); \
   from config import load_config; print(load_config()["ttsv3"]["voice"])')}"

bold "2/2 download voice: $VOICE"
mkdir -p "$VOICES_DIR"
if [ -s "$VOICES_DIR/$VOICE.onnx" ] && [ -s "$VOICES_DIR/$VOICE.onnx.json" ]; then
  ok "already present: $VOICES_DIR/$VOICE.onnx"
else
  ( cd "$SCRIPT_DIR" && uv run python -m piper.download_voices "$VOICE" \
      --download-dir "$VOICES_DIR" )
  ok "downloaded into $VOICES_DIR"
fi

echo
bold "Done."
cat <<EOF
Run the Piper TTS node:
  cd $SCRIPT_DIR && uv run python voice_server.py

Then verify (writes test_output.wav):
  cd $SCRIPT_DIR && uv run python test_tts.py --host 127.0.0.1
EOF
