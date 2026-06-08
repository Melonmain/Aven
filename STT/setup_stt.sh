#!/usr/bin/env bash
#
# setup_stt.sh — set up the STT node (faster-whisper, CPU). No apt, no NPU:
# everything is pip wheels. Syncs the venv and pre-downloads the Whisper model
# so the first request isn't slow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bold() { printf "\033[1m== %s\033[0m\n" "$*"; }

bold "1/2 sync uv project (faster-whisper + deps)"
( cd "$SCRIPT_DIR" && uv sync )

bold "2/2 pre-download the Whisper model"
( cd "$SCRIPT_DIR" && uv run python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent))
from config import load_config
from faster_whisper import WhisperModel
m = load_config()['stt']
print('fetching', m['model'], '(cpu/'+m['compute_type']+')')
WhisperModel(m['model'], device='cpu', compute_type=m['compute_type'])
print('ready')
" )

echo
bold "Done."
cat <<EOF
Run the STT node:
  cd $SCRIPT_DIR && uv run python stt_server.py

Verify with a 16-bit mono WAV:
  cd $SCRIPT_DIR && uv run python test_stt.py --host 127.0.0.1 --wav /path/to/speech.wav
EOF
