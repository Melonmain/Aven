#!/usr/bin/env bash
#
# setup_wakeword.sh — set up the wakeword node (openWakeWord, CPU/ONNX). No apt,
# no NPU. Syncs the venv and downloads the pretrained wake-phrase model.
#
# For live mic capture add the extra (needs system PortAudio):
#   sudo apt install -y libportaudio2 && (cd wakeword && uv sync --extra mic)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bold() { printf "\033[1m== %s\033[0m\n" "$*"; }

bold "1/2 sync uv project (openwakeword + onnxruntime)"
( cd "$SCRIPT_DIR" && uv sync )

bold "2/2 download the pretrained wakeword model"
( cd "$SCRIPT_DIR" && uv run python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent))
from config import load_config
from openwakeword.utils import download_models
name = load_config()['wakeword']['model']
print('fetching', name)
download_models([name])
print('ready')
" )

echo
bold "Done."
cat <<EOF
Run the wakeword listener over a WAV (no mic needed):
  cd $SCRIPT_DIR && uv run python wakeword_listener.py --wav /path/to/clip.wav

Live mic (needs PortAudio + the mic extra):
  sudo apt install -y libportaudio2
  cd $SCRIPT_DIR && uv sync --extra mic && uv run python wakeword_listener.py
EOF
