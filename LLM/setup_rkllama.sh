#!/usr/bin/env bash
#
# setup_rkllama.sh — install the rkllama submodule into a local venv and make it
# runnable on the LLM board (Rock 5C, RK3588 NPU).
#
# Why a script instead of `uv run`: rkllama's pyproject declares its
# rknn-toolkit-lite2 wheel with a RELATIVE url (`@ file:./src/...`), which uv
# refuses to parse ("relative path without a working directory"). This rewrites
# those urls to absolute *only for the install*, then restores pyproject so the
# submodule stays clean. We run via the venv's entry point (not `uv run`, which
# would re-sync and trip the same bug).
#
# The venv is created at LLM/rkllama/venv (already in rkllama's .gitignore), so
# it never dirties the tree. First run downloads torch + friends (large).
set -euo pipefail

PYTHON_VER="3.12"     # rknn-toolkit-lite2 wheels stop at cp312
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RKLLAMA_DIR="$SCRIPT_DIR/rkllama"
VENV="$RKLLAMA_DIR/venv"
PP="$RKLLAMA_DIR/pyproject.toml"

bold() { printf "\033[1m== %s\033[0m\n" "$*"; }
ok()   { printf "\033[32m   %s\033[0m\n" "$*"; }

if [ ! -f "$PP" ]; then
  echo "ERROR: rkllama submodule not found at $RKLLAMA_DIR" >&2
  echo "       run: git submodule update --init --recursive" >&2
  exit 1
fi

bold "1/3 create venv (Python $PYTHON_VER)"
uv venv --python "$PYTHON_VER" "$VENV"

bold "2/3 install rkllama + deps (first run downloads torch etc. — large)"
cp "$PP" "$PP.aven.bak"
trap 'mv -f "$PP.aven.bak" "$PP" 2>/dev/null || true' EXIT
# absolutize the relative wheel urls so uv can parse the metadata
sed -i "s#@ file:./src/rkllama/lib/#@ file://$RKLLAMA_DIR/src/rkllama/lib/#g" "$PP"
uv pip install --python "$VENV" "$RKLLAMA_DIR"
mv -f "$PP.aven.bak" "$PP"; trap - EXIT
ok "installed rkllama into $VENV (pyproject restored)"

bold "3/3 done"
cat <<EOF
Start the LLM backend (runs the NPU OpenAI server on :8080):
  cd $RKLLAMA_DIR && ./venv/bin/rkllama_server

One-time model pull (server must be running, in another shell):
  cd $RKLLAMA_DIR && ./venv/bin/rkllama_client pull \\
    c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1/Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm/qwen2.5-3b

Then the orchestrator:
  cd $SCRIPT_DIR && uv sync && uv run python llm_server.py
EOF
