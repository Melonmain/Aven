#!/usr/bin/env bash
#
# setup_paroli.sh — build paroli-server with RK3588 NPU support and fetch a
# streaming voice, on a Debian-family aarch64 board. Idempotent: re-running
# skips anything already done.
#
# Works on both Aven boards even though they differ:
#   * Debian/Armbian trixie : installs the packaged libdrogon-dev
#   * Debian bookworm        : Drogon isn't packaged, so it's built from source
# The script auto-detects which case applies.
#
# Usage:
#   bash TTS/setup_paroli.sh            # full setup (will call sudo for apt + libs)
#   VOICE=ljspeech bash TTS/setup_paroli.sh
#
# Everything lands under TTS/paroli/build/ (which the submodule already
# .gitignores), so it never dirties the git tree.
set -euo pipefail

# --- config -----------------------------------------------------------------
ORT_VER="1.14.1"
PIPER_PHON_TAG="2023.11.14-4"
DROGON_TAG="v1.9.1"                    # used only when building Drogon from source
VOICE="${VOICE:-ljspeech}"            # HF subfolder under marty1885/streaming-piper
HF_REPO="marty1885/streaming-piper"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAROLI_DIR="$SCRIPT_DIR/paroli"
BUILD_DIR="$PAROLI_DIR/build"
DEPS_DIR="$BUILD_DIR/deps"
MODEL_DIR="$BUILD_DIR/models/$VOICE"

ORT_ROOT="$DEPS_DIR/onnxruntime-linux-aarch64-$ORT_VER"
PIPER_ROOT="$DEPS_DIR/piper_phonemize"

bold() { printf "\033[1m== %s\033[0m\n" "$*"; }
ok()   { printf "\033[32m   %s\033[0m\n" "$*"; }

if [ ! -f "$PAROLI_DIR/CMakeLists.txt" ]; then
  echo "ERROR: paroli submodule not found at $PAROLI_DIR" >&2
  echo "       run: git submodule update --init --recursive" >&2
  exit 1
fi
mkdir -p "$DEPS_DIR"

# --- 1. system packages -----------------------------------------------------
bold "1/7 apt dependencies"
# Common deps available on both trixie and bookworm.
PKGS=(build-essential g++ cmake pkg-config git wget
      libspdlog-dev libfmt-dev libsoxr-dev libopusenc-dev libopus-dev libogg-dev
      xtensor-dev libjsoncpp-dev libssl-dev uuid-dev zlib1g-dev
      libc-ares-dev libbrotli-dev)   # last two: for Drogon/Trantor

if apt-cache show libdrogon-dev >/dev/null 2>&1; then
  # Distro ships Drogon (trixie+). Its CMake config marks the DB/codec backends
  # REQUIRED, so install them alongside the package.
  PKGS+=(libdrogon-dev libpq-dev libsqlite3-dev default-libmysqlclient-dev
         libhiredis-dev libyaml-cpp-dev)
  DROGON_FROM_SOURCE=0
else
  ok "libdrogon-dev not in apt (e.g. Debian bookworm) -> will build from source"
  DROGON_FROM_SOURCE=1
fi
sudo apt-get update
sudo apt-get install -y "${PKGS[@]}"

# --- 1b. Drogon from source (only where it isn't packaged) ------------------
if [ "$DROGON_FROM_SOURCE" = 1 ]; then
  bold "1b/7 Drogon $DROGON_TAG from source"
  if ls /usr/local/lib*/cmake/Drogon/DrogonConfig.cmake >/dev/null 2>&1; then
    ok "drogon already installed in /usr/local"
  else
    DROGON_SRC="$DEPS_DIR/drogon"
    [ -d "$DROGON_SRC/.git" ] || git clone --depth 1 --branch "$DROGON_TAG" \
      --recurse-submodules https://github.com/drogonframework/drogon "$DROGON_SRC"
    cmake -S "$DROGON_SRC" -B "$DROGON_SRC/build" \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF
    cmake --build "$DROGON_SRC/build" -j"$(nproc)"
    sudo cmake --install "$DROGON_SRC/build"
    sudo ldconfig
    ok "installed drogon -> /usr/local"
  fi
fi

# --- 2. onnxruntime ---------------------------------------------------------
bold "2/7 onnxruntime $ORT_VER (aarch64)"
if [ -d "$ORT_ROOT" ]; then
  ok "present: $ORT_ROOT"
else
  url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-aarch64-$ORT_VER.tgz"
  wget -qO "$DEPS_DIR/ort.tgz" "$url"
  tar -xzf "$DEPS_DIR/ort.tgz" -C "$DEPS_DIR"
  rm -f "$DEPS_DIR/ort.tgz"
  ok "installed: $ORT_ROOT"
fi

# --- 3. piper-phonemize -----------------------------------------------------
bold "3/7 piper-phonemize $PIPER_PHON_TAG"
if [ -d "$PIPER_ROOT" ] && [ -d "$PIPER_ROOT/share/espeak-ng-data" ]; then
  ok "present: $PIPER_ROOT"
else
  url="https://github.com/rhasspy/piper-phonemize/releases/download/$PIPER_PHON_TAG/piper-phonemize_linux_aarch64.tar.gz"
  wget -qO "$DEPS_DIR/pp.tgz" "$url"
  tar -xzf "$DEPS_DIR/pp.tgz" -C "$DEPS_DIR"   # extracts to piper_phonemize/
  rm -f "$DEPS_DIR/pp.tgz"
  ok "installed: $PIPER_ROOT"
fi

# --- 4. rknn runtime (NPU) --------------------------------------------------
bold "4/7 rknn runtime (librknnrt.so + header)"
RKNN_BASE="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api"
if [ -f /usr/lib/librknnrt.so ] && [ -f /usr/include/rknn_api.h ]; then
  ok "already installed in /usr"
else
  wget -qO "$DEPS_DIR/librknnrt.so" "$RKNN_BASE/aarch64/librknnrt.so"
  wget -qO "$DEPS_DIR/rknn_api.h"   "$RKNN_BASE/include/rknn_api.h"
  sudo cp "$DEPS_DIR/librknnrt.so" /usr/lib/
  sudo cp "$DEPS_DIR/rknn_api.h"   /usr/include/
  sudo ldconfig
  ok "installed librknnrt.so -> /usr/lib, rknn_api.h -> /usr/include"
fi

# --- 5. build paroli --------------------------------------------------------
bold "5/7 build paroli (USE_RKNN=ON)"
cmake -S "$PAROLI_DIR" -B "$BUILD_DIR" \
  -DUSE_RKNN=ON \
  -DORT_ROOT="$ORT_ROOT" \
  -DPIPER_PHONEMIZE_ROOT="$PIPER_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"
# espeak-ng-data must sit next to the binary (or pass --espeak_data)
cp -r "$PIPER_ROOT/share/espeak-ng-data" "$BUILD_DIR/" 2>/dev/null || true
ok "built: $BUILD_DIR/paroli-server"

# --- 6. download the voice --------------------------------------------------
bold "6/7 voice model: $VOICE"
mkdir -p "$MODEL_DIR"
for f in encoder.onnx decoder.rknn config.json; do
  if [ -s "$MODEL_DIR/$f" ]; then
    ok "present: $f"
  else
    wget -qO "$MODEL_DIR/$f" \
      "https://huggingface.co/$HF_REPO/resolve/main/$VOICE/$f"
    ok "downloaded: $f"
  fi
done

# --- launcher (sets LD_LIBRARY_PATH so the bundled onnxruntime/espeak resolve) -
LAUNCHER="$BUILD_DIR/run-paroli-server.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Auto-generated by setup_paroli.sh — starts the NPU TTS engine.
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="\$HERE/deps/piper_phonemize/lib:\$HERE/deps/onnxruntime-linux-aarch64-$ORT_VER/lib:\${LD_LIBRARY_PATH:-}"
exec "\$HERE/paroli-server" \\
  --encoder "\$HERE/models/$VOICE/encoder.onnx" \\
  --decoder "\$HERE/models/$VOICE/decoder.rknn" \\
  -c "\$HERE/models/$VOICE/config.json" \\
  --espeak_data "\$HERE/espeak-ng-data" \\
  --ip 0.0.0.0 --port 8848 "\$@"
EOF
chmod +x "$LAUNCHER"

echo
bold "Done."
cat <<EOF
Run the NPU TTS engine:
  $LAUNCHER

Then, in another shell, the Aven TTS node + smoke test:
  cd $SCRIPT_DIR && uv sync && uv run python voice_server.py
  cd $SCRIPT_DIR && uv run python test_tts.py --host 127.0.0.1
EOF
