#!/usr/bin/env bash
#
# setup_kokoro.sh — build kokoro-server (Kokoro-82M) with RK3588 NPU support on a
# Debian-family aarch64 board. Idempotent: re-running skips anything already done.
#
# Works on both Aven boards even though they differ:
#   * Debian/Armbian trixie : installs the packaged libdrogon-dev
#   * Debian bookworm        : Drogon isn't packaged, so it's built from source
#
# IMPORTANT — models are NOT downloadable. Kokoro's encoder/har/decoder ONNX,
# the RKNN decoder, and the voices_npy/ must be produced on an x86 host with
# `python3 build.py` (needs PyTorch + rknn-toolkit2 + the Kokoro-82M weights),
# then copied to:  TTSV2/kokoro-server/build/models/
#     models/onnx/{kokoro_encoder.onnx,har_generator.onnx,kokoro_decoder.rknn}
#     models/voices_npy/*.npy
#     models/config.json        (the Kokoro-82M/config.json vocab)
# This script builds the server binary and tells you if the models are missing.
#
# Everything lands under TTSV2/kokoro-server/build/ (gitignored), so it never
# dirties the git tree.
set -euo pipefail

# --- config -----------------------------------------------------------------
ORT_VER="1.14.1"
DROGON_TAG="v1.9.1"                    # used only when building Drogon from source

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOKORO_DIR="$SCRIPT_DIR/kokoro-server"
BUILD_DIR="$KOKORO_DIR/build"
DEPS_DIR="$BUILD_DIR/deps"
MODEL_DIR="$BUILD_DIR/models"
ORT_ROOT="$DEPS_DIR/onnxruntime-linux-aarch64-$ORT_VER"

bold() { printf "\033[1m== %s\033[0m\n" "$*"; }
ok()   { printf "\033[32m   %s\033[0m\n" "$*"; }
warn() { printf "\033[33m   %s\033[0m\n" "$*"; }

if [ ! -f "$KOKORO_DIR/CMakeLists.txt" ]; then
  echo "ERROR: kokoro-server submodule not found at $KOKORO_DIR" >&2
  echo "       run: git submodule update --init --recursive" >&2
  exit 1
fi
# Make sure the nested misaki-cpp submodule is present.
git -C "$KOKORO_DIR" submodule update --init --recursive >/dev/null 2>&1 || true
mkdir -p "$DEPS_DIR" "$MODEL_DIR"

# --- 1. system packages -----------------------------------------------------
bold "1/6 apt dependencies"
PKGS=(build-essential g++ cmake pkg-config git wget
      libfmt-dev libspdlog-dev nlohmann-json3-dev
      libopenblas-dev libespeak-ng-dev espeak-ng-data
      libsoxr-dev libopus-dev libopusenc-dev libogg-dev
      libjsoncpp-dev libssl-dev uuid-dev zlib1g-dev libc-ares-dev libbrotli-dev)

if apt-cache show libdrogon-dev >/dev/null 2>&1; then
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
  bold "1b/6 Drogon $DROGON_TAG from source"
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
bold "2/6 onnxruntime $ORT_VER (aarch64)"
if [ -d "$ORT_ROOT" ]; then
  ok "present: $ORT_ROOT"
else
  url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-aarch64-$ORT_VER.tgz"
  wget -qO "$DEPS_DIR/ort.tgz" "$url"
  tar -xzf "$DEPS_DIR/ort.tgz" -C "$DEPS_DIR"
  rm -f "$DEPS_DIR/ort.tgz"
  ok "installed: $ORT_ROOT"
fi

# --- 3. rknn runtime (NPU) --------------------------------------------------
bold "3/6 rknn runtime (librknnrt.so + header)"
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

# --- 4. build kokoro-server -------------------------------------------------
bold "4/6 build kokoro-server (USE_RKNN=ON)"
# kokoro links -lcblas; OpenBLAS carries the CBLAS symbols but Debian doesn't
# ship a libcblas.so link name, so shim one pointing at libopenblas (the .so's
# soname is libopenblas.so.0, so nothing extra is needed at runtime).
BLAS_SO="$(ls /usr/lib/*/libopenblas.so 2>/dev/null | head -1)"
SHIM_DIR="$DEPS_DIR/cblas-shim"
mkdir -p "$SHIM_DIR"
[ -n "$BLAS_SO" ] && ln -sfn "$BLAS_SO" "$SHIM_DIR/libcblas.so"
cmake -S "$KOKORO_DIR" -B "$BUILD_DIR" \
  -DUSE_RKNN=ON -DBUILD_SERVER=ON -DBUILD_CLI=ON \
  -DORT_ROOT="$ORT_ROOT" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$SHIM_DIR" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"
ok "built: $BUILD_DIR/kokoro-server"

# --- 5. launcher ------------------------------------------------------------
bold "5/6 run wrapper"
LAUNCHER="$BUILD_DIR/run-kokoro-server.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Auto-generated by setup_kokoro.sh — starts the Kokoro NPU TTS engine.
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="\$HERE/deps/onnxruntime-linux-aarch64-$ORT_VER/lib:/usr/local/lib:\${LD_LIBRARY_PATH:-}"
exec "\$HERE/kokoro-server" \\
  --encoder    "\$HERE/models/onnx/kokoro_encoder.onnx" \\
  --har-gen    "\$HERE/models/onnx/har_generator.onnx" \\
  --decoder    "\$HERE/models/onnx/kokoro_decoder.rknn" \\
  --vocab      "\$HERE/models/config.json" \\
  --voices-dir "\$HERE/models/voices_npy" \\
  --ip 0.0.0.0 --port 8848 "\$@"
EOF
chmod +x "$LAUNCHER"
ok "wrote: $LAUNCHER"

# --- 6. model check ---------------------------------------------------------
bold "6/6 models"
missing=0
for f in onnx/kokoro_encoder.onnx onnx/har_generator.onnx onnx/kokoro_decoder.rknn \
         config.json; do
  [ -s "$MODEL_DIR/$f" ] || { warn "missing: models/$f"; missing=1; }
done
[ -d "$MODEL_DIR/voices_npy" ] && ls "$MODEL_DIR"/voices_npy/*.npy >/dev/null 2>&1 \
  || { warn "missing: models/voices_npy/*.npy"; missing=1; }

echo
bold "Done."
if [ "$missing" = 1 ]; then
  cat <<EOF
The server is built, but the Kokoro models are not in place yet. Generate them
on an x86 host (PyTorch + rknn-toolkit2 + Kokoro-82M weights):

  git clone --recurse-submodules https://github.com/marty1885/kokoro-server
  cd kokoro-server   # place Kokoro-82M/ and kokoro-src/ here per its README
  python3 build.py   # -> onnx/  and  voices_npy/

Then copy onto this board into $MODEL_DIR :
  models/onnx/{kokoro_encoder.onnx,har_generator.onnx,kokoro_decoder.rknn}
  models/voices_npy/*.npy
  models/config.json     (the Kokoro-82M/config.json)
EOF
else
  cat <<EOF
Run the Kokoro NPU TTS engine:
  $LAUNCHER

Then, in another shell, the Aven TTSV2 node + smoke test:
  cd $SCRIPT_DIR && uv sync && uv run python voice_server.py
  cd $SCRIPT_DIR && uv run python test_tts.py --host 127.0.0.1
EOF
fi
