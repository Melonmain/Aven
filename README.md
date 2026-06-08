# Aven — Rockchip-based AI Assistant

A voice-assistant pipeline built as **microservices** so each stage can run on
its own Rock 5C and use the **NPU** (LLM, TTS) or the **CPU** (STT, wakeword).
The **LLM** and **TTS** stages are wired end-to-end; the **STT** (faster-whisper)
and **wakeword** (openWakeWord) stages exist as standalone CPU services and are
being wired into the full voice loop:

```
  mic ─▶ wakeword (CPU) ─▶ STT (CPU) ─▶ LLM (NPU) ─▶ TTS (NPU/CPU) ─▶ speaker
       "hey jarvis"      faster-whisper   rkllama      paroli/kokoro/piper
```

```
  connector (laptop)
        │  ws://…:8765   {"text": "..."}
        ▼
  LLM node  (main Rock 5C, 100.108.158.94)
    ├── rkllama  (NPU LLM, http://127.0.0.1:8080)   ← Qwen2.5-3B
    └── llm_server.py  streams tokens → clauses
        │  ws://…:8766   {"text": "a clause"}
        ▼
  TTS node  (second Rock 5C, 100.113.61.126)
    ├── voice_server.py
    └── paroli-server  (NPU VITS, ws://127.0.0.1:8848)
        │  raw PCM relayed back up the chain
        ▼
  connector plays the audio as it arrives
```

> ⚠️ The TTS node runs on a **separate** Rock 5C. Don't run the LLM and TTS
> servers on the same board at the same time (they each want the NPU).

## Layout

| Path                 | What runs there                          | UV project        |
|----------------------|------------------------------------------|-------------------|
| `config.yaml`        | Shared topology + settings (edit this)   | —                 |
| `config.py`          | Tiny loader imported by every service    | —                 |
| `connector.py`       | Test client (laptop)                     | `./pyproject.toml`|
| `LLM/llm_server.py`  | LLM orchestrator (main Rock 5C)          | `LLM/`            |
| `LLM/rkllama/`       | NPU LLM backend (submodule)              | `LLM/rkllama/`    |
| `TTS/voice_server.py`| TTS node (second Rock 5C)                | `TTS/`            |
| `TTS/paroli/`        | NPU TTS backend (submodule, C++)         | built separately  |
| `STT/stt_server.py`  | STT node (faster-whisper, CPU)           | `STT/`            |
| `wakeword/wakeword_listener.py` | Wakeword (openWakeWord, CPU)  | `wakeword/`       |
| `coordinator/coordinator.py` | Full voice loop client (mic→…→speaker) | `coordinator/` |

Every service reads the **single `config.yaml`** at the repo root, so there are
no hardcoded IPs in the code. Change a host/port once, everywhere picks it up.

## Requirements

- [UV](https://docs.astral.sh/uv/) (already installed)
- Submodules checked out: `git submodule update --init --recursive`

## 1. LLM node (main Rock 5C — 100.108.158.94)

rkllama needs **Python 3.12** (its `rknn-toolkit-lite2` wheels stop at cp312).
It can't be launched with `uv run` directly — rkllama's pyproject declares its
NPU wheel with a relative `file:./…` URL that uv refuses to parse — so
`setup_rkllama.sh` installs it into a local venv (rewriting that URL to absolute
just for the install) and you run it from that venv. The orchestrator is a
separate, lightweight UV project that only talks to rkllama over HTTP.

```bash
# a) one-time: install rkllama into LLM/rkllama/venv (downloads torch etc. — large)
bash LLM/setup_rkllama.sh

# b) rkllama NPU backend (serves http://0.0.0.0:8080)
cd LLM/rkllama && ./venv/bin/rkllama_server

# c) one-time: pull the model (in a second shell, server must be running)
cd LLM/rkllama && ./venv/bin/rkllama_client pull \
  c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1/Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm/qwen2.5-3b
# → the local model becomes "qwen2.5-3b" (matches llm.model in config.yaml)

# d) the orchestrator
cd LLM
uv sync
uv run python llm_server.py                  # serves ws://0.0.0.0:8765
```

## 2. TTS node (second Rock 5C — 100.113.61.126)

`paroli-server` is a C++ NPU engine. `setup_paroli.sh` builds it and fetches a
voice in one shot (idempotent; prompts for sudo once for apt + the NPU runtime):

The script auto-adapts to the board's OS: on trixie it installs the packaged
`libdrogon-dev`; on Debian **bookworm** (where Drogon isn't packaged) it builds
Drogon from source automatically (first run takes a few minutes longer).

```bash
# a) one-time build + voice download (default voice: ljspeech)
bash TTS/setup_paroli.sh

# b) start the NPU engine (wrapper sets LD_LIBRARY_PATH + model paths for you)
TTS/paroli/build/run-paroli-server.sh         # serves ws://127.0.0.1:8848

# c) the TTS WebSocket node
cd TTS
uv sync
uv run python voice_server.py                # serves ws://0.0.0.0:8766

# d) verify (writes test_output.wav)
uv run python test_tts.py --host 127.0.0.1
```

### 2b. TTSV2 — kokoro-server (improved voice, optional)

[`TTSV2/`](TTSV2/) is a drop-in alternative to paroli backed by
[kokoro-server](https://github.com/marty1885/kokoro-server) (Kokoro-82M). It
listens on the **same** `services.tts` port (8766) and speaks the same protocol,
so the LLM node reaches it unchanged — run **either** paroli **or** kokoro.

```bash
# a) build the C++ engine (auto-handles trixie vs bookworm Drogon)
bash TTSV2/setup_kokoro.sh
```

⚠️ **Models are not downloadable.** kokoro's encoder/har/decoder ONNX, the RKNN
decoder, and `voices_npy/` must be generated on an **x86 host** with
`python3 build.py` (PyTorch + rknn-toolkit2 + Kokoro-82M weights), then copied to
`TTSV2/kokoro-server/build/models/` (`onnx/`, `voices_npy/`, `config.json`). The
setup script tells you exactly what's missing.

```bash
# b) once models are in place
TTSV2/kokoro-server/build/run-kokoro-server.sh   # serves ws://127.0.0.1:8848
cd TTSV2 && uv sync && uv run python voice_server.py
uv run python test_tts.py --host 127.0.0.1
```

Voice/speed/rate are set under `ttsv2:` in [`config.yaml`](config.yaml).

### 2c. TTSV3 — Piper (CPU, simplest, optional)

[`TTSV3/`](TTSV3/) is the simplest TTS: vanilla [Piper](https://github.com/OHF-Voice/piper1-gpl)
running **in-process on the CPU** (no submodule, no C++ build, no NPU). It listens
on the same `services.tts` port (8766) and speaks the same protocol, so it's a
drop-in for paroli/kokoro — handy as a fallback or on a board without a free NPU.

```bash
# a) sync the venv and download a voice (default: en_US-lessac-medium)
bash TTSV3/setup_piper.sh

# b) run it (synthesizes in-process — no separate backend)
cd TTSV3 && uv run python voice_server.py

# c) verify (writes test_output.wav)
uv run python test_tts.py --host 127.0.0.1
```

Voice/length-scale/speaker are set under `ttsv3:` in [`config.yaml`](config.yaml);
browse voices at [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

## 3. Voice input — STT + wakeword (CPU)

These two run on the **CPU** (no NPU), so they can live on any board — including
the LLM board alongside rkllama. They're standalone services today; wiring them
into the full mic→LLM loop is the next step.

### 3a. STT — faster-whisper

[`STT/`](STT/) transcribes speech to text with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8 on
CPU — much lighter than `openai-whisper`; the `STT/whisper` submodule stays as
upstream reference). A client streams an utterance as PCM and gets back text.

```bash
bash STT/setup_stt.sh                       # uv sync + pre-download the model
cd STT && uv run python stt_server.py        # serves ws://0.0.0.0:8767
# verify with any 16-bit mono WAV:
uv run python test_stt.py --host 127.0.0.1 --wav /path/to/speech.wav
```

Model/language/compute-type live under `stt:` in [`config.yaml`](config.yaml)
(default `base.en`, int8). Audio at any rate is resampled to 16 kHz.

### 3b. Wakeword — openWakeWord

[`wakeword/`](wakeword/) listens for a wake phrase with
[openWakeWord](https://github.com/dscripka/openWakeWord) (ONNX on CPU). The
default phrase is the pretrained **"hey jarvis"** model, downloaded on setup.

```bash
bash wakeword/setup_wakeword.sh             # uv sync + download the model
# test over a WAV (no mic needed):
cd wakeword && uv run python wakeword_listener.py --wav /path/to/clip.wav
# live mic (needs PortAudio):
sudo apt install -y libportaudio2
uv sync --extra mic && uv run python wakeword_listener.py
```

Phrase/threshold live under `wakeword:` in [`config.yaml`](config.yaml); other
phrases (`alexa`, `hey_mycroft`, …) are downloaded on demand.

## 4. Connector (test client, e.g. your laptop)

```bash
uv sync                 # add `--extra audio` for PyAudio; otherwise falls back to aplay
uv run python connector.py
# type a prompt; commands: /clear, /help, exit
```

## 5. Coordinator — the full voice loop

[`coordinator/`](coordinator/) is the one client that combines every stage. Run
it on the device with the **mic + speakers**; it chains them and reaches the
remote stages over WebSocket (STT `:8767`, LLM `:8765`, which itself drives TTS):

```
  mic ─▶ wakeword (local CPU) ─▶ record ─▶ STT ─▶ LLM ─▶ play reply ─▶ (loop)
```

```bash
cd coordinator && uv sync
sudo apt install -y libportaudio2          # mic + speaker capture/playback

# full voice loop: say the wake word ("hey jarvis"), then your request
uv run python coordinator.py

# testable without a mic:
uv run python coordinator.py --text "What is the capital of France?"
uv run python coordinator.py --wav clip.wav        # STT a WAV, then ask the LLM
uv run python coordinator.py --wav clip.wav --no-audio   # headless (print only)
```

Prereqs: the **STT** node (`STT/stt_server.py`) and the **LLM** node
(`LLM/llm_server.py`, with its TTS node up) must be running. Wakeword phrase and
recording behaviour (silence timeout, etc.) live under `wakeword:` /
`coordinator:` in [`config.yaml`](config.yaml).

## Configuration

All knobs live in [`config.yaml`](config.yaml): the host/port of each service,
the rkllama model name, the system prompt, and TTS sample rate / speaker.
Every server still accepts CLI flags (`--host`, `--port`, `--tts-host`, …) that
override the config for one-off runs — e.g. to colocate TTS on the LLM board for
a quick test:

```bash
cd LLM && uv run python llm_server.py --tts-host 127.0.0.1
```

## Submodules

| Submodule              | Used by      | Status            |
|------------------------|--------------|-------------------|
| `LLM/rkllama`          | LLM node     | active            |
| `TTS/paroli`           | TTS node     | active            |
| `TTSV2/kokoro-server`  | TTS node (v2)| active (optional) |
| `STT/whisper`          | STT          | reference (STT uses the `faster-whisper` pip pkg) |
| `wakeword/openWakeWord`| Wakeword     | reference (wakeword uses the `openwakeword` pip pkg) |

## Hardware

- 1–2 Rock 5C (lite)
- Microphone + speakers (for the full voice loop later)
