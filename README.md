# Aven — Rockchip-based AI Assistant

A voice-assistant pipeline built as **microservices** so each stage can run on
its own Rock 5C and use the **NPU**. Right now only the **LLM** and **TTS**
stages are wired up; STT (whisper) and wakeword (openWakeWord) are present as
submodules for later.

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

Every service reads the **single `config.yaml`** at the repo root, so there are
no hardcoded IPs in the code. Change a host/port once, everywhere picks it up.

## Requirements

- [UV](https://docs.astral.sh/uv/) (already installed)
- Submodules checked out: `git submodule update --init --recursive`

## 1. LLM node (main Rock 5C — 100.108.158.94)

rkllama needs **Python 3.12** (its `rknn-toolkit-lite2` wheels stop at cp312),
so it is pinned explicitly. The orchestrator is a separate, lightweight UV
project that only talks to rkllama over HTTP.

```bash
# a) rkllama NPU backend — its own venv inside the submodule
cd LLM/rkllama
uv run --python 3.12 rkllama_server          # serves http://0.0.0.0:8080

# b) one-time: pull the model (in a second shell, server must be running)
cd LLM/rkllama
uv run --python 3.12 rkllama_client pull \
  c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1/Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm/qwen2.5-3b
# → the local model becomes "qwen2.5-3b" (matches llm.model in config.yaml)

# c) the orchestrator
cd LLM
uv sync
uv run python llm_server.py                  # serves ws://0.0.0.0:8765
```

## 2. TTS node (second Rock 5C — 100.113.61.126)

`paroli-server` is a C++ NPU engine. `setup_paroli.sh` builds it and fetches a
voice in one shot (idempotent; prompts for sudo once for apt + the NPU runtime):

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

## 3. Connector (test client, e.g. your laptop)

```bash
uv sync                 # add `--extra audio` for PyAudio; otherwise falls back to aplay
uv run python connector.py
# type a prompt; commands: /clear, /help, exit
```

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
| `STT/whisper`          | (future STT) | present, unused   |
| `wakeword/openWakeWord`| (future)     | present, unused   |

## Hardware

- 1–2 Rock 5C (lite)
- Microphone + speakers (for the full voice loop later)
