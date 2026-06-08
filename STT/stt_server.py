#!/usr/bin/env python3
"""STT node — speech-to-text with faster-whisper on the CPU.

A WebSocket service: a client streams an utterance as raw PCM, then asks for a
transcription. faster-whisper (CTranslate2) runs the Whisper model on the CPU in
int8, which is the practical choice on a Rock 5C (no NPU, no torch).

Eventually the voice loop is:  wakeword fires -> record mic -> STT (here) ->
llm_server -> TTS. For now this is a standalone, testable service.

Protocol (one utterance):
  client -> node : {"command":"config","sample_rate":N}   (optional, default 16000)
                   <binary PCM frames>  (16-bit signed mono at sample_rate)
                   {"command":"transcribe"}
  node -> client : {"type":"transcript","text":"...","language":"en","seconds":S}
  client -> node : {"command":"reset"}                    (drop buffered audio)

Audio at any sample_rate is resampled to 16 kHz (what Whisper expects).

Run:  python stt_server.py            # uses config.yaml
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import soxr

# Shared config lives at the repo root; this file is one level down (STT/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

_CFG = load_config()
ROCK5C_IP, STT_WS_PORT = service_addr("stt")
_STT = _CFG["stt"]
TARGET_RATE = 16000


def pcm_to_float16k(pcm_bytes, in_rate):
    """int16 PCM at in_rate -> float32 mono at 16 kHz, in [-1, 1]."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if in_rate != TARGET_RATE:
        audio = soxr.resample(audio, in_rate, TARGET_RATE)
    return audio


def make_handler(model, language, beam_size):
    from websockets.exceptions import ConnectionClosed

    def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        print(f"[+] STT client connected: {peer}", flush=True)
        buf = bytearray()
        in_rate = TARGET_RATE
        try:
            for message in conn:
                if isinstance(message, (bytes, bytearray)):
                    buf.extend(message)
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                cmd = data.get("command")
                if cmd == "config":
                    in_rate = int(data.get("sample_rate", TARGET_RATE))
                elif cmd == "reset":
                    buf = bytearray()
                elif cmd == "transcribe":
                    if not buf:
                        conn.send(json.dumps({"type": "transcript", "text": "",
                                              "language": None, "seconds": 0.0}))
                        continue
                    audio = pcm_to_float16k(bytes(buf), in_rate)
                    seconds = len(audio) / TARGET_RATE
                    segments, info = model.transcribe(
                        audio, language=language, beam_size=beam_size)
                    text = "".join(seg.text for seg in segments).strip()
                    print(f"[>] {peer}: ({seconds:.1f}s) -> {text!r}", flush=True)
                    conn.send(json.dumps({
                        "type": "transcript", "text": text,
                        "language": info.language, "seconds": round(seconds, 2),
                    }))
                    buf = bytearray()
        except ConnectionClosed:
            pass
        finally:
            print(f"[-] STT client disconnected: {peer}", flush=True)

    return handler


def main():
    parser = argparse.ArgumentParser(description="STT node (faster-whisper, CPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=STT_WS_PORT)
    parser.add_argument("--model", default=_STT["model"])
    parser.add_argument("--language", default=_STT["language"])
    parser.add_argument("--compute-type", default=_STT["compute_type"])
    parser.add_argument("--beam-size", type=int, default=_STT["beam_size"])
    args = parser.parse_args()

    try:
        from websockets.sync.server import serve
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'websockets'.\033[0m Install: uv sync")
        sys.exit(1)
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper '{args.model}' (cpu/{args.compute_type})…", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    handler = make_handler(model, args.language, args.beam_size)

    print("\033[32mSTT node ready.\033[0m")
    print(f"  Model     : {args.model}  (cpu/{args.compute_type}, beam {args.beam_size})")
    print(f"  Language  : {args.language or 'auto'}")
    print(f"  Listening : ws://{args.host}:{args.port}  (reach me at ws://{ROCK5C_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
