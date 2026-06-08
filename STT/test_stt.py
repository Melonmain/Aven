#!/usr/bin/env python3
"""Smoke test for the STT node — sends a WAV's audio and prints the transcript.

Run (with stt_server.py running):
    cd STT && uv run python test_stt.py --wav /path/to/speech.wav
    cd STT && uv run python test_stt.py --host 127.0.0.1 --wav sample.wav
"""

import argparse
import json
import pathlib
import sys
import time
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import service_addr  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def main():
    host, port = service_addr("stt")
    parser = argparse.ArgumentParser(description="STT smoke test")
    parser.add_argument("--wav", required=True, help="16-bit mono WAV to transcribe")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    wav_path = pathlib.Path(args.wav)
    if not wav_path.exists():
        print(f"{RED}FAIL:{RESET} no such file: {wav_path}")
        return 1
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            print(f"{RED}FAIL:{RESET} need 16-bit mono WAV "
                  f"(got width={wf.getsampwidth()} ch={wf.getnchannels()})")
            return 1
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError:
        print(f"{RED}FAIL:{RESET} missing 'websockets' (run: uv sync)")
        return 1

    url = f"ws://{args.host}:{args.port}"
    print(f"{YELLOW}connecting:{RESET} {url}  ({len(pcm)} bytes @ {rate} Hz)")
    try:
        ws = connect(url, max_size=None, open_timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}FAIL:{RESET} cannot reach STT node at {url} ({exc})")
        print("      start it with: uv run python stt_server.py")
        return 1

    with ws:
        ws.send(json.dumps({"command": "config", "sample_rate": rate}))
        for i in range(0, len(pcm), 32000):       # ~1s chunks
            ws.send(pcm[i:i + 32000])
        t0 = time.perf_counter()
        ws.send(json.dumps({"command": "transcribe"}))
        result = json.loads(ws.recv())
    dt = time.perf_counter() - t0

    if result.get("type") != "transcript":
        print(f"{RED}FAIL:{RESET} unexpected reply: {result}")
        return 1
    text = result.get("text", "")
    print(f"{GREEN}PASS:{RESET} ({result.get('seconds')}s audio, lang={result.get('language')}, "
          f"transcribed in {dt:.1f}s)")
    print(f"  transcript: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
