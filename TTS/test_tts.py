#!/usr/bin/env python3
"""Smoke test for the TTS stage — talks to voice_server.py (which fronts paroli).

No LLM node is involved, so this is safe to run on the TTS board on its own. It
connects to the TTS WebSocket, sends one clause, collects the streamed PCM, and
writes it to a WAV file you can play back.

Run (on the TTS board, with voice_server.py + paroli-server running):
    cd TTS && uv run python test_tts.py
    cd TTS && uv run python test_tts.py --host 127.0.0.1 --text "Hello there."
"""

import argparse
import json
import pathlib
import sys
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import service_addr  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def main():
    host, port = service_addr("tts")
    parser = argparse.ArgumentParser(description="TTS stage smoke test (voice_server)")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--text", default="Hello, this is a text to speech test.")
    parser.add_argument("--out", default="test_output.wav")
    args = parser.parse_args()

    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError:
        print(f"{RED}FAIL:{RESET} missing 'websockets' (run: uv sync)")
        return 1

    url = f"ws://{args.host}:{args.port}"
    print(f"{YELLOW}connecting:{RESET} {url}")
    try:
        ws = connect(url, max_size=None, open_timeout=5)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}FAIL:{RESET} cannot reach TTS node at {url} ({exc})")
        print("      start it with: uv run python voice_server.py")
        return 1

    sample_rate = None
    pcm = bytearray()
    error = None
    with ws:
        ws.send(json.dumps({"text": args.text}))
        for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                pcm.extend(msg)
                continue
            event = json.loads(msg)
            etype = event.get("type")
            if etype == "audio_start":
                sample_rate = event["sample_rate"]
            elif etype == "error":
                error = event.get("message", "unknown error")
            elif etype == "done":
                break

    if error:
        print(f"{RED}FAIL:{RESET} TTS node reported: {error}")
        print("      is paroli-server running on the TTS board (port 8848)?")
        return 1
    if not pcm:
        print(f"{RED}FAIL:{RESET} no audio received.")
        return 1

    rate = sample_rate or 22050
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(rate)
        wf.writeframes(bytes(pcm))

    seconds = len(pcm) / (2 * rate)
    print(f"{GREEN}PASS:{RESET} received {len(pcm)} bytes "
          f"(~{seconds:.1f}s @ {rate} Hz) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
