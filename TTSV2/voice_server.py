#!/usr/bin/env python3
"""TTSV2 node backed by kokoro-server — an improved Kokoro-82M NPU TTS engine.

Drop-in replacement for TTS/voice_server.py (paroli). It speaks the SAME
WebSocket protocol toward llm_server.py and listens on the same `services.tts`
port (8766), so the LLM node reaches it unchanged — only the backend differs.

    laptop --(8765)--> llm_server.py --(8766)--> TTSV2/voice_server.py
                                                      |
                                          --(8848)--> kokoro-server (NPU)

kokoro-server's API mirrors paroli's (WS /api/v1/stream, binary PCM frames, a
final {"status":"ok"}), but it selects a *voice by name* (e.g. af_heart) with a
`speed` instead of paroli's speaker_id / length_scale, and runs at 24000 Hz.

You must build + run kokoro-server yourself (C++ binary + Kokoro models):
    https://github.com/marty1885/kokoro-server   (see TTSV2/setup_kokoro.sh)

Protocol toward llm_server.py (one clause), identical to TTS/voice_server.py:
  in  : {"text": "a clause"}   |   {"command": "info"}
  out : {"type":"audio_start","sample_rate":N,"channels":1,"sample_width":2}
        <binary PCM frames>
        {"type":"done"}                |  {"type":"error","message":"..."}
        {"type":"info","sample_rate":N}   (reply to "info")

Run:  python voice_server.py            # uses config.yaml
"""

import argparse
import json
import pathlib
import re
import sys

# Shared config lives at the repo root; this file is one level down (TTSV2/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

# --- Layout from config.yaml ------------------------------------------------
_CFG = load_config()
ROCK5C_TTS_IP, TTS_WS_PORT = service_addr("tts")        # this node (banner + port)
_KOKORO_HOST, KOKORO_PORT = service_addr("kokoro")      # local kokoro-server
_V2 = _CFG["ttsv2"]

# --- Speech sanitization (same as TTS/voice_server.py) ----------------------
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"(?m)^\s*[-*+]\s+")
_STRIP_CHARS = re.compile(r"[*_`~#>|]+")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def sanitize_for_speech(text):
    """Remove markdown / special characters that would otherwise be spoken."""
    text = _LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = _STRIP_CHARS.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def kokoro_synth(kws, text, voice, speed):
    """Send one clause to kokoro-server's /api/v1/stream; yield audio/errors.

    Yields ('pcm', bytes) for each audio blob, ('error', msg) on failure.
    Returns (leaving the connection open for reuse) when kokoro reports finished.
    """
    req = {"text": text, "audio_format": "pcm"}
    if voice:
        req["voice"] = voice
    if speed is not None:
        req["speed"] = speed
    kws.send(json.dumps(req))

    while True:
        msg = kws.recv()
        if isinstance(msg, (bytes, bytearray)):
            if msg:                      # ignore empty keepalive/pong frames
                yield ("pcm", bytes(msg))
        else:
            status = json.loads(msg)
            if status.get("status") == "ok":
                return
            yield ("error", status.get("message", "kokoro synthesis failed"))
            return


def make_handler(kokoro_url, sample_rate, voice, speed):
    from websockets.sync.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed

    def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        print(f"[+] TTS client connected: {peer}", flush=True)
        kws = None  # persistent connection to kokoro-server, opened lazily

        def get_kokoro():
            nonlocal kws
            if kws is None:
                kws = ws_connect(kokoro_url, max_size=None, open_timeout=5)
            return kws

        try:
            for message in conn:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    data = {"text": message}

                if data.get("command") == "info":
                    conn.send(json.dumps({"type": "info", "sample_rate": sample_rate}))
                    continue

                text = sanitize_for_speech(data.get("text") or "")
                conn.send(json.dumps({
                    "type": "audio_start", "sample_rate": sample_rate,
                    "channels": 1, "sample_width": 2,
                }))

                if text:
                    try:
                        k = get_kokoro()
                        for tag, val in kokoro_synth(k, text, voice, speed):
                            if tag == "pcm":
                                conn.send(val)                       # relay raw PCM
                            elif tag == "error":
                                conn.send(json.dumps({"type": "error", "message": val}))
                    except (ConnectionClosed, OSError) as exc:
                        kws = None  # force reconnect next clause
                        conn.send(json.dumps({
                            "type": "error",
                            "message": f"Cannot reach kokoro-server at {kokoro_url}: {exc}",
                        }))

                conn.send(json.dumps({"type": "done"}))
        except ConnectionClosed:
            pass
        finally:
            if kws is not None:
                try:
                    kws.close()
                except Exception:
                    pass
            print(f"[-] TTS client disconnected: {peer}", flush=True)

    return handler


def check_kokoro(http_base):
    """Best-effort reachability check against kokoro's REST API."""
    import requests
    try:
        r = requests.get(f"{http_base}/api/v1/voices", timeout=4)
        if r.status_code == 200:
            try:
                voices = r.json()
                n = len(voices) if isinstance(voices, (list, dict)) else "?"
            except ValueError:
                n = "?"
            print(f"  kokoro-server reachable ({n} voice(s))")
            return
        print(f"\033[33m  kokoro-server responded {r.status_code} (continuing anyway)\033[0m")
    except Exception as exc:  # noqa: BLE001
        print(f"\033[33m  WARNING: kokoro-server not reachable yet at {http_base} ({exc}).\033[0m")
        print("  Start it with: TTSV2/kokoro-server/build/run-kokoro-server.sh")


def main():
    parser = argparse.ArgumentParser(description="Kokoro-backed TTS node (TTSV2, NPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the LLM node")
    parser.add_argument("--port", type=int, default=TTS_WS_PORT)
    parser.add_argument("--kokoro-host", default=_KOKORO_HOST,
                        help="Where kokoro-server is running")
    parser.add_argument("--kokoro-port", type=int, default=KOKORO_PORT)
    parser.add_argument("--sample-rate", type=int, default=_V2["sample_rate"],
                        help="Native PCM rate of Kokoro-82M (24000)")
    parser.add_argument("--voice", default=_V2["voice"],
                        help="Kokoro voice name (e.g. af_heart)")
    parser.add_argument("--speed", type=float, default=_V2["speed"],
                        help="Speaking rate (>1 faster, <1 slower)")
    args = parser.parse_args()

    try:
        from websockets.sync.server import serve
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'websockets'.\033[0m Install: uv sync")
        sys.exit(1)

    kokoro_url = f"ws://{args.kokoro_host}:{args.kokoro_port}/api/v1/stream"
    http_base = f"http://{args.kokoro_host}:{args.kokoro_port}"

    handler = make_handler(kokoro_url, args.sample_rate, args.voice, args.speed)

    print("\033[32mKokoro TTS node (TTSV2) ready.\033[0m")
    print(f"  Backend   : {kokoro_url}  (NPU)")
    check_kokoro(http_base)
    print(f"  Voice     : {args.voice}  (speed {args.speed})")
    print(f"  PCM rate  : {args.sample_rate} Hz")
    print(f"  Listening : ws://{args.host}:{args.port}  (reach me at ws://{ROCK5C_TTS_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
