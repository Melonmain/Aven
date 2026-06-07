#!/usr/bin/env python3
"""
TTS node backed by Paroli — the NPU port of Piper. Drop-in for tts_server.py.

Instead of running Piper on the CPU (onnxruntime), this proxies each clause to a
running `paroli-server`, which executes the VITS encoder on the RK3588 NPU. It
speaks the SAME WebSocket protocol toward llm_server.py, so nothing else changes:
just point the LLM node at this node with --tts-host / --tts-port.

    laptop --(8765)--> llm_server.py --(8766)--> voice_server_paroli.py
                                                      |
                                          --(8848)--> paroli-server (NPU)

You must build and run paroli-server yourself (it's a C++ binary + RKNN models):
    https://github.com/marty1885/paroli
    paroli-server --encoder encoder.onnx --decoder decoder.onnx \
                  -c model.json --ip 0.0.0.0 --port 8848

Protocol toward llm_server.py (one clause), identical to tts_server.py:
  in  : {"text": "a clause"}   |   {"command": "info"}
  out : {"type":"audio_start","sample_rate":N,"channels":1,"sample_width":2}
        <binary PCM frames>
        {"type":"done"}                |  {"type":"error","message":"..."}
        {"type":"info","sample_rate":N}   (reply to "info")

Run:  python voice_server_paroli.py --paroli-host 127.0.0.1 --paroli-port 8848
"""

import argparse
import json
import re
import sys

TTS_WS_PORT = 8766          # our port (what llm_server.py connects to)
PAROLI_PORT = 8848          # default paroli-server port
ROCK5C_TTS_IP = "100.108.158.94"   # board running this node (for the banner)

# --- Speech sanitization (same as tts_server.py) ----------------------------
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"(?m)^\s*[-*+]\s+")
_STRIP_CHARS = re.compile(r"[*_`~#>|]+")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def sanitize_for_speech(text):
    """Remove markdown / special characters that would otherwise be spelled out."""
    text = _LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = _STRIP_CHARS.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def paroli_synth(pws, text, speaker_id, length_scale):
    """Send one clause to paroli-server's /api/v1/stream; yield audio/errors.

    Yields ('pcm', bytes) for each audio blob, ('error', msg) on failure.
    Returns (leaving the connection open for reuse) when paroli reports finished.
    """
    req = {"text": text, "audio_format": "pcm"}
    if speaker_id is not None:
        req["speaker_id"] = speaker_id
    if length_scale is not None:
        req["length_scale"] = length_scale
    pws.send(json.dumps(req))

    while True:
        msg = pws.recv()
        if isinstance(msg, (bytes, bytearray)):
            if msg:                      # ignore empty keepalive/pong frames
                yield ("pcm", bytes(msg))
        else:
            status = json.loads(msg)
            if status.get("status") == "ok":
                return
            yield ("error", status.get("message", "paroli synthesis failed"))
            return


def make_handler(paroli_url, sample_rate, speaker_id, length_scale):
    from websockets.sync.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed

    def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        print(f"[+] TTS client connected: {peer}", flush=True)
        pws = None  # persistent connection to paroli-server, opened lazily

        def get_paroli():
            nonlocal pws
            if pws is None:
                pws = ws_connect(paroli_url, max_size=None, open_timeout=5)
            return pws

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
                        p = get_paroli()
                        for tag, val in paroli_synth(p, text, speaker_id, length_scale):
                            if tag == "pcm":
                                conn.send(val)                       # relay raw PCM
                            elif tag == "error":
                                conn.send(json.dumps({"type": "error", "message": val}))
                    except (ConnectionClosed, OSError) as exc:
                        pws = None  # force reconnect next clause
                        conn.send(json.dumps({
                            "type": "error",
                            "message": f"Cannot reach paroli-server at {paroli_url}: {exc}",
                        }))

                conn.send(json.dumps({"type": "done"}))
        except ConnectionClosed:
            pass
        finally:
            if pws is not None:
                try:
                    pws.close()
                except Exception:
                    pass
            print(f"[-] TTS client disconnected: {peer}", flush=True)

    return handler


def check_paroli(http_base):
    """Best-effort reachability check against paroli's REST API."""
    import requests
    try:
        r = requests.get(f"{http_base}/api/v1/speakers", timeout=4)
        if r.status_code == 200:
            speakers = r.json()
            note = f"{len(speakers)} speaker(s)" if speakers else "single-speaker model"
            print(f"  paroli-server reachable ({note})")
            return
        print(f"\033[33m  paroli-server responded {r.status_code} (continuing anyway)\033[0m")
    except Exception as exc:  # noqa: BLE001
        print(f"\033[33m  WARNING: paroli-server not reachable yet at {http_base} ({exc}).\033[0m")
        print("  Start it with: paroli-server --encoder ... --decoder ... -c model.json --port 8848")


def main():
    parser = argparse.ArgumentParser(description="Paroli-backed TTS node (NPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the LLM node")
    parser.add_argument("--port", type=int, default=TTS_WS_PORT)
    parser.add_argument("--paroli-host", default="127.0.0.1",
                        help="Where paroli-server is running")
    parser.add_argument("--paroli-port", type=int, default=PAROLI_PORT)
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Native PCM rate of the Paroli model (Piper voices are 22050)")
    parser.add_argument("--speaker-id", type=int, default=None,
                        help="Speaker id for multi-speaker models")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="Speaking rate (>1 slower, <1 faster)")
    args = parser.parse_args()

    try:
        from websockets.sync.server import serve
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'websockets'.\033[0m Install: .venv/bin/pip install websockets")
        sys.exit(1)

    paroli_url = f"ws://{args.paroli_host}:{args.paroli_port}/api/v1/stream"
    http_base = f"http://{args.paroli_host}:{args.paroli_port}"

    handler = make_handler(paroli_url, args.sample_rate, args.speaker_id, args.length_scale)

    print("\033[32mParoli TTS node ready.\033[0m")
    print(f"  Backend   : {paroli_url}  (NPU)")
    check_paroli(http_base)
    print(f"  PCM rate  : {args.sample_rate} Hz")
    print(f"  Listening : ws://{args.host}:{args.port}  (reach me at ws://{ROCK5C_TTS_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
