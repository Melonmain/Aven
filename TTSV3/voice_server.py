#!/usr/bin/env python3
"""TTSV3 node — vanilla Piper TTS, running in-process on the CPU.

Unlike TTS (paroli) and TTSV2 (kokoro), which proxy to a separate C++ NPU
backend, this synthesizes directly with the `piper-tts` package (onnxruntime,
CPU). There is no second process and no NPU involved, so it runs anywhere.

It speaks the SAME WebSocket protocol toward llm_server.py and listens on the
same `services.tts` port (8766), so the LLM node reaches it unchanged.

    laptop --(8765)--> llm_server.py --(8766)--> TTSV3/voice_server.py
                                                  (piper synthesizes in-process)

Protocol toward llm_server.py (one clause), identical to the other TTS nodes:
  in  : {"text": "a clause"}   |   {"command": "info"}
  out : {"type":"audio_start","sample_rate":N,"channels":1,"sample_width":2}
        <binary PCM frames>
        {"type":"done"}                |  {"type":"error","message":"..."}
        {"type":"info","sample_rate":N}   (reply to "info")

Run:  python voice_server.py            # uses config.yaml + TTSV3/voices/
"""

import argparse
import json
import pathlib
import re
import sys

# Shared config lives at the repo root; this file is one level down (TTSV3/).
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from config import load_config, service_addr  # noqa: E402

# --- Layout from config.yaml ------------------------------------------------
_CFG = load_config()
ROCK5C_TTS_IP, TTS_WS_PORT = service_addr("tts")        # this node (banner + port)
_V3 = _CFG["ttsv3"]
VOICES_DIR = _HERE / "voices"

# --- Speech sanitization (same as the other TTS nodes) ----------------------
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


def load_voice(voice_name, voices_dir):
    """Load a Piper voice from voices_dir/<voice_name>.onnx."""
    from piper import PiperVoice

    model = pathlib.Path(voices_dir) / f"{voice_name}.onnx"
    if not model.exists():
        print(f"\033[31mVoice not found: {model}\033[0m")
        print(f"  download it with: python -m piper.download_voices {voice_name} "
              f"--download-dir {voices_dir}")
        print("  (or run TTSV3/setup_piper.sh)")
        sys.exit(1)
    return PiperVoice.load(str(model))


def make_handler(voice, syn_config, sample_rate):
    from websockets.exceptions import ConnectionClosed

    def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        print(f"[+] TTS client connected: {peer}", flush=True)
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
                        for chunk in voice.synthesize(text, syn_config=syn_config):
                            conn.send(chunk.audio_int16_bytes)   # raw 16-bit PCM
                    except Exception as exc:  # noqa: BLE001
                        conn.send(json.dumps({"type": "error", "message": f"piper: {exc}"}))

                conn.send(json.dumps({"type": "done"}))
        except ConnectionClosed:
            pass
        finally:
            print(f"[-] TTS client disconnected: {peer}", flush=True)

    return handler


def main():
    parser = argparse.ArgumentParser(description="Piper TTS node (TTSV3, CPU, in-process)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the LLM node")
    parser.add_argument("--port", type=int, default=TTS_WS_PORT)
    parser.add_argument("--voice", default=_V3["voice"], help="Piper voice name")
    parser.add_argument("--voices-dir", default=str(VOICES_DIR))
    parser.add_argument("--length-scale", type=float, default=_V3["length_scale"],
                        help="Speaking rate (>1 slower, <1 faster)")
    parser.add_argument("--speaker-id", type=int, default=_V3["speaker_id"],
                        help="Speaker id for multi-speaker voices")
    args = parser.parse_args()

    try:
        from websockets.sync.server import serve
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'websockets'.\033[0m Install: uv sync")
        sys.exit(1)
    try:
        from piper import SynthesisConfig
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'piper-tts'.\033[0m Install: uv sync")
        sys.exit(1)

    voice = load_voice(args.voice, args.voices_dir)
    sample_rate = getattr(voice.config, "sample_rate", 22050)
    syn_config = SynthesisConfig(
        length_scale=args.length_scale,
        speaker_id=args.speaker_id,
    )
    handler = make_handler(voice, syn_config, sample_rate)

    print("\033[32mPiper TTS node (TTSV3) ready.\033[0m")
    print(f"  Backend   : piper-tts (in-process, CPU)")
    print(f"  Voice     : {args.voice}  (length_scale {args.length_scale})")
    print(f"  PCM rate  : {sample_rate} Hz")
    print(f"  Listening : ws://{args.host}:{args.port}  (reach me at ws://{ROCK5C_TTS_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
