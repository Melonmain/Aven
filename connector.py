#!/usr/bin/env python3
"""
Phase 5 — audio streaming & playback client (runs on your LAPTOP, 100.88.207.19).

Since phases 1+2 (mic + Whisper) are skipped for now, you type a prompt instead
of speaking it. The text is sent to the Rock 5C voice daemon, which streams back
the spoken reply as raw PCM; this client plays it as it arrives, so the first
clause is heard while the Rock 5C is still generating the rest.

Dependencies (on the laptop):
    pip install websockets
    pip install pyaudio          # preferred; falls back to `aplay` on Linux

Run:
    python phase5_client.py                 # connects to the Rock 5C over Tailscale
    python phase5_client.py --host 127.0.0.1 --port 8765
"""

import argparse
import json
import sys

from config import service_addr

# LLM orchestrator endpoint (the one service the laptop talks to).
ROCK5C_IP, VOICE_PORT = service_addr("llm")

CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


class AudioSink:
    """Plays raw 16-bit mono PCM, preferring PyAudio, falling back to aplay."""

    def __init__(self):
        self.rate = None
        self._pa = None
        self._stream = None
        self._aplay = None
        try:
            import pyaudio
            self._pyaudio = pyaudio
            self._backend = "pyaudio"
        except ModuleNotFoundError:
            import shutil
            if shutil.which("aplay"):
                self._backend = "aplay"
            else:
                print(f"{RED}No audio backend: install pyaudio or aplay.{RESET}")
                sys.exit(1)
        print(f"{GREEN}Audio backend: {self._backend}{RESET}")

    def begin(self, sample_rate):
        self.rate = sample_rate
        if self._backend == "pyaudio":
            if self._stream is None:
                self._pa = self._pyaudio.PyAudio()
                self._stream = self._pa.open(
                    format=self._pyaudio.paInt16, channels=1,
                    rate=sample_rate, output=True,
                )
        else:
            import subprocess
            self._aplay = subprocess.Popen(
                ["aplay", "-q", "-r", str(sample_rate), "-f", "S16_LE",
                 "-c", "1", "-t", "raw", "-"],
                stdin=subprocess.PIPE,
            )

    def write(self, pcm):
        if self._backend == "pyaudio":
            self._stream.write(pcm)
        elif self._aplay and self._aplay.stdin:
            self._aplay.stdin.write(pcm)

    def end(self):
        # Finish the current utterance (drain the buffer).
        if self._backend == "aplay" and self._aplay:
            try:
                self._aplay.stdin.close()
                self._aplay.wait()
            except Exception:
                pass
            self._aplay = None

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()


def chat_loop(url, sink):
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed

    print(f"{CYAN}Connecting to {url} ...{RESET}")
    with connect(url, max_size=None) as ws:
        print(f"{GREEN}Connected.{RESET} Commands: {YELLOW}/clear{RESET}, {YELLOW}/help{RESET}, {YELLOW}exit{RESET}\n")
        while True:
            try:
                prompt = input(f"{CYAN}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt.lower() == "exit":
                break
            if prompt == "/help":
                print(f"  {YELLOW}/clear{RESET}  reset the conversation history\n"
                      f"  {YELLOW}/help{RESET}   show this message\n"
                      f"  {YELLOW}exit{RESET}    quit\n")
                continue
            if prompt == "/clear":
                ws.send(json.dumps({"command": "clear"}))
                try:
                    ack = json.loads(ws.recv())
                    if ack.get("type") != "cleared":
                        raise ValueError
                except Exception:
                    pass
                print(f"{GREEN}History cleared.{RESET}\n")
                continue

            ws.send(json.dumps({"text": prompt}))
            sys.stdout.write(f"{CYAN}{BOLD}Assistant:{RESET} ")
            sys.stdout.flush()

            try:
                for message in ws:
                    if isinstance(message, bytes):
                        sink.write(message)          # PCM audio chunk
                        continue
                    event = json.loads(message)
                    etype = event.get("type")
                    if etype == "audio_start":
                        sink.begin(event["sample_rate"])
                    elif etype == "llm":
                        sys.stdout.write(event["text"])  # live transcript
                        sys.stdout.flush()
                    elif etype == "error":
                        print(f"\n{RED}Server error: {event.get('message')}{RESET}")
                    elif etype == "done":
                        sink.end()
                        break
            except ConnectionClosed:
                print(f"\n{RED}Connection closed by server.{RESET}")
                break
            print("\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 voice playback client")
    parser.add_argument("--host", default=ROCK5C_IP, help="Rock 5C address")
    parser.add_argument("--port", type=int, default=VOICE_PORT)
    args = parser.parse_args()

    try:
        import websockets  # noqa: F401
    except ModuleNotFoundError:
        print(f"{RED}Missing dependency 'websockets'. Install: pip install websockets{RESET}")
        sys.exit(1)

    sink = AudioSink()
    try:
        chat_loop(f"ws://{args.host}:{args.port}", sink)
    finally:
        sink.close()
    print(f"{RED}Goodbye.{RESET}")


if __name__ == "__main__":
    main()
