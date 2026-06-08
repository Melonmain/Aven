#!/usr/bin/env python3
"""Aven coordinator — the full voice loop, run on the mic+speaker device.

It ties every stage together so you talk to one program:

    mic ─▶ wakeword (local, CPU) ─▶ record utterance ─▶ STT service ─▶
        ─▶ LLM service (which itself drives the TTS node) ─▶ play the reply

Each remote stage is reached over its WebSocket from config.yaml:
  STT : services.stt   (ws, send PCM -> get transcript)
  LLM : services.llm   (ws, send text -> stream transcript + PCM audio)
The wakeword model, mic capture, and speaker playback are local.

Modes:
  (default)        full loop — needs a mic + speakers (PortAudio / sounddevice)
  --text "..."     skip mic+wakeword+STT; just send text to the LLM and play
  --wav FILE       skip mic+wakeword; transcribe FILE via STT, then LLM
  --no-audio       don't play audio (headless; still prints the transcript)

Run:
  cd coordinator && uv sync && uv run python coordinator.py
  cd coordinator && uv run python coordinator.py --text "What time is it?"
  cd coordinator && uv run python coordinator.py --wav clip.wav --no-audio
"""

import argparse
import json
import pathlib
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

CYAN, GREEN, RED, YELLOW, BOLD, RESET = (
    "\033[36m", "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m")

_CFG = load_config()
RATE = 16000
FRAME = 1280  # 80 ms @ 16 kHz


# --------------------------------------------------------------------------- #
# Speaker playback (sounddevice, opened lazily at the rate the LLM/TTS sends)
# --------------------------------------------------------------------------- #
class Player:
    def __init__(self, enabled, device=None):
        self.enabled = enabled
        self.device = device
        self._sd = None
        self._stream = None
        self._rate = None
        if enabled:
            try:
                import sounddevice as sd
                self._sd = sd
            except (ModuleNotFoundError, OSError) as exc:
                print(f"{YELLOW}playback disabled (no audio device): {exc}{RESET}")
                self.enabled = False

    def begin(self, rate):
        if not self.enabled:
            return
        if self._stream is None or self._rate != rate:
            self._close_stream()
            self._stream = self._sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.device)
            self._stream.start()
            self._rate = rate

    def write(self, pcm):
        if self.enabled and self._stream:
            self._stream.write(pcm)

    def beep(self, freq=880, ms=140, volume=0.3):
        """Play a short sine 'I heard you' tone through the speaker."""
        if not self.enabled:
            return
        rate = 16000
        n = int(rate * ms / 1000)
        t = np.arange(n) / rate
        tone = np.sin(2 * np.pi * freq * t)
        fade = max(1, int(rate * 0.005))          # 5 ms fades to avoid clicks
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        pcm = (tone * env * volume * 32767).astype(np.int16)
        try:
            self.begin(rate)
            self.write(pcm.tobytes())
            time.sleep(ms / 1000 + 0.03)          # let it finish before we record
        except Exception:
            pass

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    def close(self):
        self._close_stream()


# --------------------------------------------------------------------------- #
# Stage clients
# --------------------------------------------------------------------------- #
def transcribe(stt_url, pcm_bytes, rate):
    """Send PCM to the STT service and return the transcript text."""
    from websockets.sync.client import connect
    with connect(stt_url, max_size=None, open_timeout=10) as ws:
        ws.send(json.dumps({"command": "config", "sample_rate": rate}))
        for i in range(0, len(pcm_bytes), 32000):
            ws.send(pcm_bytes[i:i + 32000])
        ws.send(json.dumps({"command": "transcribe"}))
        result = json.loads(ws.recv())
    return result.get("text", "")


def converse(llm_url, text, player):
    """Send text to the LLM node; stream the transcript and play the audio."""
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed
    with connect(llm_url, max_size=None) as ws:
        ws.send(json.dumps({"text": text}))
        sys.stdout.write(f"{CYAN}{BOLD}Aven:{RESET} ")
        sys.stdout.flush()
        try:
            for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    player.write(message)
                    continue
                event = json.loads(message)
                etype = event.get("type")
                if etype == "audio_start":
                    player.begin(event["sample_rate"])
                elif etype == "llm":
                    sys.stdout.write(event["text"]); sys.stdout.flush()
                elif etype == "error":
                    print(f"\n{RED}server error: {event.get('message')}{RESET}")
                elif etype == "done":
                    break
        except ConnectionClosed:
            print(f"\n{RED}LLM connection closed.{RESET}")
    print()


# --------------------------------------------------------------------------- #
# Mic capture + wakeword
# --------------------------------------------------------------------------- #
def _score_of(prediction, model_name):
    if model_name in prediction:
        return prediction[model_name]
    for k, v in prediction.items():
        if k.startswith(model_name):
            return v
    return max(prediction.values()) if prediction else 0.0


def _rms(frame_bytes):
    a = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def record_utterance(stream, cc):
    """Read mic frames after the wake word until trailing silence; return PCM."""
    frames = []
    elapsed = silence = voiced = 0.0
    dur = FRAME / RATE
    while True:
        data, _ = stream.read(FRAME)
        b = bytes(data)
        frames.append(b)
        elapsed += dur
        if _rms(b) < cc["energy_threshold"]:
            silence += dur
        else:
            silence = 0.0
            voiced += dur
        if elapsed >= cc["max_record_seconds"]:
            break
        if voiced >= cc["min_record_seconds"] and silence >= cc["silence_timeout"]:
            break
    return b"".join(frames)


def wake_loop(stt_url, llm_url, model_name, threshold, cc, player, input_device=None):
    import sounddevice as sd
    from openwakeword.model import Model

    print(f"Loading wakeword '{model_name}' (cpu/onnx)…", flush=True)
    try:
        model = Model(wakeword_models=[model_name], inference_framework="onnx")
    except Exception:
        model = Model(inference_framework="onnx")

    print(f"{GREEN}Aven is listening.{RESET} Say '{model_name.replace('_', ' ')}'. "
          f"(Ctrl+C to quit)")
    with sd.RawInputStream(samplerate=RATE, blocksize=FRAME, dtype="int16",
                           channels=1, device=input_device) as stream:
        while True:
            data, _ = stream.read(FRAME)
            frame = np.frombuffer(bytes(data), dtype=np.int16)
            if _score_of(model.predict(frame), model_name) >= threshold:
                print(f"\n{YELLOW}● wake — listening…{RESET}", flush=True)
                player.beep()
                pcm = record_utterance(stream, cc)
                try:
                    text = transcribe(stt_url, pcm, RATE)
                except Exception as exc:  # noqa: BLE001
                    print(f"{RED}STT error: {exc}{RESET}"); continue
                print(f"{CYAN}You:{RESET} {text}")
                if text.strip():
                    try:
                        converse(llm_url, text, player)
                    except Exception as exc:  # noqa: BLE001
                        print(f"{RED}LLM error: {exc}{RESET}")
                if hasattr(model, "reset"):
                    model.reset()   # avoid re-triggering on the same audio
                print(f"{GREEN}listening…{RESET}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    stt_host, stt_port = service_addr("stt")
    llm_host, llm_port = service_addr("llm")
    ww = _CFG["wakeword"]
    cc = _CFG["coordinator"]

    def _dev(v):  # accept a device index (int) or a name substring (str)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    p = argparse.ArgumentParser(description="Aven coordinator (full voice loop)")
    p.add_argument("--text", help="Skip mic+STT; send this text to the LLM")
    p.add_argument("--wav", help="Skip mic+wakeword; transcribe this WAV via STT")
    p.add_argument("--no-audio", action="store_true", help="Don't play the reply")
    p.add_argument("--model", default=ww["model"], help="Wakeword model name")
    p.add_argument("--threshold", type=float, default=ww["threshold"])
    p.add_argument("--input-device", default=cc.get("input_device"),
                   help="Mic device (index or name substring); default = system default")
    p.add_argument("--output-device", default=cc.get("output_device"),
                   help="Speaker device (index or name substring); default = system default")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    stt_url = f"ws://{stt_host}:{stt_port}"
    llm_url = f"ws://{llm_host}:{llm_port}"
    in_dev, out_dev = _dev(args.input_device), _dev(args.output_device)
    player = Player(enabled=not args.no_audio, device=out_dev)
    print(f"  STT : {stt_url}\n  LLM : {llm_url}")

    try:
        if args.text:
            converse(llm_url, args.text, player)
        elif args.wav:
            with wave.open(args.wav, "rb") as wf:
                if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                    print(f"{RED}need 16-bit mono WAV{RESET}"); return 1
                rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
            text = transcribe(stt_url, pcm, rate)
            print(f"{CYAN}You:{RESET} {text}")
            if text.strip():
                converse(llm_url, text, player)
        else:
            wake_loop(stt_url, llm_url, args.model, args.threshold, cc, player, in_dev)
    except KeyboardInterrupt:
        print()
    finally:
        player.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
