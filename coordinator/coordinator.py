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
import logging
import pathlib
import sys
import time
import wave

import numpy as np
import soxr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

CYAN, GREEN, RED, YELLOW, BOLD, RESET = (
    "\033[36m", "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m")

log = logging.getLogger("aven")

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
        self._src_rate = None     # rate of the PCM we're handed
        self._dev_rate = None     # rate the device actually runs at
        self._channels = 1
        if enabled:
            try:
                import sounddevice as sd
                self._sd = sd
            except (ModuleNotFoundError, OSError) as exc:
                print(f"{YELLOW}playback disabled (no audio device): {exc}{RESET}")
                self.enabled = False

    def begin(self, rate):
        """Note the source PCM rate; open the device once at ITS native rate.

        USB speakers are usually 44.1/48 kHz only, so we resample our 16/22/24 kHz
        audio up to the device rate in write() rather than asking it for a rate it
        can't do (which silently produces nothing on the raw ALSA device).
        """
        if not self.enabled:
            return
        self._src_rate = rate
        if self._stream is None:
            self._channels, self._dev_rate = 1, rate
            if self.device is not None:
                try:
                    info = self._sd.query_devices(self.device)
                    self._channels = min(max(int(info["max_output_channels"]), 1), 2)
                    self._dev_rate = int(info["default_samplerate"])
                except Exception:
                    self._channels, self._dev_rate = 1, rate
            self._stream = self._sd.RawOutputStream(
                samplerate=self._dev_rate, channels=self._channels, dtype="int16",
                device=self.device)
            self._stream.start()

    def write(self, pcm):
        if not (self.enabled and self._stream):
            return
        mono = np.frombuffer(pcm, dtype=np.int16)
        if self._src_rate != self._dev_rate:         # resample to the device rate
            r = soxr.resample(mono.astype(np.float32), self._src_rate, self._dev_rate)
            mono = np.clip(r, -32768, 32767).astype(np.int16)
        if self._channels > 1:                       # mono -> interleaved N-channel
            mono = np.repeat(mono[:, None], self._channels, axis=1).ravel()
        self._stream.write(mono.tobytes())

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
    audio_s = len(pcm_bytes) / 2 / rate
    t0 = time.perf_counter()
    with connect(stt_url, max_size=None, open_timeout=10) as ws:
        ws.send(json.dumps({"command": "config", "sample_rate": rate}))
        for i in range(0, len(pcm_bytes), 32000):
            ws.send(pcm_bytes[i:i + 32000])
        ws.send(json.dumps({"command": "transcribe"}))
        result = json.loads(ws.recv())
    text = result.get("text", "")
    log.info("STT  : %.0f ms for %.1fs audio -> %r", (time.perf_counter() - t0) * 1000,
             audio_s, text)
    return text


def _stream_reply(ws, text, player):
    """Send one prompt on an open LLM ws; stream transcript + play audio.

    Logs what the LLM/TTS node streams back (transcript, audio_start params, PCM
    chunk/byte counts) and the latency of each milestone. Lets ConnectionClosed
    propagate so a persistent caller can reconnect.
    """
    t0 = time.perf_counter()
    ms = lambda t: (t - t0) * 1000
    first_tok = first_audio = done_t = None
    chunks = nbytes = 0
    rate = None

    ws.send(json.dumps({"text": text}))
    log.info("LLM  : sent prompt (%d chars)", len(text))
    sys.stdout.write(f"{CYAN}{BOLD}Aven:{RESET} ")
    sys.stdout.flush()
    for message in ws:
        if isinstance(message, (bytes, bytearray)):
            if first_audio is None:
                first_audio = time.perf_counter()
                log.info("TTS  : first audio chunk @ %.0f ms", ms(first_audio))
            chunks += 1
            nbytes += len(message)
            player.write(message)
            continue
        event = json.loads(message)
        etype = event.get("type")
        if etype == "audio_start":
            rate = event.get("sample_rate")
            log.info("TTS  : audio_start %s Hz, %s ch, %s-byte samples @ %.0f ms",
                     rate, event.get("channels"), event.get("sample_width"),
                     ms(time.perf_counter()))
            player.begin(rate)
        elif etype == "llm":
            if first_tok is None:
                first_tok = time.perf_counter()
                log.info("LLM  : first token @ %.0f ms", ms(first_tok))
            sys.stdout.write(event["text"]); sys.stdout.flush()
        elif etype == "error":
            log.warning("server error: %s", event.get("message"))
            print(f"\n{RED}server error: {event.get('message')}{RESET}")
        elif etype == "done":
            done_t = time.perf_counter()
            break
    print()

    audio_s = nbytes / 2 / rate if rate else 0.0
    log.info("TTS  : received %d PCM chunks, %d bytes (~%.2fs audio)", chunks, nbytes, audio_s)
    log.info("LLM/TTS done @ %.0f ms (first_token %.0f ms, first_audio %.0f ms)",
             ms(done_t) if done_t else 0, ms(first_tok) if first_tok else 0,
             ms(first_audio) if first_audio else 0)
    return {"total_s": (done_t - t0) if done_t else 0.0, "audio_s": audio_s}


def converse(llm_url, text, player):
    """One-shot turn on a fresh connection (used by --text / --wav modes)."""
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed
    with connect(llm_url, max_size=None) as ws:
        try:
            return _stream_reply(ws, text, player)
        except ConnectionClosed:
            print(f"\n{RED}LLM connection closed.{RESET}")
            return {}


class LLMSession:
    """Persistent LLM connection so history accumulates across wake-words.

    The LLM node keeps conversation history per connection, so reusing one
    connection gives multi-turn memory; `clear()` (sent when too long has passed
    between wake-words) resets it to a fresh conversation.
    """

    def __init__(self, url):
        self.url = url
        self._ws = None

    def _ensure(self):
        if self._ws is None:
            from websockets.sync.client import connect
            self._ws = connect(self.url, max_size=None, open_timeout=10)
        return self._ws

    def _reset(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def clear(self):
        """Drop the server-side history (start a new conversation)."""
        from websockets.exceptions import ConnectionClosed
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({"command": "clear"}))
            self._ws.recv()   # consume the {"type":"cleared"} ack
        except (ConnectionClosed, OSError):
            self._reset()

    def turn(self, text, player):
        from websockets.exceptions import ConnectionClosed
        try:
            return _stream_reply(self._ensure(), text, player)
        except (ConnectionClosed, OSError):
            self._reset()   # reconnect (fresh history) on the next turn
            raise

    def close(self):
        self._reset()


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


def _rms(frame):
    a = frame.astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def read_mono(stream, channels):
    """Read one FRAME; if the device is multi-channel, keep channel 0 (mono)."""
    data, _ = stream.read(FRAME)
    a = np.frombuffer(bytes(data), dtype=np.int16)
    if channels > 1:
        a = a.reshape(-1, channels)[:, 0]
    return np.ascontiguousarray(a)


def record_utterance(stream, cc, channels):
    """Read mic frames after the wake word until trailing silence; return PCM."""
    frames = []
    elapsed = silence = voiced = 0.0
    dur = FRAME / RATE
    while True:
        m = read_mono(stream, channels)
        frames.append(m.tobytes())
        elapsed += dur
        if _rms(m) < cc["energy_threshold"]:
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

    # Match the mic's native channel count (the PS3 Eye is a 4-mic array); we
    # downmix to channel 0. openWakeWord/STT want 16 kHz mono.
    channels = 1
    if input_device is not None:
        try:
            channels = max(int(sd.query_devices(input_device)["max_input_channels"]), 1)
        except Exception:
            channels = 1

    session = LLMSession(llm_url)
    mem_timeout = cc.get("memory_timeout", 60)
    last_wake = None

    print(f"{GREEN}Aven is listening.{RESET} Say '{model_name.replace('_', ' ')}'. "
          f"(mic {channels}ch; Ctrl+C to quit)")
    with sd.RawInputStream(samplerate=RATE, blocksize=FRAME, dtype="int16",
                           channels=channels, device=input_device) as stream:
        while True:
            frame = read_mono(stream, channels)
            if _score_of(model.predict(frame), model_name) >= threshold:
                t_wake = time.perf_counter()
                # New conversation if too long since the previous wake-word.
                gap = (time.time() - last_wake) if last_wake else None
                if gap is not None and gap >= mem_timeout:
                    session.clear()
                    log.info("MEM  : reset conversation (%.0fs since last wake)", gap)
                last_wake = time.time()
                log.info("WAKE : '%s' detected%s", model_name,
                         "" if gap is None else f" ({gap:.0f}s since last)")
                print(f"\n{YELLOW}● wake — listening…{RESET}", flush=True)
                player.beep()
                pcm = record_utterance(stream, cc, channels)
                log.info("REC  : %.2fs (beep+record) -> %.1fs audio captured",
                         time.perf_counter() - t_wake, len(pcm) / 2 / RATE)
                try:
                    text = transcribe(stt_url, pcm, RATE)
                except Exception as exc:  # noqa: BLE001
                    log.warning("STT error: %s", exc)
                    print(f"{RED}STT error: {exc}{RESET}"); continue
                print(f"{CYAN}You:{RESET} {text}")
                if text.strip():
                    try:
                        session.turn(text, player)
                    except Exception:  # noqa: BLE001 — reconnect and retry once
                        try:
                            session.turn(text, player)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("LLM error: %s", exc)
                            print(f"{RED}LLM error: {exc}{RESET}")
                log.info("TURN : total %.2fs (wake -> reply done)",
                         time.perf_counter() - t_wake)
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

    # Timestamped logs to stdout (the daemon redirects this to logs/coordinator.log).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

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
