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
import re
import sys
import threading
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
# Log label reflecting which brain the LLM node uses (see llm.backend in config).
LLM_LABEL = "CLAUDE" if (_CFG.get("llm", {}).get("backend") or "claude").strip().lower() == "claude" else "LLM"
RATE = 16000
FRAME = 1280  # 80 ms @ 16 kHz

# Serialize speaker use so a firing timer never overlaps reply/beep playback.
PLAYBACK_LOCK = threading.Lock()
_timers = []  # keep references so scheduled timers aren't garbage-collected


def speaker_present(card):
    """True if an ALSA card with this id is currently plugged in.

    Lets the speaker be hot-pluggable: the USB DAC (and the dmix 'default' that
    sits on it) appears/disappears as a card in /proc/asound/cards.
    """
    if not card:
        return True
    try:
        with open("/proc/asound/cards") as f:
            data = f.read()
    except OSError:
        return True  # can't tell -> assume present, let the open attempt decide
    return re.search(rf"\[\s*{re.escape(card)}\s*\]", data) is not None


# --------------------------------------------------------------------------- #
# Speaker playback (sounddevice, opened lazily at the rate the LLM/TTS sends)
# --------------------------------------------------------------------------- #
class Player:
    def __init__(self, enabled, device=None, card=None):
        self.enabled = enabled
        self.device = device
        self.card = card          # ALSA card id to probe for hot-plug (default dev only)
        self._sd = None
        self._stream = None
        self._src_rate = None     # rate of the PCM we're handed
        self._dev_rate = None     # rate the device actually runs at
        self._channels = 1
        self._warned_absent = False
        if enabled:
            try:
                import sounddevice as sd
                self._sd = sd
            except (ModuleNotFoundError, OSError) as exc:
                print(f"{YELLOW}playback disabled (no audio device): {exc}{RESET}")
                self.enabled = False

    def available(self):
        """Whether a speaker is currently connected (hot-plug aware).

        For the default device (output_device: null) this tracks the USB DAC's
        ALSA card so the speaker can be unplugged/replugged at will. With an
        explicit device we assume it's there and let the open attempt decide.
        """
        if not self.enabled:
            return False
        if self.device is None:
            return speaker_present(self.card)
        return True

    def begin(self, rate):
        """Open the speaker for one playback session at ITS native rate.

        Opened per session (and closed by end()) so unplug/replug between
        replies is picked up. USB speakers are usually 44.1/48 kHz only, so we
        resample our 16/22/24 kHz audio up to the device rate in write() rather
        than asking it for a rate it can't do.
        """
        if not self.enabled:
            return
        if not self.available():
            if not self._warned_absent:
                log.info("playback: no speaker connected — dropping audio")
                self._warned_absent = True
            return
        self._warned_absent = False
        self._src_rate = rate
        if self._stream is not None:
            return
        # Open at the device's native rate and resample to it in write(), rather
        # than asking for our source rate. The ALSA 'default' (dmix on the USB
        # speaker) is locked to 48 kHz, so requesting 24 kHz fails with
        # 'Invalid sample rate' (-9997). Query the default device too, not just
        # an explicit one — kind="output" resolves None to the default.
        self._channels, self._dev_rate = 1, rate
        try:
            info = self._sd.query_devices(self.device, kind="output")
            self._channels = min(max(int(info["max_output_channels"]), 1), 2)
            self._dev_rate = int(info["default_samplerate"])
        except Exception:
            self._channels, self._dev_rate = 1, rate
        try:
            self._stream = self._sd.RawOutputStream(
                samplerate=self._dev_rate, channels=self._channels, dtype="int16",
                device=self.device)
            self._stream.start()
        except Exception as exc:  # speaker vanished between probe and open
            log.info("playback: couldn't open speaker (%s)", exc)
            self._stream = None

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
        if not self.available():
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
        finally:
            self.end()

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    def end(self):
        """Close the stream after a playback session so the next one re-probes
        the speaker (this is what makes unplug/replug between turns work)."""
        self._close_stream()

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
    log.info("%-4s : sent prompt (%d chars)", LLM_LABEL, len(text))
    sys.stdout.write(f"{CYAN}{BOLD}Aven:{RESET} ")
    sys.stdout.flush()
    with PLAYBACK_LOCK:                       # exclusive speaker for this reply
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
                    log.info("%-4s : first token @ %.0f ms", LLM_LABEL, ms(first_tok))
                sys.stdout.write(event["text"]); sys.stdout.flush()
            elif etype == "timer":
                schedule_timer(int(event.get("seconds", 0)), player)
            elif etype == "error":
                log.warning("server error: %s", event.get("message"))
                print(f"\n{RED}server error: {event.get('message')}{RESET}")
            elif etype == "done":
                done_t = time.perf_counter()
                break
        player.end()                          # release/re-probe speaker for next turn
    print()

    audio_s = nbytes / 2 / rate if rate else 0.0
    log.info("TTS  : received %d PCM chunks, %d bytes (~%.2fs audio)", chunks, nbytes, audio_s)
    log.info("%s/TTS done @ %.0f ms (first_token %.0f ms, first_audio %.0f ms)", LLM_LABEL,
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
        self._lock = threading.Lock()   # serialize ws use (turns vs music commands)

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
        with self._lock:
            if self._ws is None:
                return
            try:
                self._ws.send(json.dumps({"command": "clear"}))
                self._ws.recv()   # consume the {"type":"cleared"} ack
            except (ConnectionClosed, OSError):
                self._reset()

    def _command(self, name, ack_key):
        """Send a control command and return its boolean ack field (False on error)."""
        from websockets.exceptions import ConnectionClosed
        with self._lock:
            try:
                ws = self._ensure()
                ws.send(json.dumps({"command": name}))
                return bool(json.loads(ws.recv()).get(ack_key))
            except (ConnectionClosed, OSError, ValueError):
                self._reset()
                return False

    def pause_music(self):
        """Pause Spotify for a capture; True iff something was actually paused."""
        return self._command("pause_music", "paused")

    def resume_music(self):
        """Resume what pause_music() paused."""
        return self._command("resume_music", "resumed")

    def turn(self, text, player):
        from websockets.exceptions import ConnectionClosed
        with self._lock:
            try:
                return _stream_reply(self._ensure(), text, player)
            except (ConnectionClosed, OSError):
                self._reset()   # reconnect (fresh history) on the next turn
                raise

    def close(self):
        self._reset()


# --------------------------------------------------------------------------- #
# Timers (scheduled locally; only this client has the speaker)
# --------------------------------------------------------------------------- #
def tts_say(text, player):
    """Speak a phrase by synthesizing it directly on the TTS node (no LLM)."""
    from websockets.sync.client import connect
    host, port = service_addr("tts")
    try:
        with connect(f"ws://{host}:{port}", max_size=None, open_timeout=5) as ws:
            ws.send(json.dumps({"text": text}))
            for m in ws:
                if isinstance(m, (bytes, bytearray)):
                    player.write(m)
                    continue
                ev = json.loads(m)
                if ev.get("type") == "audio_start":
                    player.begin(ev["sample_rate"])
                elif ev.get("type") == "done":
                    break
    except Exception as exc:  # noqa: BLE001
        log.warning("tts_say failed: %s", exc)
    finally:
        player.end()


def fire_timer(player):
    log.info("TIMER: finished")
    print(f"\n{YELLOW}⏰ timer finished{RESET}", flush=True)
    with PLAYBACK_LOCK:
        player.beep()
        tts_say("Your timer is finished.", player)


def schedule_timer(seconds, player):
    log.info("TIMER: set for %ds", seconds)
    t = threading.Timer(seconds, fire_timer, args=(player,))
    t.daemon = True
    t.start()
    _timers.append(t)


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


def make_vad(cc):
    """webrtcvad voice-activity detector, or None to fall back to energy."""
    try:
        import webrtcvad
        return webrtcvad.Vad(int(cc.get("vad_aggressiveness", 2)))
    except Exception as exc:  # noqa: BLE001
        log.warning("webrtcvad unavailable (%s); using energy threshold", exc)
        return None


_VAD_SUB = 320  # 20 ms @ 16 kHz — webrtcvad needs 10/20/30 ms frames


def _is_speech(m, vad, energy_threshold):
    """True if the 80 ms frame contains speech (VAD if available, else energy)."""
    if vad is None:
        return _rms(m) >= energy_threshold
    for k in range(0, len(m) - _VAD_SUB + 1, _VAD_SUB):
        if vad.is_speech(m[k:k + _VAD_SUB].tobytes(), RATE):
            return True
    return False


def record_utterance(stream, cc, channels, vad):
    """Record from wake until the speaker stops (end-pointing).

    Wait for speech to start, then stop once `silence_timeout` of non-speech
    follows — so it ends right when you finish, instead of a fixed timeout.
    """
    frames = []
    elapsed = trailing_silence = 0.0
    started = False
    dur = FRAME / RATE
    while True:
        m = read_mono(stream, channels)
        frames.append(m.tobytes())
        elapsed += dur
        if _is_speech(m, vad, cc["energy_threshold"]):
            started, trailing_silence = True, 0.0
        elif started:
            trailing_silence += dur
        if elapsed >= cc["max_record_seconds"]:
            break
        if started and trailing_silence >= cc["silence_timeout"]:
            break
    return b"".join(frames)


def _open_mic(sd, device, refresh=True):
    """Open the mic and return (stream, channels), blocking until it's there.

    Makes the mic hot-pluggable: a transient unplug reconnects instead of
    crashing. PortAudio caches its device list at init, so a replugged USB mic
    isn't visible until we _terminate/_initialize it. The PS3 Eye is a 4-mic
    array; we keep its native channel count and downmix to channel 0 later.
    """
    delay, warned = 1.0, False
    while True:
        if refresh:
            try:
                sd._terminate(); sd._initialize()   # re-scan for hot-plugged devices
            except Exception:
                pass
        try:
            channels = 1
            if device is not None:
                try:
                    channels = max(int(sd.query_devices(device)["max_input_channels"]), 1)
                except Exception:
                    channels = 1
            stream = sd.RawInputStream(samplerate=RATE, blocksize=FRAME, dtype="int16",
                                       channels=channels, device=device)
            stream.start()
            return stream, channels
        except Exception as exc:  # noqa: BLE001 — mic absent; wait for it
            if not warned:
                log.warning("MIC  : not available (%s); waiting…", exc)
                print(f"{YELLOW}mic not available — waiting…{RESET}", flush=True)
                warned = True
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)
            refresh = True


def wake_loop(stt_url, llm_url, model_name, threshold, cc, player, input_device=None):
    import sounddevice as sd
    from openwakeword.model import Model

    print(f"Loading wakeword '{model_name}' (cpu/onnx)…", flush=True)
    try:
        model = Model(wakeword_models=[model_name], inference_framework="onnx")
    except Exception:
        model = Model(inference_framework="onnx")

    session = LLMSession(llm_url)
    mem_timeout = cc.get("memory_timeout", 60)
    last_wake = None
    vad = make_vad(cc)
    log.info("end-pointing: %s", "webrtcvad" if vad else "energy threshold")

    stream, channels = _open_mic(sd, input_device, refresh=False)
    print(f"{GREEN}Aven is listening.{RESET} Say '{model_name.replace('_', ' ')}'. "
          f"(mic {channels}ch; Ctrl+C to quit)")
    try:
        while True:
            try:
                frame = read_mono(stream, channels)
                if _score_of(model.predict(frame), model_name) < threshold:
                    continue
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
                # Duck Spotify while we capture, so it doesn't bleed into the
                # recording. Backgrounded so the beep isn't delayed by the API
                # round-trip; the resume waits for this to settle first.
                paused = {"v": False}
                pause_t = threading.Thread(
                    target=lambda: paused.update(v=session.pause_music()), daemon=True)
                pause_t.start()
                with PLAYBACK_LOCK:
                    player.beep()
                pcm = record_utterance(stream, cc, channels, vad)
                log.info("REC  : %.2fs (beep+record) -> %.1fs audio captured",
                         time.perf_counter() - t_wake, len(pcm) / 2 / RATE)

                # Audio is going to STT now -> resume whatever we paused.
                def _resume():
                    pause_t.join(timeout=5)
                    if paused["v"]:
                        session.resume_music()
                threading.Thread(target=_resume, daemon=True).start()
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
            except sd.PortAudioError as exc:    # mic unplugged mid-read
                log.warning("MIC  : I/O error (%s) — reconnecting…", exc)
                print(f"\n{YELLOW}mic disconnected — waiting to reconnect…{RESET}", flush=True)
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
                time.sleep(0.5)                 # don't hot-spin if it flaps
                stream, channels = _open_mic(sd, input_device)
                if hasattr(model, "reset"):
                    model.reset()
                log.info("MIC  : reconnected (%dch)", channels)
                print(f"{GREEN}mic reconnected — listening…{RESET}", flush=True)
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        session.close()


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
    # When using the default device (null), probe this ALSA card for hot-plug.
    speaker_card = cc.get("speaker_card", "V3")
    player = Player(enabled=not args.no_audio, device=out_dev, card=speaker_card)
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
