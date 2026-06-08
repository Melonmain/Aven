#!/usr/bin/env python3
"""Wakeword node — openWakeWord on the CPU (ONNX).

Listens to the microphone and fires when the wake phrase (default: "hey jarvis",
a pretrained model shipped by openWakeWord) is spoken above a score threshold.
In the full voice loop this is the trigger that starts recording for STT.

openWakeWord wants 16 kHz mono 16-bit audio in 80 ms frames (1280 samples).

Modes:
  live mic (default) : needs the `mic` extra (sounddevice + PortAudio)
  --wav FILE         : run detection over a WAV file (for testing, no mic needed)

Run:
  cd wakeword && uv sync --extra mic && uv run python wakeword_listener.py
  cd wakeword && uv run python wakeword_listener.py --wav /path/to/clip.wav
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config  # noqa: E402

_WW = load_config()["wakeword"]
RATE = 16000
FRAME = 1280  # 80 ms @ 16 kHz


def load_model(model_name):
    from openwakeword.model import Model
    try:
        return Model(wakeword_models=[model_name], inference_framework="onnx")
    except Exception:
        # Fall back to loading every downloaded model.
        return Model(inference_framework="onnx")


def score_of(prediction, model_name):
    """Pull this model's score out of the prediction dict (keys may be suffixed)."""
    if model_name in prediction:
        return prediction[model_name]
    for k, v in prediction.items():
        if k.startswith(model_name):
            return v
    return max(prediction.values()) if prediction else 0.0


def run_wav(model, model_name, threshold, wav_path):
    import wave
    import soxr

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            print(f"need 16-bit mono WAV (got width={wf.getsampwidth()} ch={wf.getnchannels()})")
            return 1
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if rate != RATE:
        f = soxr.resample(audio.astype(np.float32), rate, RATE)
        audio = np.clip(f, -32768, 32767).astype(np.int16)

    peak = 0.0
    fired = False
    for i in range(0, len(audio) - FRAME, FRAME):
        s = score_of(model.predict(audio[i:i + FRAME]), model_name)
        peak = max(peak, s)
        if s >= threshold and not fired:
            fired = True
            print(f"\033[32mDETECTED\033[0m '{model_name}' at {i / RATE:.2f}s (score {s:.2f})")
    print(f"peak score over clip: {peak:.3f}  (threshold {threshold})  "
          f"{'-> would fire' if fired else '-> no detection'}")
    return 0


def run_mic(model, model_name, threshold):
    try:
        import sounddevice as sd
    except (ModuleNotFoundError, OSError) as exc:
        print(f"\033[31mMic mode needs the 'mic' extra:\033[0m uv sync --extra mic  ({exc})")
        return 1

    print(f"\033[32mListening for '{model_name}'…\033[0m (Ctrl+C to stop)")
    last = 0.0
    with sd.RawInputStream(samplerate=RATE, blocksize=FRAME, dtype="int16",
                           channels=1) as stream:
        while True:
            data, _ = stream.read(FRAME)
            frame = np.frombuffer(data, dtype=np.int16)
            s = score_of(model.predict(frame), model_name)
            now = time.time()
            if s >= threshold and now - last > 1.0:   # debounce 1s
                last = now
                print(f"\033[32mDETECTED\033[0m '{model_name}' (score {s:.2f}) "
                      f"at {time.strftime('%H:%M:%S')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Wakeword listener (openWakeWord, CPU)")
    parser.add_argument("--model", default=_WW["model"])
    parser.add_argument("--threshold", type=float, default=_WW["threshold"])
    parser.add_argument("--wav", help="Run detection over a WAV file instead of the mic")
    args = parser.parse_args()

    print(f"Loading openWakeWord model '{args.model}' (cpu/onnx)…", flush=True)
    model = load_model(args.model)

    if args.wav:
        return run_wav(model, args.model, args.threshold, pathlib.Path(args.wav))
    return run_mic(model, args.model, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
