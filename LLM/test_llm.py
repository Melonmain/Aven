#!/usr/bin/env python3
"""Smoke test for the LLM stage — talks to the local rkllama NPU backend.

This tests the brain that llm_server.py depends on, with no TTS node involved,
so it is safe to run on the LLM board on its own. It:
  1. checks the rkllama server is up and the configured model is available,
  2. streams a short chat completion and prints the tokens as they arrive.

Run (on the LLM board, with rkllama_server already running):
    cd LLM && uv run python test_llm.py
    cd LLM && uv run python test_llm.py --prompt "Say hi in five words."
"""

import argparse
import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def main():
    cfg = load_config()
    host, port = service_addr("rkllama")
    parser = argparse.ArgumentParser(description="LLM stage smoke test (rkllama)")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model", default=cfg["llm"]["model"])
    parser.add_argument("--prompt", default="In one short sentence, what are you?")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # 1. Server reachable + model present?
    try:
        models = requests.get(f"{base}/models", timeout=5).json().get("models", [])
    except requests.RequestException as exc:
        print(f"{RED}FAIL:{RESET} cannot reach rkllama at {base} ({exc})")
        print(f"      start it with: uv run --python 3.12 rkllama_server")
        return 1

    print(f"{GREEN}OK:{RESET} rkllama reachable at {base}; models: {models}")
    if args.model not in models:
        print(f"{YELLOW}WARN:{RESET} model '{args.model}' not found. Pull it with:")
        print("      uv run --python 3.12 rkllama_client pull "
              "c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1/"
              "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm/qwen2.5-3b")
        return 1

    # 2. Stream a short completion.
    print(f"\n{YELLOW}prompt:{RESET} {args.prompt}\n{YELLOW}reply :{RESET} ", end="", flush=True)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
    }
    tokens = 0
    try:
        with requests.post(f"{base}/v1/chat/completions", json=payload,
                           stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.removeprefix("data: ").strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices and "delta" in choices[0]:
                    token = choices[0]["delta"].get("content") or ""
                    if token:
                        print(token, end="", flush=True)
                        tokens += 1
    except requests.RequestException as exc:
        print(f"\n{RED}FAIL:{RESET} streaming error ({exc})")
        return 1

    print()
    if tokens == 0:
        print(f"{RED}FAIL:{RESET} no tokens received.")
        return 1
    print(f"\n{GREEN}PASS:{RESET} streamed {tokens} tokens from '{args.model}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
