#!/usr/bin/env python3
"""
LLM node — Phase 3 (token streaming + clause chunking), runs on its OWN Rock 5C.

This is the node your laptop connects to. For each prompt it:
  Phase 3:  streams tokens from the local rkllama server (NPU / rkllm) and chops
            the stream into clause-sized chunks.
  bridge :  forwards each clause to the TTS node (tts_server.py, on the other
            Rock 5C), and relays the returned PCM straight back to the laptop.

So the laptop still talks to ONE endpoint; the LLM<->TTS hop is internal.

Protocol to the laptop (one turn):
  laptop -> node : {"text": "..."}  or  {"command": "clear"}
  node -> laptop : {"type":"audio_start","sample_rate":N,"channels":1,"sample_width":2}
                   {"type":"llm","text":"<token>"}     (transcript)
                   <binary frames>                     (raw PCM relayed from TTS node)
                   {"type":"done"}  |  {"type":"error","message":"..."}  |  {"type":"cleared"}

Run:  python llm_server.py                       # uses config.yaml
      python llm_server.py --tts-host 127.0.0.1  # run TTS on this same board to test
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

import requests

# Shared config lives at the repo root; this file is one level down (LLM/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import load_config, service_addr  # noqa: E402

# --- Layout from config.yaml ------------------------------------------------
_CFG = load_config()
_RKLLAMA_HOST, _RKLLAMA_PORT = service_addr("rkllama")
LLM_URL = f"http://{_RKLLAMA_HOST}:{_RKLLAMA_PORT}"   # local rkllama server
ROCK5C_IP, LLM_WS_PORT = service_addr("llm")          # this node; client connects here
_TTS_HOST, TTS_WS_PORT = service_addr("tts")          # the TTS node

DEFAULT_MODEL = _CFG["llm"]["model"]
DEFAULT_SYSTEM = _CFG["llm"]["system_prompt"]

# --- Smart-home tools (Tasmota plugs) ---------------------------------------
LIGHTS = _CFG.get("lights", {}) or {}

if LIGHTS:
    DEFAULT_SYSTEM += (
        " You can switch these smart lights with the set_light tool: "
        + ", ".join(LIGHTS) + ", or 'all' for every light at once. Call set_light "
        "whenever the user asks to turn a light on or off; do not ask for "
        "confirmation."
    )


def build_tools():
    """OpenAI tool spec for set_light, with the light names from config."""
    if not LIGHTS:
        return None
    return [{
        "type": "function",
        "function": {
            "name": "set_light",
            "description": "Turn one of the smart lights on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "light": {"type": "string", "enum": list(LIGHTS) + ["all"],
                              "description": "which light to switch ('all' = every light)"},
                    "state": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["light", "state"],
            },
        },
    }]


TOOLS = build_tools()


def execute_light(light, state):
    """Switch a Tasmota plug over HTTP. Return (ok, error)."""
    ip = LIGHTS.get(light)
    if not ip:
        return False, f"unknown light '{light}'"
    cmnd = "Power%20On" if state == "on" else "Power%20Off"
    try:
        r = requests.get(f"http://{ip}/cm?cmnd={cmnd}", timeout=5)
        return r.status_code == 200, None
    except requests.RequestException as exc:
        return False, str(exc)


def handle_tool_calls(calls):
    """Run each tool call; return a short spoken confirmation."""
    parts = []
    for call in calls:
        if call.get("name") != "set_light":
            parts.append("Sorry, I can't do that.")
            continue
        args = call.get("arguments") or {}
        light, state = args.get("light"), args.get("state")
        targets = list(LIGHTS) if light == "all" else [light]
        failed = []
        for t in targets:
            ok, err = execute_light(t, state)
            print(f"[tool] set_light({t}, {state}) -> {'ok' if ok else 'FAIL: ' + str(err)}",
                  flush=True)
            if not ok:
                failed.append(t)
        if light == "all":
            if not failed:
                parts.append(f"Okay, I've turned all the lights {state}.")
            else:
                parts.append("Sorry, I couldn't reach the "
                             + " and ".join(failed) + " light.")
        elif not failed:
            parts.append(f"Okay, I've turned the {light} light {state}.")
        else:
            parts.append(f"Sorry, I couldn't reach the {light} light.")
    return " ".join(parts) if parts else "Okay."


# --- Clause chunking --------------------------------------------------------
HARD_BOUNDARIES = ".!?\n"
SOFT_BOUNDARIES = ",;:"


def extract_clauses(buffer, min_soft_len=18, max_len=120):
    """Split off speakable clauses; return (clauses, remaining_tail).

    Flush at sentence enders (.!? and newline) always, at soft boundaries
    (, ; :) once the clause is long enough, and force a flush at the last word
    boundary past max_len so a long run-on sentence doesn't stall first audio.
    """
    clauses = []
    start = 0
    for i, ch in enumerate(buffer):
        seg = buffer[start:i + 1]
        stripped = seg.strip()
        if ch in HARD_BOUNDARIES or (ch in SOFT_BOUNDARIES and len(stripped) >= min_soft_len):
            if stripped:
                clauses.append(stripped)
            start = i + 1
        elif len(seg) >= max_len:
            cut = buffer.rfind(" ", start + 1, i + 1)
            if cut == -1:
                continue
            chunk = buffer[start:cut].strip()
            if chunk:
                clauses.append(chunk)
            start = cut + 1
    return clauses, buffer[start:]


def stream_llm(messages, model, tools=None):
    """Stream from rkllama; yield ('text', token) or ('tool', [calls]).

    Tool-call argument fragments are accumulated by index (rkllama sends them as
    a JSON string while streaming) and emitted once as parsed dicts at the end.
    """
    payload = {"model": model, "messages": messages, "stream": True}
    if tools:
        payload["tools"] = tools
    tool_acc = {}
    with requests.post(
        f"{LLM_URL}/v1/chat/completions", json=payload, stream=True, timeout=300
    ) as resp:
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
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            tcs = delta.get("tool_calls")
            if tcs:
                for tc in tcs:
                    acc = tool_acc.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        acc["arguments"] += args
                    elif isinstance(args, dict):
                        acc["arguments"] = json.dumps(args)
                continue
            token = delta.get("content") or ""
            if token:
                yield ("text", token)
    if tool_acc:
        calls = []
        for idx in sorted(tool_acc):
            acc = tool_acc[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": acc["name"], "arguments": args})
        yield ("tool", calls)


def synth_clause(tts_ws, text):
    """Send one clause to the TTS node; yield ('rate', n) / ('pcm', b) / ('error', m)."""
    tts_ws.send(json.dumps({"text": text}))
    for msg in tts_ws:
        if isinstance(msg, bytes):
            yield ("pcm", msg)
        else:
            ev = json.loads(msg)
            etype = ev.get("type")
            if etype == "audio_start":
                yield ("rate", ev["sample_rate"])
            elif etype == "error":
                yield ("error", ev.get("message", "TTS error"))
                return
            elif etype == "done":
                return


def run_turn(conn, tts_url, prompt, history, model):
    """Stream LLM -> clauses -> TTS node -> relay PCM to the laptop."""
    from websockets.sync.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed

    try:
        tts_ws = ws_connect(tts_url, max_size=None, open_timeout=5)
    except Exception as exc:  # noqa: BLE001
        conn.send(json.dumps({"type": "error", "message": f"Cannot reach TTS node at {tts_url}: {exc}"}))
        conn.send(json.dumps({"type": "done"}))
        return

    history.append({"role": "user", "content": prompt})
    events: queue.Queue = queue.Queue()

    def produce():
        try:
            collected = []
            buf = ""
            tool_reply = None
            for kind, val in stream_llm(history, model, TOOLS):
                if kind == "text":
                    collected.append(val)
                    events.put(("llm", val))
                    buf += val
                    clauses, buf = extract_clauses(buf)
                    for clause in clauses:
                        events.put(("clause", clause))
                elif kind == "tool":
                    # Run the tool(s), then speak a confirmation instead of a
                    # generated reply.
                    tool_reply = handle_tool_calls(val)
                    events.put(("llm", tool_reply))
                    clauses, tail = extract_clauses(tool_reply + "\n")
                    for clause in clauses:
                        events.put(("clause", clause))
                    if tail.strip():
                        events.put(("clause", tail.strip()))
                    collected = [tool_reply]
            if tool_reply is None:
                tail = buf.strip()
                if tail:
                    events.put(("clause", tail))
            events.put(("eot", "".join(collected)))
        except Exception as exc:  # noqa: BLE001
            events.put(("error", str(exc)))

    threading.Thread(target=produce, daemon=True).start()

    reply = ""
    audio_started = False
    with tts_ws:
        while True:
            kind, payload = events.get()
            if kind == "llm":
                conn.send(json.dumps({"type": "llm", "text": payload}))
            elif kind == "clause":
                try:
                    for tag, val in synth_clause(tts_ws, payload):
                        if tag == "rate":
                            if not audio_started:
                                conn.send(json.dumps({
                                    "type": "audio_start", "sample_rate": val,
                                    "channels": 1, "sample_width": 2,
                                }))
                                audio_started = True
                        elif tag == "pcm":
                            conn.send(val)
                        elif tag == "error":
                            conn.send(json.dumps({"type": "error", "message": val}))
                except ConnectionClosed:
                    conn.send(json.dumps({"type": "error", "message": "TTS node connection lost"}))
                    break
            elif kind == "eot":
                reply = payload
                break
            elif kind == "error":
                conn.send(json.dumps({"type": "error", "message": payload}))
                history.pop()
                conn.send(json.dumps({"type": "done"}))
                return

    conn.send(json.dumps({"type": "done"}))
    if reply.strip():
        history.append({"role": "assistant", "content": reply})
    else:
        history.pop()


def make_handler(tts_url, default_model, system_prompt):
    from websockets.exceptions import ConnectionClosed

    def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        print(f"[+] laptop connected: {peer}", flush=True)
        history = [{"role": "system", "content": system_prompt}]
        try:
            for message in conn:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    data = {"text": message}
                if data.get("command") == "clear":
                    history = [{"role": "system", "content": system_prompt}]
                    conn.send(json.dumps({"type": "cleared"}))
                    print(f"[i] {peer}: history cleared", flush=True)
                    continue
                prompt = (data.get("text") or "").strip()
                model = data.get("model") or default_model
                if not prompt:
                    continue
                print(f"[>] {peer}: {prompt}", flush=True)
                run_turn(conn, tts_url, prompt, history, model)
        except ConnectionClosed:
            pass
        finally:
            print(f"[-] laptop disconnected: {peer}", flush=True)

    return handler


def pick_model(default_model):
    try:
        models = requests.get(f"{LLM_URL}/models", timeout=5).json().get("models", [])
    except requests.RequestException:
        print(f"\033[31mCannot reach rkllama server at {LLM_URL}.\033[0m Start it with: rkllama serve")
        sys.exit(1)
    if not models:
        print("\033[31mNo models available on the rkllama server.\033[0m")
        sys.exit(1)
    if default_model and default_model in models:
        return default_model
    if default_model:
        print(f"\033[33mModel '{default_model}' not found; using '{models[0]}'.\033[0m")
    return models[0]


def main():
    parser = argparse.ArgumentParser(description="LLM node (Phase 3)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the laptop")
    parser.add_argument("--port", type=int, default=LLM_WS_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="rkllama model")
    parser.add_argument("--tts-host", default=_TTS_HOST,
                        help="Address of the TTS node (the other Rock 5C). "
                             "Use 127.0.0.1 to run TTS on this same board.")
    parser.add_argument("--tts-port", type=int, default=TTS_WS_PORT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    args = parser.parse_args()

    try:
        from websockets.sync.server import serve
    except ModuleNotFoundError:
        print("\033[31mMissing dependency 'websockets'.\033[0m Install: .venv/bin/pip install websockets")
        sys.exit(1)

    model = pick_model(args.model)
    tts_url = f"ws://{args.tts_host}:{args.tts_port}"
    handler = make_handler(tts_url, model, args.system)

    print("\033[32mLLM node ready.\033[0m")
    print(f"  LLM model : {model}  (via {LLM_URL})")
    print(f"  TTS node  : {tts_url}")
    print(f"  Listening : ws://{args.host}:{args.port}  (laptop connects to ws://{ROCK5C_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
