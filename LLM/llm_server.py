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
import os
import pathlib
import queue
import re
import sys
import threading
import time

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
KEEPALIVE_MIN = _CFG["llm"].get("keepalive_minutes", 0)

# --- Tools (Tasmota plugs + weather) ----------------------------------------
LIGHTS = _CFG.get("lights", {}) or {}
WEATHER = _CFG.get("weather", {}) or {}
WEATHER_KEY = os.environ.get("WEATHERAPI_KEY")          # from .env.local, not the repo
WEATHER_LOC = WEATHER.get("location", "Fulda")
WEATHER_ENABLED = bool(WEATHER and WEATHER_KEY)

SPOTIFY = _CFG.get("spotify", {}) or {}                 # Spotify Connect device name
SPOTIFY_DEVICE = SPOTIFY.get("device", "Aven")
SPOTIFY_ENABLED = bool(os.environ.get("SPOTIPY_CLIENT_ID")
                       and os.environ.get("SPOTIPY_CLIENT_SECRET"))
SPOTIFY_CACHE = str(pathlib.Path(__file__).resolve().parent / ".spotify_cache")

if LIGHTS:
    DEFAULT_SYSTEM += " Use set_light to turn lights on or off when asked; don't confirm first."
if WEATHER_ENABLED:
    DEFAULT_SYSTEM += f" Use get_weather for the current weather in {WEATHER_LOC}."
if SPOTIFY_ENABLED:
    DEFAULT_SYSTEM += (" Use play_music to play music, resume_music to continue"
                      " paused music, and stop_music to stop it on Spotify.")
DEFAULT_SYSTEM += " Use set_timer to start a timer for a number of minutes, and get_time for the current time."


def build_tools():
    """OpenAI tool specs for the enabled tools."""
    tools = []
    if LIGHTS:
        tools.append({"type": "function", "function": {
            "name": "set_light",
            "description": "Turn a light on or off.",
            "parameters": {"type": "object", "properties": {
                "light": {"type": "string", "enum": list(LIGHTS) + ["all"]},
                "state": {"type": "string", "enum": ["on", "off"]},
            }, "required": ["light", "state"]},
        }})
    if WEATHER_ENABLED:
        tools.append({"type": "function", "function": {
            "name": "get_weather",
            "description": f"Get the current weather in {WEATHER_LOC}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }})
    tools.append({"type": "function", "function": {
        "name": "set_timer",
        "description": "Start a timer for a number of minutes.",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "number"}}, "required": ["minutes"]},
    }})
    tools.append({"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current local time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }})
    if SPOTIFY_ENABLED:
        tools.append({"type": "function", "function": {
            "name": "play_music",
            "description": "Play a song or artist on Spotify.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "song or artist to play"}},
                "required": ["query"]},
        }})
        tools.append({"type": "function", "function": {
            "name": "resume_music",
            "description": "Resume or continue paused music.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }})
        tools.append({"type": "function", "function": {
            "name": "stop_music",
            "description": "Stop or pause the music.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }})
    return tools or None


TOOLS = build_tools()


def get_weather():
    """Fetch current weather; return the 'current' block as a JSON string."""
    try:
        r = requests.get("http://api.weatherapi.com/v1/current.json", timeout=8,
                         params={"key": WEATHER_KEY, "q": WEATHER_LOC, "aqi": "no"})
        r.raise_for_status()
        data = r.json()
        print(f"[tool] get_weather({WEATHER_LOC}) -> ok", flush=True)
        return json.dumps({"location": data.get("location", {}).get("name", WEATHER_LOC),
                           "current": data.get("current", {})})
    except (requests.RequestException, ValueError) as exc:
        print(f"[tool] get_weather({WEATHER_LOC}) -> FAIL: {exc}", flush=True)
        return json.dumps({"error": f"weather unavailable: {exc}"})


# --- Spotify (play_music) ---------------------------------------------------
# Credentials come from the environment (set in .env.local); a one-time sign-in
# (spotify_auth.py) writes the token cache the server reuses. Config above.
_spotify_client = None


def _spotify():
    global _spotify_client
    if _spotify_client is None:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        auth = SpotifyOAuth(
            scope="user-modify-playback-state user-read-playback-state",
            redirect_uri=os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            cache_path=SPOTIFY_CACHE, open_browser=False)
        _spotify_client = (spotipy.Spotify(auth_manager=auth), auth)
    return _spotify_client


def play_music(query):
    """Search Spotify and start playback on the Connect device (SPOTIFY_DEVICE)."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        items = sp.search(q=query, type="track", limit=1).get("tracks", {}).get("items", [])
        if not items:
            return f"I couldn't find {query} on Spotify."
        track = items[0]
        dev_id = next((d["id"] for d in sp.devices().get("devices", [])
                       if d.get("name") == SPOTIFY_DEVICE), None)
        if not dev_id:
            return f"The {SPOTIFY_DEVICE} speaker isn't available on Spotify right now."
        sp.start_playback(device_id=dev_id, uris=[track["uri"]])
        print(f"[tool] play_music({query!r}) -> {track['name']}", flush=True)
        return f"Playing {track['name']} by {track['artists'][0]['name']}."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] play_music({query!r}) -> FAIL: {exc}", flush=True)
        return "Sorry, I couldn't play that on Spotify."


def resume_music():
    """Resume the last Spotify playback (whatever was playing before it paused)."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        cur = sp.current_playback()
        if cur and cur.get("is_playing"):
            return "Music is already playing."
        # Prefer the Aven Connect device; otherwise the device the last
        # playback was on, so resume lands somewhere audible.
        dev_id = next((d["id"] for d in sp.devices().get("devices", [])
                       if d.get("name") == SPOTIFY_DEVICE), None)
        if not dev_id and cur:
            dev_id = cur.get("device", {}).get("id")
        if not dev_id:
            return f"The {SPOTIFY_DEVICE} speaker isn't available on Spotify right now."
        sp.start_playback(device_id=dev_id)  # no uris -> resume last context
        print(f"[tool] resume_music -> resumed on {SPOTIFY_DEVICE}", flush=True)
        return "Okay, resuming the music."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] resume_music -> {exc}", flush=True)
        return "Sorry, there's nothing for me to resume."


def stop_music():
    """Pause whatever is playing on Spotify (target the active device explicitly)."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        cur = sp.current_playback()
        if not cur or not cur.get("is_playing"):
            return "Nothing is playing."
        sp.pause_playback(device_id=cur["device"]["id"])
        print(f"[tool] stop_music -> paused {cur['device']['name']}", flush=True)
        return "Okay, I've stopped the music."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] stop_music -> {exc}", flush=True)
        return "Sorry, I couldn't stop the music."


# Deterministic shortcuts: critical commands the small model fumbles as tool
# calls (it sometimes just says "stopped" without doing it). Handle them here,
# before the LLM, so they always work. Returns a spoken reply or None.
_STOP_RE = re.compile(
    r"^\s*(stop|pause|quiet|be quiet|shut up|halt)"
    r"( (the )?(music|song|playback|playing|spotify))?\s*[.!]*\s*$", re.I)
_RESUME_RE = re.compile(
    r"^\s*(continue|resume|unpause|carry on|keep (playing|going))"
    r"( (the )?(music|song|playback|playing|spotify))?\s*[.!]*\s*$", re.I)

# Lights: match either word order ("turn on the bed light" / "bed light off"),
# any configured light name plus all/lights/everything -> "all".
_ALL_WORDS = {"all", "lights", "light", "everything"}
if LIGHTS:
    _names = "|".join(map(re.escape, LIGHTS)) + r"|all|lights|light|everything"
    _LIGHT_RE = re.compile(
        r"^\s*(?:turn\s+|switch\s+|put\s+)?(?:"
        r"(?P<state1>on|off)\s+(?:the\s+)?(?P<tgt1>" + _names + r")(?:\s+lights?)?"
        r"|"
        r"(?:the\s+)?(?P<tgt2>" + _names + r")(?:\s+lights?)?\s+(?P<state2>on|off)"
        r")\s*[.!]*\s*$", re.I)
else:
    _LIGHT_RE = None


def quick_intent(prompt):
    """Deterministic handling for commands the small model fumbles. Spoken reply or None."""
    p = prompt or ""
    if SPOTIFY_ENABLED:
        if _STOP_RE.match(p):
            return stop_music()
        if _RESUME_RE.match(p):
            return resume_music()
    if _LIGHT_RE is not None:
        m = _LIGHT_RE.match(p)
        if m:
            tgt = (m.group("tgt1") or m.group("tgt2")).lower()
            state = (m.group("state1") or m.group("state2")).lower()
            return apply_light("all" if tgt in _ALL_WORDS else tgt, state)
    return None


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


def apply_light(light, state):
    """Switch one light (or 'all'); return the spoken confirmation."""
    targets = list(LIGHTS) if light == "all" else [light]
    failed = []
    for t in targets:
        ok, err = execute_light(t, state)
        print(f"[tool] set_light({t}, {state}) -> {'ok' if ok else 'FAIL: ' + str(err)}",
              flush=True)
        if not ok:
            failed.append(t)
    if light == "all":
        return (f"Okay, I've turned all the lights {state}." if not failed
                else "Sorry, I couldn't reach the " + " and ".join(failed) + " light.")
    if not failed:
        return f"Okay, I've turned the {light} light {state}."
    return f"Sorry, I couldn't reach the {light} light."


def _fmt_duration(secs):
    if secs % 60 == 0:
        m = secs // 60
        return f"{m} minute" + ("" if m == 1 else "s")
    if secs < 60:
        return f"{secs} seconds"
    return f"{secs // 60} minutes {secs % 60} seconds"


def handle_tool_calls(calls):
    """Run action tool calls; return (spoken confirmation, control events).

    Control events are JSON dicts forwarded to the client (e.g. a timer the
    client schedules locally, since only it has the speaker).
    """
    parts, controls = [], []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments") or {}
        if name == "set_light":
            parts.append(apply_light(args.get("light"), args.get("state")))
        elif name == "set_timer":
            try:
                secs = max(1, int(round(float(args.get("minutes")) * 60)))
            except (TypeError, ValueError):
                parts.append("Sorry, I didn't catch the timer length.")
                continue
            controls.append({"type": "timer", "seconds": secs})
            print(f"[tool] set_timer({secs}s)", flush=True)
            parts.append(f"Okay, timer set for {_fmt_duration(secs)}.")
        elif name == "get_time":
            now = time.strftime("%H:%M")
            print(f"[tool] get_time -> {now}", flush=True)
            parts.append(f"It's {now}.")
        elif name == "play_music":
            parts.append(play_music(args.get("query", "")))
        elif name == "resume_music":
            parts.append(resume_music())
        elif name == "stop_music":
            parts.append(stop_music())
        else:
            parts.append("Sorry, I can't do that.")
    return (" ".join(parts) if parts else "Okay."), controls


def run_tool(call):
    """Execute a tool and return its result as a string (for the model)."""
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "get_weather":
        return get_weather()
    if name == "set_light":
        ok, err = execute_light(args.get("light"), args.get("state"))
        return json.dumps({"ok": ok, "light": args.get("light"),
                           "state": args.get("state"), "error": err})
    return json.dumps({"error": f"unknown tool {name}"})


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
    n = len(buffer)
    for i, ch in enumerate(buffer):
        seg = buffer[start:i + 1]
        stripped = seg.strip()
        boundary = ch in HARD_BOUNDARIES or (ch in SOFT_BOUNDARIES and len(stripped) >= min_soft_len)
        # Don't split inside a number: . , : between digits (15.7, 1,000, 15:30).
        # At the buffer's end, a digit-dot is ambiguous, so wait for the next char.
        if boundary and ch in ".,:" and i > 0 and buffer[i - 1].isdigit():
            nxt = buffer[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt.isdigit():
                boundary = False
        if boundary:
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
                    acc = tool_acc.setdefault(tc.get("index", 0),
                                              {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        acc["id"] = tc["id"]
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
            calls.append({"id": acc.get("id", ""), "name": acc["name"], "arguments": args})
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
            # Deterministic shortcut (e.g. "stop") — run it, skip the LLM.
            shortcut = quick_intent(prompt)
            if shortcut is not None:
                events.put(("llm", shortcut))
                clauses, tail = extract_clauses(shortcut + "\n")
                for clause in clauses:
                    events.put(("clause", clause))
                if tail.strip():
                    events.put(("clause", tail.strip()))
                events.put(("eot", shortcut))
                return
            for kind, val in stream_llm(history, model, TOOLS):
                if kind == "text":
                    collected.append(val)
                    events.put(("llm", val))
                    buf += val
                    clauses, buf = extract_clauses(buf)
                    for clause in clauses:
                        events.put(("clause", clause))
                elif kind == "tool":
                    if any(c.get("name") == "get_weather" for c in val):
                        # Data tool: feed the result back so the model answers in
                        # words (a second, tool-less completion -> streamed reply).
                        history.append({
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": c.get("id") or f"call_{i}", "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.dumps(c.get("arguments") or {})},
                            } for i, c in enumerate(val)],
                        })
                        for i, c in enumerate(val):
                            history.append({"role": "tool",
                                            "tool_call_id": c.get("id") or f"call_{i}",
                                            "content": run_tool(c)})
                        for kind2, val2 in stream_llm(history, model, None):
                            if kind2 == "text":
                                collected.append(val2)
                                events.put(("llm", val2))
                                buf += val2
                                clauses, buf = extract_clauses(buf)
                                for clause in clauses:
                                    events.put(("clause", clause))
                        tail = buf.strip()
                        if tail:
                            events.put(("clause", tail))
                        tool_reply = "".join(collected)
                    else:
                        # Action tools (set_light/set_timer): fixed confirmation,
                        # plus any control events for the client (e.g. a timer).
                        tool_reply, controls = handle_tool_calls(val)
                        for ctrl in controls:
                            events.put(("control", ctrl))
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
            if kind == "control":
                conn.send(json.dumps(payload))     # e.g. {"type":"timer","seconds":N}
            elif kind == "llm":
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


def keep_warm(model, minutes):
    """Ping rkllama on an interval so it never unloads the model.

    rkllama drops an idle model after ~30 min; a tiny periodic generation resets
    that timer, so the first request after a long gap (overnight, etc.) isn't a
    slow cold reload. Negligible NPU cost.
    """
    interval = minutes * 60
    while True:
        time.sleep(interval)
        try:
            requests.post(f"{LLM_URL}/v1/chat/completions", timeout=60, json={
                "model": model, "max_tokens": 1, "stream": False,
                "messages": [{"role": "user", "content": "ping"}],
            })
            print("[keepalive] pinged rkllama", flush=True)
        except requests.RequestException as exc:
            print(f"[keepalive] ping failed: {exc}", flush=True)


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

    if KEEPALIVE_MIN > 0:
        threading.Thread(target=keep_warm, args=(model, KEEPALIVE_MIN), daemon=True).start()

    print("\033[32mLLM node ready.\033[0m")
    print(f"  LLM model : {model}  (via {LLM_URL})")
    print(f"  TTS node  : {tts_url}")
    print(f"  Keep-warm : {'every %d min' % KEEPALIVE_MIN if KEEPALIVE_MIN > 0 else 'off'}")
    print(f"  Listening : ws://{args.host}:{args.port}  (laptop connects to ws://{ROCK5C_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
