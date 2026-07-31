#!/usr/bin/env python3
"""Aven MCP server (stdio).

Exposes Aven's device/info tools to the Claude CLI as NATIVE tools, so the brain
calls them as structured tool calls instead of the old text-JSON directive
protocol (which the model kept mangling with preambles/markdown, getting the raw
JSON spoken aloud). Self-contained tools (lights, weather, music, volume, time,
date, web) run here and return a real result the model phrases naturally.

Timers are special: only the coordinator has the speaker, so it must do the beep
when a timer fires. We keep the timer *state* here in a small shared file (so
timer_time_left / cancel_timer give accurate spoken answers), while llm_server
watches this server's tool calls in the CLI stream and sends the coordinator a
control event to actually schedule/cancel the firing. State + firing stay in
sync because both are driven by the same tool call.

Transport: newline-delimited JSON-RPC 2.0 over stdio (the MCP stdio transport).
Run: the Claude CLI spawns it via --mcp-config; not run by hand.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llm_server as A  # noqa: E402  (reuse the tool implementations + specs)

PROTOCOL_VERSION = "2024-11-05"
TIMER_FILE = pathlib.Path(__file__).resolve().parent / ".timers.json"


# --- Timer state (shared file; llm_server emits the firing control event) ----
def _load_timers():
    try:
        return json.loads(TIMER_FILE.read_text())
    except Exception:  # noqa: BLE001
        return []


def _save_timers(ts):
    try:
        TIMER_FILE.write_text(json.dumps(ts))
    except Exception:  # noqa: BLE001
        pass


def _active(ts, now):
    return [t for t in ts if t.get("deadline", 0) > now]


def _do_set_timer(minutes):
    try:
        secs = max(1, int(round(float(minutes) * 60)))
    except (TypeError, ValueError):
        return "Sorry, I didn't catch the timer length."
    now = time.time()
    ts = _active(_load_timers(), now)
    ts.append({"deadline": now + secs})
    _save_timers(ts)
    return f"Okay, timer set for {A._fmt_duration(secs)}."


def _do_cancel_timer():
    now = time.time()
    active = _active(_load_timers(), now)
    _save_timers([])
    if not active:
        return "There's no timer running."
    if len(active) == 1:
        return "Okay, I cancelled the timer."
    return f"Okay, I cancelled all {len(active)} timers."


def _do_timer_time_left():
    now = time.time()
    active = sorted(t["deadline"] - now for t in _active(_load_timers(), now))
    _save_timers([{"deadline": now + r} for r in active])  # prune expired
    if not active:
        return "There's no timer running."
    secs = int(round(active[0]))
    return f"You have {A._fmt_duration(secs)} left on your timer."


# --- Tool dispatch ----------------------------------------------------------
def call_tool(name, args):
    """Execute one Aven tool; return a plain-text result for the model."""
    if name == "set_light":
        return A.apply_light(args.get("light"), args.get("state"))
    if name == "get_weather":
        return A.get_weather()
    if name == "play_music":
        return A.play_music(args.get("query", ""))
    if name == "play_playlist":
        return A.play_playlist(args.get("query", ""))
    if name == "skip_track":
        return A.skip_track()
    if name == "set_shuffle":
        return A.set_shuffle(args.get("on"))
    if name == "resume_music":
        return A.resume_music()
    if name == "stop_music":
        return A.stop_music()
    if name == "set_volume":
        return A.set_volume(args.get("direction"), args.get("level"))
    if name == "get_time":
        return "It's " + time.strftime("%H:%M") + "."
    if name == "get_date":
        return "Today is " + time.strftime("%A, %B %-d, %Y") + "."
    if name == "search_web":
        return A.search_web(args.get("query", ""))
    if name == "say":
        return (args.get("text") or "").strip() or "Okay."
    if name == "set_timer":
        return _do_set_timer(args.get("minutes"))
    if name == "cancel_timer":
        return _do_cancel_timer()
    if name == "timer_time_left":
        return _do_timer_time_left()
    return f"Unknown tool: {name}"


def _tool_specs():
    """MCP tool list, derived from llm_server's OpenAI-style specs.

    In web/yolo mode the CLI already gives the model native WebSearch, so we drop
    our own search_web (which spawns a whole nested Claude — slow and redundant,
    and let both fire on one 'search the web' request). It stays only in 'off'
    mode, where it's the only way to reach the internet."""
    drop = set()
    if A.CLAUDE_AGENT_MODE in ("web", "yolo"):
        drop.add("search_web")
    specs = []
    for t in A.TOOLS or []:
        fn = t["function"]
        if fn["name"] in drop:
            continue
        specs.append({"name": fn["name"], "description": fn.get("description", ""),
                      "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return specs


# --- JSON-RPC / MCP plumbing ------------------------------------------------
def _send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _result(rpc_id, result):
    _send({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _error(rpc_id, code, message):
    _send({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


def handle(msg):
    method = msg.get("method")
    rpc_id = msg.get("id")
    if method == "initialize":
        ver = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        _result(rpc_id, {"protocolVersion": ver,
                         "capabilities": {"tools": {}},
                         "serverInfo": {"name": "aven", "version": "1.0"}})
    elif method in ("notifications/initialized", "notifications/cancelled"):
        pass  # notifications carry no id and need no response
    elif method == "ping":
        _result(rpc_id, {})
    elif method == "tools/list":
        _result(rpc_id, {"tools": _tool_specs()})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text = call_tool(name, args)
        except Exception as exc:  # noqa: BLE001
            text = f"Sorry, that action failed: {exc}"
        _result(rpc_id, {"content": [{"type": "text", "text": str(text)}]})
    elif rpc_id is not None:
        _error(rpc_id, -32601, f"Method not found: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(msg)
        except Exception as exc:  # noqa: BLE001  keep the server alive on any error
            if msg.get("id") is not None:
                _error(msg["id"], -32603, str(exc))


if __name__ == "__main__":
    main()
