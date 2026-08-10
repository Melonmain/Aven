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
  laptop -> node : {"text": "..."}
                   {"command": "clear"}                 -> {"type":"cleared"}
                   {"command": "pause_music"}           -> {"type":"music","paused":bool}
                   {"command": "resume_music"}          -> {"type":"music","resumed":bool}
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
import shutil
import subprocess
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
# The un-augmented prompt. The per-tool "Use X to ..." sentences appended below
# help the big models but just bloat the context for the small local one, which
# already gets an explicit action list.
BASE_SYSTEM = DEFAULT_SYSTEM
KEEPALIVE_MIN = _CFG["llm"].get("keepalive_minutes", 0)

# Backend: "claude" (Claude CLI, default) or "rkllama" (local NPU model).
BACKEND = (_CFG["llm"].get("backend") or "claude").strip().lower()
CLAUDE_BIN = _CFG["llm"].get("claude_bin") or shutil.which("claude") or "claude"
CLAUDE_MODEL = _CFG["llm"].get("claude_model")   # None -> the CLI's default model
# How autonomous the Claude brain is (its NATIVE CLI tools + permission prompts,
# separate from Aven's own tools): "off" = sandboxed, no native tools; "web" =
# read-only WebSearch/WebFetch; "yolo" = every tool incl. Bash/file edits with
# all permission prompts skipped. See config.yaml for the risk note.
CLAUDE_AGENT_MODE = (_CFG["llm"].get("claude_agent_mode") or "off").strip().lower()

# Local NPU backend ("rkllm"): a .rkllm model run in-process via librkllmrt.
RKLLM_CFG = _CFG["llm"].get("rkllm", {}) or {}
RKLLM_MODEL_PATH = RKLLM_CFG.get("model_path") or ""
RKLLM_LIB_PATH = RKLLM_CFG.get("lib_path") or (
    "/home/melon/Aven/LLM/rkllama/venv/lib/python3.12/site-packages/rkllama/lib/librkllmrt.so")

# MCP: Aven's own tools are exposed to the CLI as NATIVE tools via aven_mcp.py, so
# the brain calls them as structured tool calls (reliable) instead of the old
# text-JSON directive protocol. The config points the CLI at that server, run
# with this same interpreter (so it can import llm_server + our deps).
_MCP_SERVER = str(pathlib.Path(__file__).resolve().parent / "aven_mcp.py")
_MCP_CONFIG = str(pathlib.Path(__file__).resolve().parent / ".mcp-config.json")


def _write_mcp_config():
    try:
        cfg = {"mcpServers": {"aven": {"command": sys.executable, "args": [_MCP_SERVER]}}}
        pathlib.Path(_MCP_CONFIG).write_text(json.dumps(cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[mcp] could not write config: {exc}", flush=True)

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

# Research tool: delegate to a dedicated Claude CLI call with web tools enabled.
# Reuses the host's Claude auth (no API key in the repo). Available whenever the
# CLI is present, regardless of which brain answers normally.
RESEARCH_ENABLED = bool(shutil.which(CLAUDE_BIN) or CLAUDE_BIN)
# Volume tool: ALSA mixer on the speaker card (same board as this server).
SPEAKER_CARD = (_CFG.get("coordinator", {}) or {}).get("speaker_card") or "V3"
VOLUME_ENABLED = bool(shutil.which("amixer"))

if LIGHTS:
    DEFAULT_SYSTEM += " Use set_light to turn lights on or off when asked; don't confirm first."
if WEATHER_ENABLED:
    DEFAULT_SYSTEM += f" Use get_weather for the current weather in {WEATHER_LOC}."
if SPOTIFY_ENABLED:
    DEFAULT_SYSTEM += (" Use play_music to play a song or artist, play_playlist to play a"
                      " playlist (e.g. a genre or the user's favorites), skip_track to skip"
                      " to the next song, set_shuffle to turn shuffle on or off, resume_music"
                      " to continue paused music, and stop_music to stop it on Spotify.")
if VOLUME_ENABLED or SPOTIFY_ENABLED:
    DEFAULT_SYSTEM += (" Use set_volume to make it louder or quieter; while music plays it"
                       " changes the music volume, otherwise the assistant's.")
if RESEARCH_ENABLED:
    DEFAULT_SYSTEM += (" Use search_web to look up facts, news, or anything you're"
                       " unsure about or that may have changed recently.")
DEFAULT_SYSTEM += (" Use set_timer to start a timer for a number of minutes, cancel_timer"
                   " to cancel a running timer, timer_time_left to say how long is left on"
                   " it, get_time for the current time, and get_date for today's date.")


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
        "name": "cancel_timer",
        "description": "Cancel a running timer.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }})
    tools.append({"type": "function", "function": {
        "name": "timer_time_left",
        "description": "Say how much time is left on the running timer.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }})
    tools.append({"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current local time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }})
    tools.append({"type": "function", "function": {
        "name": "get_date",
        "description": "Get today's date.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }})
    tools.append({"type": "function", "function": {
        "name": "say",
        "description": "Speak a line of text to the user. ONLY use this inside a "
                       "multi-action array when you also want to say something alongside "
                       "the actions (e.g. tell a joke while turning on a light). For a "
                       "plain spoken answer with no actions, just reply with text instead.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to say, in plain spoken words"}},
            "required": ["text"]},
    }})
    if VOLUME_ENABLED or SPOTIFY_ENABLED:
        tools.append({"type": "function", "function": {
            "name": "set_volume",
            "description": "Change the volume (the music if Spotify is playing, otherwise "
                           "the assistant). Use direction 'up' or 'down' to raise or lower "
                           "it, or 'set' with a level (0-100) for an exact level.",
            "parameters": {"type": "object", "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "set"]},
                "level": {"type": "number",
                          "description": "0-100; the target for 'set', or step size for up/down"},
            }, "required": ["direction"]},
        }})
    if RESEARCH_ENABLED:
        tools.append({"type": "function", "function": {
            "name": "search_web",
            "description": "Search the internet to answer a factual question or look up "
                           "current information. Pass a clear search query.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "what to look up"}},
                "required": ["query"]},
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
            "name": "play_playlist",
            "description": "Play a Spotify playlist. Use for requests like 'play a classic "
                           "playlist' or 'play my favorites'. Checks the user's own playlists "
                           "(including private) first, then public playlists.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "playlist name or theme, e.g. 'classic', 'favorites'"}},
                "required": ["query"]},
        }})
        tools.append({"type": "function", "function": {
            "name": "skip_track",
            "description": "Skip to the next song on Spotify.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }})
        tools.append({"type": "function", "function": {
            "name": "set_shuffle",
            "description": "Turn Spotify shuffle on or off.",
            "parameters": {"type": "object", "properties": {
                "on": {"type": "boolean", "description": "true = shuffle on, false = shuffle off"}},
                "required": ["on"]},
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

# Data tools return information the model must phrase into a spoken answer (fed
# back via data_followup), unlike action tools which get a fixed confirmation.
DATA_TOOLS = {"get_weather", "search_web"}


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
_last_librespot_restart = 0.0


def _restart_librespot():
    """Restart raspotify so the Aven Connect device comes back.

    librespot's process can stay alive while its Spirc session dies (the journal
    shows "Spirc shut down unexpectedly" / "Websocket peer does not respond"),
    which silently drops "Aven" from Spotify's Web API — every play/resume then
    fails with "the speaker isn't available". systemd can't catch it because the
    process never exits, so we restart it ourselves. Needs the sudoers drop-in
    from deploy/aven-raspotify-sudoers. Rate-limited so a genuinely offline
    Spotify can't cause a restart loop.
    """
    global _last_librespot_restart
    if time.time() - _last_librespot_restart < 60:
        return False
    _last_librespot_restart = time.time()
    try:
        r = subprocess.run(["sudo", "-n", "systemctl", "restart", "raspotify"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            print(f"[spotify] librespot restart failed: {(r.stderr or '').strip()[:120]}", flush=True)
            return False
        print("[spotify] restarted librespot (dead Connect session)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[spotify] librespot restart error: {exc}", flush=True)
        return False


def _aven_device_id(sp, heal=True):
    """Id of the Aven Connect device, healing a dead librespot session if needed."""
    def find():
        try:
            return next((d["id"] for d in sp.devices().get("devices", [])
                         if d.get("name") == SPOTIFY_DEVICE), None)
        except Exception as exc:  # noqa: BLE001
            print(f"[spotify] device list failed: {exc}", flush=True)
            return None

    dev_id = find()
    if dev_id or not heal:
        return dev_id
    if not _restart_librespot():
        return None
    for _ in range(12):          # librespot needs a few seconds to log back in
        time.sleep(1.5)
        dev_id = find()
        if dev_id:
            print("[spotify] Aven device is back", flush=True)
            return dev_id
    print("[spotify] Aven device did not return after restart", flush=True)
    return None


def _spotify():
    global _spotify_client
    if _spotify_client is None:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        auth = SpotifyOAuth(
            scope="user-modify-playback-state user-read-playback-state "
                  "user-read-recently-played playlist-read-private",
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
        dev_id = _aven_device_id(sp)
        if not dev_id:
            return f"The {SPOTIFY_DEVICE} speaker isn't available on Spotify right now."
        sp.start_playback(device_id=dev_id, uris=[track["uri"]])
        print(f"[tool] play_music({query!r}) -> {track['name']}", flush=True)
        return f"Playing {track['name']} by {track['artists'][0]['name']}."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] play_music({query!r}) -> FAIL: {exc}", flush=True)
        return "Sorry, I couldn't play that on Spotify."


# Generic words that shouldn't drive a playlist match (else "liked songs" matches
# any playlist containing "songs", e.g. "Similar songs to …").
_PLAYLIST_STOPWORDS = {"song", "songs", "music", "playlist", "playlists", "similar",
                       "to", "the", "a", "my", "some", "mix", "list", "play"}


def _playlist_match_score(query, name):
    """Loose name match so 'favorites' finds 'Favoriten', 'classic' finds 'Classic',
    ignoring generic filler words so they don't cause spurious matches."""
    def norm(s):
        return [w for w in re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
                if w not in _PLAYLIST_STOPWORDS]
    qw, nw = norm(query), norm(name)
    if not qw or not nw:
        return 0
    qs, ns = " ".join(qw), " ".join(nw)
    if qs == ns or (len(qs) >= 4 and (qs in ns or ns in qs)):
        return 3
    hits = sum(1 for a in qw for b in nw
               if a == b or (len(a) >= 4 and (b.startswith(a) or a.startswith(b))))
    return hits


def play_liked_songs(sp, dev_id):
    """Play the user's Liked Songs (saved tracks), newest addition first."""
    # current_user_saved_tracks returns newest-first, so playing them in order
    # starts with the most recently liked song.
    items = sp.current_user_saved_tracks(limit=50).get("items", []) or []
    uris = [i["track"]["uri"] for i in items if i.get("track") and i["track"].get("uri")]
    if not uris:
        return None
    sp.start_playback(device_id=dev_id, uris=uris)
    # Turn shuffle off so it actually keeps starting from the newest one.
    try:
        sp.shuffle(False, device_id=dev_id)
    except Exception:  # noqa: BLE001
        pass
    print(f"[tool] play_playlist -> Liked Songs ({len(uris)} tracks, newest first)", flush=True)
    return "Playing your liked songs, starting with the newest."


def play_playlist(query):
    """Play a Spotify playlist: prefer the user's own playlists (incl. private,
    so 'my favorites' works), else the best public search result."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        dev_id = _aven_device_id(sp)
        if not dev_id:
            return f"The {SPOTIFY_DEVICE} speaker isn't available on Spotify right now."
        # 0) "liked / saved songs" -> the real Liked Songs collection, not a playlist.
        if re.search(r"\b(liked|saved)\b", query.lower()):
            said = play_liked_songs(sp, dev_id)
            if said:
                return said
        # 1) Best match among the user's own playlists (includes private ones).
        own = sp.current_user_playlists(limit=50).get("items", []) or []
        best, best_score = None, 0
        for p in own:
            if not p:
                continue
            s = _playlist_match_score(query, p.get("name", ""))
            if s > best_score:
                best, best_score = p, s
        chosen, where = (best, "your") if best_score >= 1 else (None, None)
        # 2) Otherwise, the top public playlist for the query. Spotify's playlist
        #    search sprinkles null items in, so fetch several and take the first real one.
        if chosen is None:
            items = sp.search(q=query, type="playlist", limit=10).get("playlists", {}).get("items", [])
            items = [p for p in items if p and p.get("uri")]
            if items:
                chosen, where = items[0], "the"
        if chosen is None:
            return f"I couldn't find a {query} playlist on Spotify."
        sp.start_playback(device_id=dev_id, context_uri=chosen["uri"])
        try:
            sp.shuffle(False, device_id=dev_id)   # never shuffle by default
        except Exception:  # noqa: BLE001
            pass
        print(f"[tool] play_playlist({query!r}) -> {chosen['name']!r} (score {best_score})", flush=True)
        return f"Playing {where} {chosen['name']} playlist."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] play_playlist({query!r}) -> FAIL: {exc}", flush=True)
        return "Sorry, I couldn't play that playlist."


def _active_device_id(sp):
    """The device Spotify is playing on, else the Aven Connect device, else None."""
    cur = sp.current_playback()
    if cur and (cur.get("device") or {}).get("id"):
        return cur["device"]["id"]
    return _aven_device_id(sp)


def skip_track():
    """Skip to the next song on Spotify."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        dev_id = _active_device_id(sp)
        if not dev_id:
            return "Nothing is playing."
        sp.next_track(device_id=dev_id)
        print("[tool] skip_track", flush=True)
        return "Okay, skipping ahead."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] skip_track -> {exc}", flush=True)
        return "Sorry, I couldn't skip the song."


def set_shuffle(on):
    """Turn Spotify shuffle on or off (plain shuffle, not smart shuffle)."""
    if not SPOTIFY_ENABLED:
        return "Spotify isn't set up."
    on = bool(on)
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return "Spotify isn't authorized yet."
        dev_id = _active_device_id(sp)
        if not dev_id:
            return "Nothing is playing."
        sp.shuffle(on, device_id=dev_id)
        print(f"[tool] set_shuffle({on})", flush=True)
        return "Shuffle is on." if on else "Shuffle is off."
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] set_shuffle({on}) -> {exc}", flush=True)
        return "Sorry, I couldn't change shuffle."


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
        dev_id = _aven_device_id(sp)
        if not dev_id and cur:
            dev_id = cur.get("device", {}).get("id")
        if not dev_id:
            return f"The {SPOTIFY_DEVICE} speaker isn't available on Spotify right now."
        # A paused-but-still-live session resumes in place (keeps exact position).
        if cur is not None:
            sp.start_playback(device_id=dev_id)  # no uris -> resume in place
            print(f"[tool] resume_music -> resumed paused session on {SPOTIFY_DEVICE}", flush=True)
            return "Okay, resuming the music."
        # No live session (it expired after a gap): bare resume does nothing, so
        # restart the last track you played in its playlist/album context. Exact
        # position isn't exposed by the API, so it restarts that track.
        recent = sp.current_user_recently_played(limit=1).get("items", [])
        if not recent:
            return "There's nothing for me to resume."
        track = recent[0]["track"]
        ctx = (recent[0].get("context") or {}).get("uri")
        started = False
        if ctx:  # play the track within its context; some contexts reject offset
            try:
                sp.start_playback(device_id=dev_id, context_uri=ctx,
                                  offset={"uri": track["uri"]})
                started = True
            except Exception as ce:  # noqa: BLE001
                print(f"[tool] resume_music context retry ({ce})", flush=True)
        if not started:
            sp.start_playback(device_id=dev_id, uris=[track["uri"]])
        print(f"[tool] resume_music -> restarted {track['name']!r}", flush=True)
        return f"Okay, picking up with {track['name']}."
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


# Wake-word ducking: the coordinator pauses Spotify the moment the wake word
# fires (so the music doesn't bleed into the recording) and resumes it once the
# utterance is sent to STT. These return a bool so the coordinator only resumes
# what it actually paused — never starting music that wasn't already playing.
def pause_for_wake():
    if not SPOTIFY_ENABLED:
        return False
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return False
        cur = sp.current_playback()
        if not cur or not cur.get("is_playing"):
            return False
        sp.pause_playback(device_id=cur["device"]["id"])
        print("[wake] paused Spotify for capture", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[wake] pause failed: {exc}", flush=True)
        return False


def resume_after_wake():
    if not SPOTIFY_ENABLED:
        return False
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return False
        cur = sp.current_playback()
        if cur and cur.get("is_playing"):
            return True
        dev_id = _aven_device_id(sp)
        if not dev_id and cur:
            dev_id = cur.get("device", {}).get("id")
        if not dev_id:
            return False
        sp.start_playback(device_id=dev_id)
        print("[wake] resumed Spotify after capture", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[wake] resume failed: {exc}", flush=True)
        return False


# --- Volume (set_volume) ----------------------------------------------------
# Two independent volume domains:
#  * Spotify's own Connect volume (set over the Web API), applied by librespot —
#    changes the music without touching the assistant's voice.
#  * The hardware ALSA 'PCM' control on the USB card — the master for everything
#    (TTS + Spotify) at the DAC.
# So when music is playing, volume commands target Spotify (what the user means);
# otherwise they fall back to the hardware PCM.
def _spotify_playing_device():
    """(sp, device_id, volume_percent) if Spotify is actively playing here, else None."""
    if not SPOTIFY_ENABLED:
        return None
    try:
        sp, auth = _spotify()
        if not auth.cache_handler.get_cached_token():
            return None
        cur = sp.current_playback()
        if not (cur and cur.get("is_playing")):
            return None
        dev = cur.get("device") or {}
        if not dev.get("supports_volume") or dev.get("id") is None:
            return None
        vol = dev.get("volume_percent")
        return sp, dev["id"], (vol if isinstance(vol, int) else 50)
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] set_volume: spotify probe failed ({exc})", flush=True)
        return None


def _target_pct(direction, level, current, default_step=15):
    """Resolve a spoken volume change to an absolute 0-100 percent, or None if bad."""
    if direction == "set":
        try:
            return max(0, min(100, int(round(float(level)))))
        except (TypeError, ValueError):
            return None
    if direction in ("up", "down"):
        step = default_step
        if level is not None:
            try:
                step = max(1, min(100, int(round(float(level)))))
            except (TypeError, ValueError):
                step = default_step
        return max(0, min(100, current + (step if direction == "up" else -step)))
    return None


def set_volume(direction, level=None):
    """Change the music volume (Spotify) if it's playing, else the hardware PCM."""
    spotify = _spotify_playing_device()
    if spotify:
        sp, dev_id, curvol = spotify
        target = _target_pct(direction, level, curvol)
        if target is None:
            return "Sorry, I didn't catch that volume change."
        try:
            sp.volume(target, device_id=dev_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[tool] set_volume(spotify {direction},{level}) -> FAIL: {exc}", flush=True)
            return "Sorry, I couldn't change the music volume."
        print(f"[tool] set_volume -> spotify {target}%", flush=True)
        if direction == "set":
            return f"Okay, music volume set to {target} percent."
        return "Okay, turned the music " + ("up." if direction == "up" else "down.")

    if not VOLUME_ENABLED:
        return "Volume control isn't available."
    if direction == "set":
        pct = _target_pct("set", level, 0)
        if pct is None:
            return "Sorry, I didn't catch the volume level."
        arg = f"{pct}%"
    elif direction in ("up", "down"):
        step = 15
        if level is not None:
            try:
                step = max(1, min(100, int(round(float(level)))))
            except (TypeError, ValueError):
                step = 15
        arg = f"{step}%{'+' if direction == 'up' else '-'}"
    else:
        return "Sorry, I didn't catch that volume change."
    try:
        # -M maps to a perceptual (human-ear) scale so "50%" sounds half as loud.
        subprocess.run(["amixer", "-c", SPEAKER_CARD, "-M", "sset", "PCM", arg],
                       capture_output=True, text=True, timeout=5, check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] set_volume({direction},{level}) -> FAIL: {exc}", flush=True)
        return "Sorry, I couldn't change the volume."
    print(f"[tool] set_volume -> PCM {arg}", flush=True)
    if direction == "set":
        return f"Okay, volume set to {arg[:-1]} percent."
    return "Okay, turned it " + ("up." if direction == "up" else "down.")


# --- Research (search_web) --------------------------------------------------
_RESEARCH_SYSTEM = (
    "You are a research assistant with web access. Search the web and answer the "
    "question factually and concisely in one to three short sentences suitable for "
    "reading aloud. Plain text only: no markdown, no bullet points, no URLs, and do "
    "not list sources.")


def _strip_sources(text):
    """The web-search model often appends a 'Sources:'/citation block and markdown
    links despite instructions; drop them so the TTS reply stays clean."""
    text = re.split(r"\n\s*(?:sources?|references?|citations?)\s*:", text, flags=re.I)[0]
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)               # bare URLs
    return " ".join(text.split()).strip()


def search_web(query):
    """Answer a question from the internet via a dedicated Claude web-search call."""
    if not RESEARCH_ENABLED:
        return json.dumps({"error": "web research is not available"})
    try:
        cmd = [CLAUDE_BIN, "-p", "--output-format", "text",
               "--allowedTools", "WebSearch", "WebFetch",
               "--max-turns", "10",
               "--system-prompt", _RESEARCH_SYSTEM]
        if CLAUDE_MODEL:
            cmd += ["--model", CLAUDE_MODEL]
        cmd.append(query)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        answer = _strip_sources((out.stdout or "").strip())
        if not answer:
            print(f"[tool] search_web({query!r}) -> empty (rc={out.returncode})", flush=True)
            return json.dumps({"error": "no result found"})
        print(f"[tool] search_web({query!r}) -> {answer[:80]!r}", flush=True)
        return json.dumps({"query": query, "answer": answer})
    except subprocess.TimeoutExpired:
        print(f"[tool] search_web({query!r}) -> TIMEOUT", flush=True)
        return json.dumps({"error": "the search timed out"})
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] search_web({query!r}) -> FAIL: {exc}", flush=True)
        return json.dumps({"error": f"search failed: {exc}"})


# Deterministic shortcuts: critical commands the small model fumbles as tool
# calls (it sometimes just says "stopped" without doing it). Handle them here,
# before the LLM, so they always work. Returns a spoken reply or None.
_STOP_RE = re.compile(
    r"^\s*(stop|pause|quiet|be quiet|shut up|halt)"
    r"( (the )?(music|song|playback|playing|spotify))?\s*[.!]*\s*$", re.I)
_RESUME_RE = re.compile(
    r"^\s*(continue|resume|unpause|carry on|keep (playing|going))"
    r"( (the )?(music|song|playback|playing|spotify))?\s*[.!]*\s*$", re.I)
_SKIP_RE = re.compile(
    r"^\s*(skip|next)( (song|track|this|it|one|please))*\s*[.!]*\s*$", re.I)
_SHUFFLE_OFF_RE = re.compile(
    r"^\s*(shuffle off|no shuffle|turn off shuffle|disable shuffle|stop shuffling)"
    r"( (the )?(music|songs?))?\s*[.!]*\s*$", re.I)
_SHUFFLE_ON_RE = re.compile(
    r"^\s*(shuffle on|turn on shuffle|enable shuffle|shuffle)"
    r"( (the )?(music|songs?))?\s*[.!]*\s*$", re.I)

# Time/date: the local NPU model guesses these instead of calling the tool (it
# once answered "10:30 AM" at 22:30), so answer them from the clock directly.
_TIME_RE = re.compile(
    r"^\s*(what('?s| is)? )?(the )?(current )?time( is it)?( now| please)?\s*[.?!]*\s*$", re.I)
_DATE_RE = re.compile(
    r"^\s*(what('?s| is)? )?(the )?(current )?(date|day)( is it)?( today| now)?\s*[.?!]*\s*$"
    r"|^\s*what day is (it|today)\s*[.?!]*\s*$", re.I)

# Lights: match either word order ("turn on the bed light" / "bed light off"),
# any configured light name plus all/lights/everything -> "all".
_ALL_WORDS = {"all", "lights", "light", "everything"}

# Light commands are matched by slots, not by one anchored pattern: find a light
# to act on and a state to put it in, and ignore the filler around them. An
# anchored regex missed most natural phrasings ("turn off ALL THE lights",
# "PLEASE turn off the bed light", "COULD YOU turn on the tv light").
_LIGHT_GENERIC = {"light", "lights", "lamp", "lamps"}
_LIGHT_ALL = {"all", "everything", "every"}
_STATE_OFF = {"off", "out", "dark", "darkness"}
_STATE_ON = {"on", "bright", "brighter"}
_OFF_VERBS = {"kill", "extinguish", "douse"}          # "kill the lights"
_ACTION_VERBS = {"turn", "switch", "put", "set", "make", "flip", "power", "shut",
                 "hit", "toggle"} | _OFF_VERBS
# Never act on a question or a negated request — a false positive silently
# switches the user's lights, which is worse than falling through to the model.
_INTERROGATIVE = {"what", "why", "which", "who", "whose", "how", "when", "whether"}
_QUESTION_LEAD = {"is", "are", "was", "were", "did", "does", "do", "has", "have", "should"}
_NEGATIONS = {"dont", "don't", "not", "never", "cant", "can't", "without", "instead"}
_POLITE = {"can", "could", "would", "will", "please", "pls", "kindly"}
# Nouns that mean the user meant some other device, not the lights.
_OTHER_DEVICE = {"music", "song", "songs", "spotify", "playback", "volume",
                 "timer", "alarm", "sound", "audio", "podcast", "radio"}


def match_light_intent(text):
    """(targets, state) for a light command, else None. Tolerant of filler words."""
    if not LIGHTS:
        return None
    words = re.findall(r"[a-z']+", (text or "").lower())
    if not words:
        return None
    ws = set(words)
    named = [n for n in LIGHTS if n in ws]
    if not (named or ws & _LIGHT_GENERIC):
        # No light named. "turn everything off" still means the lights — but only
        # when nothing else is named, so "turn off all the music" doesn't match.
        if not (ws & _LIGHT_ALL) or ws & _OTHER_DEVICE:
            return None
    if ws & _NEGATIONS or ws & _INTERROGATIVE:
        return None                                    # "don't …" / "why is …"
    # "is the bed light on?" is a question; "could you turn the light on" is not.
    if words[0] in _QUESTION_LEAD and not (ws & _POLITE and ws & _ACTION_VERBS):
        return None
    if ws & _STATE_OFF or ws & _OFF_VERBS:
        state = "off"
    elif ws & _STATE_ON:
        state = "on"
    else:
        return None                                    # no state -> not a command
    targets = ["all"] if (ws & _LIGHT_ALL or not named) else named
    return targets, state


def query_light_state(light):
    """Ask a Tasmota plug whether it's on; returns 'on'/'off'/None."""
    ip = LIGHTS.get(light)
    if not ip:
        return None
    try:
        r = requests.get(f"http://{ip}/cm?cmnd=Power", timeout=5)
        power = (r.json() or {}).get("POWER", "")
        return power.lower() if power.lower() in ("on", "off") else None
    except Exception:  # noqa: BLE001
        return None


def match_light_query(text):
    """Targets for a "is the X light on?" question, else None.

    Answering these ourselves matters: left to the model, a question about a
    light gets treated as a command (it once replied "Okay, I've turned the bed
    light off" to "is the bed light on").
    """
    if not LIGHTS:
        return None
    words = re.findall(r"[a-z']+", (text or "").lower())
    if not words:
        return None
    ws = set(words)
    if not (ws & _LIGHT_GENERIC or [n for n in LIGHTS if n in ws]):
        return None
    asking = words[0] in _QUESTION_LEAD or bool(ws & _INTERROGATIVE)
    if not asking or not (ws & (_STATE_ON | _STATE_OFF) or ws & {"status", "state"}):
        return None
    named = [n for n in LIGHTS if n in ws]
    return named or list(LIGHTS)


def answer_light_state(targets):
    states = [(t, query_light_state(t)) for t in targets]
    known = [(t, s) for t, s in states if s]
    if not known:
        return "Sorry, I couldn't reach the lights."
    if len(known) == 1:
        return f"The {known[0][0]} light is {known[0][1]}."
    return " ".join(f"The {t} light is {s}." for t, s in known)


def quick_intent(prompt):
    """Deterministic handling for commands the small model fumbles. Spoken reply or None."""
    p = prompt or ""
    if _TIME_RE.match(p):
        return "It's " + time.strftime("%H:%M") + "."
    if _DATE_RE.match(p):
        return "Today is " + time.strftime("%A, %B %-d, %Y") + "."
    if SPOTIFY_ENABLED:
        if _STOP_RE.match(p):
            return stop_music()
        if _RESUME_RE.match(p):
            return resume_music()
        if _SKIP_RE.match(p):
            return skip_track()
        if _SHUFFLE_OFF_RE.match(p):   # check 'off' before the bare-'shuffle' -> on
            return set_shuffle(False)
        if _SHUFFLE_ON_RE.match(p):
            return set_shuffle(True)
    query = match_light_query(p)          # check questions before commands
    if query:
        return answer_light_state(query)
    light = match_light_intent(p)
    if light:
        targets, state = light
        return " ".join(apply_light(t, state) for t in targets)
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
        elif name == "cancel_timer":
            # The timer lives on the coordinator (only it has the speaker), so it
            # cancels and speaks the result; we emit no spoken reply here.
            controls.append({"type": "cancel_timer"})
            print("[tool] cancel_timer", flush=True)
        elif name == "timer_time_left":
            controls.append({"type": "query_timer"})
            print("[tool] timer_time_left", flush=True)
        elif name == "get_time":
            now = time.strftime("%H:%M")
            print(f"[tool] get_time -> {now}", flush=True)
            parts.append(f"It's {now}.")
        elif name == "get_date":
            today = time.strftime("%A, %B %-d, %Y")
            print(f"[tool] get_date -> {today}", flush=True)
            parts.append(f"Today is {today}.")
        elif name == "say":
            text = (args.get("text") or "").strip()
            if text:
                parts.append(text)
        elif name == "set_volume":
            parts.append(set_volume(args.get("direction"), args.get("level")))
        elif name == "play_music":
            parts.append(play_music(args.get("query", "")))
        elif name == "resume_music":
            parts.append(resume_music())
        elif name == "stop_music":
            parts.append(stop_music())
        else:
            parts.append("Sorry, I can't do that.")
    # Tools that only emit a control event (the coordinator speaks the result)
    # contribute no parts — return an empty reply so we don't speak a stray "Okay".
    reply = " ".join(p for p in parts if p)
    if not reply and not controls:
        reply = "Okay."
    return reply, controls


def run_tool(call):
    """Execute a tool and return its result as a string (for the model)."""
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "get_weather":
        return get_weather()
    if name == "search_web":
        return search_web(args.get("query", ""))
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
    """Stream a turn from the active backend; yield ('text', token) / ('tool', [calls])."""
    if BACKEND == "claude":
        yield from stream_claude(messages, tools)
    elif BACKEND == "rkllm":
        yield from _stream_rkllm(messages, tools)
    else:
        yield from _stream_rkllama(messages, model, tools)


# --- Local NPU backend (rkllm) ----------------------------------------------
_rkllm_model = None
_rkllm_lock = threading.Lock()


def get_rkllm_model():
    """Load the .rkllm model once (first use), then reuse it."""
    global _rkllm_model
    with _rkllm_lock:
        if _rkllm_model is None:
            from rkllm_backend import RKLLMModel
            print(f"[rkllm] loading {os.path.basename(RKLLM_MODEL_PATH)} …", flush=True)
            _rkllm_model = RKLLMModel(
                RKLLM_MODEL_PATH, RKLLM_LIB_PATH,
                max_context=RKLLM_CFG.get("max_context", 4096),
                max_new_tokens=RKLLM_CFG.get("max_new_tokens", 512),
                temperature=RKLLM_CFG.get("temperature", 0.7),
                top_k=RKLLM_CFG.get("top_k", 40), top_p=RKLLM_CFG.get("top_p", 0.9),
                family=RKLLM_CFG.get("family"))
            print(f"[rkllm] ready in {_rkllm_model.load_seconds:.1f}s "
                  f"(family {_rkllm_model.family})", flush=True)
    return _rkllm_model


def _compact_tool_doc(tools):
    """Signatures only — no descriptions. Keeps the local model's prompt small."""
    lines = []
    for t in tools or []:
        fn = t["function"]
        props = (fn.get("parameters") or {}).get("properties") or {}
        args = ", ".join(f"{k}={'|'.join(v['enum'])}" if v.get("enum") else k
                         for k, v in props.items())
        lines.append(f"- {fn['name']}({args})")
    return "\n".join(lines)


def build_rkllm_prompt(messages, tools):
    """Build the single user-turn prompt for the local model.

    The chat template already supplies the turn markers, so we must NOT add
    "User:"/"Assistant:" scaffolding — a small model echoes it back. The model
    also has no native function calling (rkllm_set_function_tools needs an
    embedded chat template, which the Gemma 4 build doesn't provide in a
    parseable form), so tools go through the JSON-directive protocol Aven
    already parses. Small models need the exact shape shown, not described.
    """
    system = BASE_SYSTEM
    history, user_msg = [], ""
    for m in messages:
        role = m.get("role")
        if role == "system" and m.get("content"):
            system = m["content"]
        elif role == "user" and m.get("content"):
            if user_msg:
                history.append("Me: " + user_msg)
            user_msg = m["content"]
        elif role == "assistant" and m.get("content"):
            history.append("You: " + m["content"])
        elif role == "tool" and m.get("content"):
            history.append("Tool result: " + m["content"])

    doc = _compact_tool_doc(tools)
    parts = [system]
    if doc:
        parts.append(
            "You can perform these actions:\n" + doc +
            '\n\nIf the request needs an action, reply with ONLY a JSON object in exactly'
            ' this shape, and nothing else:\n'
            '{"tool": "set_light", "arguments": {"light": "bed", "state": "on"}}\n'
            'For several actions, reply with a JSON array of such objects.\n'
            'The "tool" value is only the name — never put arguments inside it.\n'
            'If no action is needed, just answer in one or two short spoken sentences,'
            ' with no JSON and without repeating the question.')
    # A custom chat template disables the runtime's thinking suppression, so the
    # model sometimes emits a "thought" block; ask for the answer only.
    parts.append("Give only your final answer. Never show your reasoning or thoughts.")
    if history:
        parts.append("Recent conversation:\n" + "\n".join(history[-6:]))
    parts.append("Request: " + user_msg)
    return "\n\n".join(parts)


_THOUGHT_RE = re.compile(r"^\s*(thought|thinking)\b[:\s]*", re.I)


def _strip_thought(text):
    """Drop a leaked reasoning block. skip_special_token eats the real markers,
    so a thinking turn arrives as a bare 'thought' line; keep what follows it."""
    s = (text or "").lstrip()
    if not _THOUGHT_RE.match(s):
        return text
    body = _THOUGHT_RE.sub("", s, count=1)
    # The answer, if any, follows a blank line; otherwise there is nothing usable.
    return body.split("\n\n", 1)[1].strip() if "\n\n" in body else ""


_FAKE_CALL_RE = re.compile(r'^\s*[\{\[]?\s*"?tool"?\s*[:=]|^\s*\w+\s*\([^)]*\)\s*$')


def _looks_like_broken_call(text):
    """A malformed tool attempt (e.g. 'set_timer(minutes:3)' or a bad JSON blob)
    must never be spoken aloud verbatim."""
    s = (text or "").strip()
    if not s:
        return False
    if _FAKE_CALL_RE.match(s):
        return True
    return s.startswith(("{", "[")) and '"tool"' in s


def _stream_rkllm(messages, tools=None):
    """Stream a turn from the local NPU model; yield ('text', tok) / ('tool', calls)."""
    try:
        model = get_rkllm_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[rkllm] load failed: {exc}", flush=True)
        yield ("text", "Sorry, the local model isn't available.")
        return
    prompt = build_rkllm_prompt(messages, tools)
    buf, mode = "", None
    for piece in model.generate(prompt):
        buf += piece
        if mode is None:
            stripped = buf.lstrip()
            if not stripped:
                continue
            if tools and stripped[0] in "{[`\"":
                # Might be a tool directive: buffer it so a malformed one is
                # never streamed to the speaker.
                mode = "tool"
            elif _THOUGHT_RE.match(stripped):
                mode = "thought"          # leaked reasoning; drop until the answer
            elif len(stripped) < 8 and stripped.isalpha():
                continue                  # too early to tell ("thou…")
            else:
                mode = "text"
                yield ("text", buf)       # flush what we buffered while deciding
                continue
        if mode == "thought":
            answer = _strip_thought(buf)
            if answer:                    # reasoning ended, answer started
                mode = "text"
                buf = answer
                yield ("text", answer)
            continue
        if mode == "text":
            yield ("text", piece)
    if mode != "tool":
        return
    calls = _parse_claude_tool(buf)
    if calls:
        yield ("tool", calls)
    elif _looks_like_broken_call(buf):
        print(f"[rkllm] discarded malformed tool call: {buf.strip()[:120]!r}", flush=True)
        yield ("text", "Sorry, I didn't catch that. Could you say it again?")
    else:
        yield ("text", buf)


def _stream_rkllama(messages, model, tools=None):
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


# --- Claude CLI backend -----------------------------------------------------
# Uses the installed `claude` CLI (the user's own auth — no API key) as the
# brain. Tools aren't native to the CLI, so we describe them in the system
# prompt and ask the model to reply with a single JSON line, which we parse back
# into the same ('tool', [calls]) shape the rest of the pipeline expects.
_CLAUDE_BLOCKED_TOOLS = ["Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch",
                         "Glob", "Grep", "Task", "NotebookEdit"]


def _claude_agent_flags():
    """CLI flags: load Aven's MCP tools and gate the brain's OTHER native tools
    per CLAUDE_AGENT_MODE. Aven's own tools (mcp__aven) plus ToolSearch — needed
    to load them, since MCP tools are deferred in this CLI — are always allowed;
    the mode only controls web/shell access:
      off  - Aven tools only; web/shell blocked
      web  - Aven tools + read-only WebSearch/WebFetch
      yolo - everything incl. Bash/file edits, no permission prompts."""
    base = ["--mcp-config", _MCP_CONFIG, "--strict-mcp-config"]
    if CLAUDE_AGENT_MODE == "yolo":
        return base + ["--dangerously-skip-permissions"]
    allow = ["mcp__aven", "ToolSearch"]
    block = ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "NotebookEdit"]
    if CLAUDE_AGENT_MODE == "web":
        return base + ["--allowedTools", *allow, "WebSearch", "WebFetch",
                       "--disallowedTools", *block]
    return base + ["--allowedTools", *allow,
                   "--disallowedTools", *block, "WebSearch", "WebFetch"]


def _claude_tool_doc(tools):
    lines = []
    for t in tools or []:
        fn = t["function"]
        props = (fn.get("parameters") or {}).get("properties") or {}
        sig = ", ".join(
            f"{k}: {v.get('type', 'string')}" + (f" ({'|'.join(v['enum'])})" if v.get("enum") else "")
            for k, v in props.items())
        lines.append(f'- {fn["name"]}({sig}) — {fn["description"]}')
    return "\n".join(lines)


def build_claude_system(base, tools):
    """System prompt for the Claude backend. Aven's tools are NATIVE MCP tools
    now (mcp__aven__*), so we just tell the model to use them — no text protocol."""
    # The CLI injects the operator's account identity (email) into context; a
    # voice assistant shouldn't volunteer it.
    base = base + (" Never mention or use the operator's email address or other"
                   " host-account details, even if asked.")
    names = ", ".join(t["function"]["name"] for t in (tools or []))
    if not names:
        return base
    web = (" You can also search the web when a question needs current information."
           " When you do, answer in your own words only — never read out or list sources,"
           " citations, links, or web addresses; the reply is spoken aloud."
           ) if CLAUDE_AGENT_MODE != "off" else ""
    return (base + "\n\nYou have tools (the aven tools: " + names + ") to control the"
            " user's home and answer questions. Use them to carry out what the user asks,"
            " calling several in one turn when several things are asked at once, then reply"
            " in one or two short spoken sentences." + web + " Never say you can't control"
            " devices or check things — use the tools.")


def _timer_control(name, inp):
    """Map an aven timer MCP tool call to the coordinator control event that
    actually schedules/cancels the firing (the beep). timer_time_left needs no
    control — the MCP server answers it from shared state."""
    inp = inp or {}
    if name == "mcp__aven__set_timer":
        try:
            return {"type": "timer", "seconds": max(1, int(round(float(inp.get("minutes")) * 60)))}
        except (TypeError, ValueError):
            return None
    if name == "mcp__aven__cancel_timer":
        # The model already speaks the confirmation (from the MCP result), so the
        # coordinator should cancel the firing silently.
        return {"type": "cancel_timer", "silent": True}
    return None


def _serialize_for_claude(messages):
    """Render the OpenAI-style history into a plain transcript for `claude -p`."""
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            lines.append("User: " + (m.get("content") or ""))
        elif role == "assistant":
            if m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                lines.append(f'Assistant used {fn["name"]}({fn["arguments"]})')
            elif m.get("content"):
                lines.append("Assistant: " + m["content"])
        elif role == "tool":
            lines.append("Tool result: " + (m.get("content") or ""))
    return "\n".join(lines)


def _strip_code_fence(s):
    """Drop a ```/```json markdown fence the model sometimes wraps JSON in."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def _parse_claude_tool(text):
    """Parse a JSON tool directive into a list of calls, or None.

    Accepts a single object {"tool":..,"arguments":..} or a JSON array of them
    (so several device actions can be requested in one reply, e.g. lights + music),
    tolerating a markdown code fence the model may wrap it in."""
    s = _strip_code_fence(text)
    if not (s and s[0] in "{[" and '"tool"' in s):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for i, it in enumerate(items):
        if isinstance(it, dict) and it.get("tool"):
            calls.append({"id": f"call_{i}", "name": it["tool"],
                          "arguments": it.get("arguments") or {}})
    return calls or None


def stream_claude(messages, tools=None):
    """Stream a turn via the Claude CLI; yield ('text', token) / ('tool', [calls])."""
    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else DEFAULT_SYSTEM
    # 'off' answers in one turn; agent modes need extra turns to use a tool then reply.
    max_turns = "1" if CLAUDE_AGENT_MODE == "off" else "12"
    cmd = [CLAUDE_BIN, "-p", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages", "--max-turns", max_turns,
           "--system-prompt", build_claude_system(system, tools),
           *_claude_agent_flags()]  # native tools + permission mode (off/web/yolo)
    if CLAUDE_MODEL:
        cmd += ["--model", CLAUDE_MODEL]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except OSError as exc:
        print(f"[claude] launch failed: {exc}", flush=True)
        yield ("text", "Sorry, I couldn't reach Claude.")
        return

    proc.stdin.write(_serialize_for_claude(messages))
    proc.stdin.close()

    buf = ""
    mode = None  # None=undecided, "text"=stream it, "tool"=buffer until done
    whole = ""   # fallback: full text from the final assistant message
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "assistant":     # complete message — fallback if no deltas stream
                whole = "".join(b.get("text", "") for b in ev.get("message", {}).get("content", [])
                                if b.get("type") == "text")
                continue
            if etype != "stream_event":
                continue
            inner = ev.get("event") or {}
            if inner.get("type") != "content_block_delta":
                continue
            delta = inner.get("delta") or {}
            if delta.get("type") != "text_delta":   # ignore thinking deltas
                continue
            piece = delta.get("text") or ""
            if not piece:
                continue
            buf += piece
            if mode is None:
                stripped = buf.lstrip()
                if not stripped:
                    continue
                # A tool directive (only when tools are offered) starts with '{'.
                mode = "tool" if (tools and stripped[0] in "{[`") else "text"
            if mode == "text":
                yield ("text", piece)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    # No deltas streamed (e.g. the model thought first, then emitted the whole
    # message at once) — decide from the final assistant text instead.
    if mode is None and whole.strip():
        buf = whole
        mode = "tool" if (tools and whole.lstrip()[0] in "{[`") else "text"
        if mode == "text":
            yield ("text", whole)
    if mode == "tool":
        calls = _parse_claude_tool(buf)
        if calls:
            yield ("tool", calls)
        else:                          # looked like JSON but wasn't a tool — speak it
            yield ("text", buf)


class ClaudeSession:
    """A persistent `claude -p` process driven over stream-json, so the CLI
    starts once per conversation instead of once per turn — lower latency, a
    warm prompt cache, and the process keeps the multi-turn memory itself (so we
    send only the new message each turn, not the whole history)."""

    def __init__(self, system_prompt):
        self._system = build_claude_system(system_prompt, TOOLS)
        self._proc = None

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [CLAUDE_BIN, "-p",
               "--input-format", "stream-json", "--output-format", "stream-json",
               "--include-partial-messages", "--verbose",
               "--system-prompt", self._system,
               *_claude_agent_flags()]  # native tools + permission mode (off/web/yolo)
        if CLAUDE_MODEL:
            cmd += ["--model", CLAUDE_MODEL]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, text, detect_tool=True):
        """Send one user message; yield ('text', tok) as the spoken reply streams,
        and ('control', ev) when the model calls a timer tool (so the coordinator,
        which owns the speaker, schedules/cancels the firing). Aven's other tools
        are executed by the MCP server; the model phrases the result itself."""
        try:
            self._ensure()
            msg = {"type": "user", "message": {"role": "user",
                   "content": [{"type": "text", "text": text}]}}
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            print(f"[claude] session send failed: {exc}", flush=True)
            self.close()
            yield ("text", "Sorry, I couldn't reach Claude.")
            return

        # We do NOT stream intermediate text: while using tools the model narrates
        # ("Let me search for the weather tool…") and that would be spoken. Only the
        # FINAL answer (the 'result' event, or the last assistant text) is spoken.
        # Timer tool calls are surfaced live as control events so the coordinator
        # schedules the firing during the turn.
        seen_tools = set()
        last_text = ""
        final = None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "result":
                if not ev.get("is_error"):
                    final = ev.get("result")
                break
            if etype == "assistant":
                blocks = ev.get("message", {}).get("content", [])
                text_here = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if text_here.strip():
                    last_text = text_here
                for b in blocks:            # surface tool calls (log) + timer controls
                    if b.get("type") == "tool_use" and b.get("id") not in seen_tools:
                        seen_tools.add(b.get("id"))
                        name = b.get("name") or "?"
                        yield ("tool_log", name.replace("mcp__aven__", ""))
                        ctrl = _timer_control(name, b.get("input"))
                        if ctrl:
                            yield ("control", ctrl)
        spoken = (final if (final and final.strip()) else last_text).strip()
        spoken = _strip_sources(spoken)   # web search loves to append a Sources: block
        if spoken:
            yield ("text", spoken)

    def reset(self):
        """Drop the conversation (used on 'clear') by restarting the process."""
        self.close()

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        self._proc = None


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


def run_turn(conn, tts_url, prompt, history, model, claude_session=None):
    """Stream LLM -> clauses -> TTS node -> relay PCM to the laptop.

    With the claude backend a persistent ClaudeSession owns the conversation, so
    we send only the new message (no history re-send). The rkllama backend uses
    the history list as before.
    """
    from websockets.sync.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed

    try:
        tts_ws = ws_connect(tts_url, max_size=None, open_timeout=5)
    except Exception as exc:  # noqa: BLE001
        conn.send(json.dumps({"type": "error", "message": f"Cannot reach TTS node at {tts_url}: {exc}"}))
        conn.send(json.dumps({"type": "done"}))
        return

    use_claude = claude_session is not None
    if not use_claude:
        history.append({"role": "user", "content": prompt})
    events: queue.Queue = queue.Queue()

    def first_stream():
        if use_claude:
            return claude_session.send(prompt, detect_tool=True)
        return stream_llm(history, model, TOOLS)

    def data_followup(calls):
        """Feed a data-tool result (weather/web search) back so the model answers in words."""
        data = [c for c in calls if c.get("name") in DATA_TOOLS]
        if use_claude:
            results = "; ".join(f"{c.get('name')} returned: {run_tool(c)}" for c in data)
            return claude_session.send(
                f"Tool results — {results}\nUsing only this, answer my question now in one"
                " or two short spoken sentences. Give only the information asked for; do not"
                " restate or mention any device actions.", detect_tool=False)
        calls = data
        history.append({
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": c.get("id") or f"call_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})},
            } for i, c in enumerate(calls)],
        })
        for i, c in enumerate(calls):
            history.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{i}",
                            "content": run_tool(c)})
        return stream_llm(history, model, None)

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
            for kind, val in first_stream():
                if kind == "text":
                    collected.append(val)
                    events.put(("llm", val))
                    buf += val
                    clauses, buf = extract_clauses(buf)
                    for clause in clauses:
                        events.put(("clause", clause))
                elif kind == "control":
                    # Claude/MCP path: a timer tool call -> forward the schedule/
                    # cancel event to the coordinator (it owns the speaker).
                    events.put(("control", val))
                elif kind == "tool_log":
                    events.put(("tool_log", val))   # forwarded to the coordinator's log
                elif kind == "tool":
                    # A batch may mix action tools (set_light/set_timer/say — fixed
                    # confirmations + control events) and data tools (weather/web —
                    # the model phrases the result). Run both so a mixed request
                    # ("turn on the light and what's the weather") works fully.
                    action_calls = [c for c in val if c.get("name") not in DATA_TOOLS]
                    data_calls = [c for c in val if c.get("name") in DATA_TOOLS]
                    # 1. Actions first: complete confirmation string(s) + control events.
                    if action_calls:
                        areply, controls = handle_tool_calls(action_calls)
                        for ctrl in controls:
                            events.put(("control", ctrl))
                        if areply:
                            piece = areply + (" " if data_calls else "")  # separate from data answer
                            collected.append(piece)
                            events.put(("llm", piece))
                            buf += areply + "\n"          # \n forces a full flush
                            clauses, buf = extract_clauses(buf)
                            for clause in clauses:
                                events.put(("clause", clause))
                    # 2. Data tools: the model streams the spoken answer.
                    if data_calls:
                        for kind2, val2 in data_followup(data_calls):
                            if kind2 == "text":
                                collected.append(val2)
                                events.put(("llm", val2))
                                buf += val2               # raw token fragments, no spaces
                                clauses, buf = extract_clauses(buf)
                                for clause in clauses:
                                    events.put(("clause", clause))
                    tail = buf.strip()
                    if tail:
                        events.put(("clause", tail))
                        buf = ""
                    tool_reply = "".join(collected)
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
            elif kind == "tool_log":
                conn.send(json.dumps({"type": "tool", "name": payload}))
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
                if not use_claude:
                    history.pop()
                conn.send(json.dumps({"type": "done"}))
                return

    conn.send(json.dumps({"type": "done"}))
    if not use_claude:                       # claude's process owns its own memory
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
        # One persistent claude process per connection (started lazily on the
        # first turn); None for the rkllama backend.
        claude_session = ClaudeSession(system_prompt) if BACKEND == "claude" else None
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
                    if claude_session is not None:
                        claude_session.reset()
                    conn.send(json.dumps({"type": "cleared"}))
                    print(f"[i] {peer}: history cleared", flush=True)
                    continue
                if data.get("command") == "pause_music":
                    conn.send(json.dumps({"type": "music", "paused": pause_for_wake()}))
                    continue
                if data.get("command") == "resume_music":
                    conn.send(json.dumps({"type": "music", "resumed": resume_after_wake()}))
                    continue
                prompt = (data.get("text") or "").strip()
                model = data.get("model") or default_model
                if not prompt:
                    continue
                print(f"[>] {peer}: {prompt}", flush=True)
                run_turn(conn, tts_url, prompt, history, model, claude_session)
        except ConnectionClosed:
            pass
        finally:
            if claude_session is not None:
                claude_session.close()
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

    if BACKEND == "claude":
        model = CLAUDE_MODEL or "cli default"
        backend_desc = f"claude CLI ({CLAUDE_BIN})"
        _write_mcp_config()            # Aven's tools -> native MCP tools for the CLI
        if CLAUDE_AGENT_MODE == "yolo":
            print("\033[31m⚠ claude_agent_mode=yolo: the brain can run ANY tool "
                  "(incl. Bash) on this host with no prompts.\033[0m", flush=True)
    elif BACKEND == "rkllm":
        if not os.path.exists(RKLLM_MODEL_PATH):
            print(f"\033[31mModel not found: {RKLLM_MODEL_PATH}\033[0m")
            sys.exit(1)
        model = os.path.basename(RKLLM_MODEL_PATH)
        backend_desc = "rkllm (local NPU, in-process)"
        threading.Thread(target=get_rkllm_model, daemon=True).start()   # preload
    else:
        model = pick_model(args.model)
        backend_desc = f"rkllama (via {LLM_URL})"
    tts_url = f"ws://{args.tts_host}:{args.tts_port}"
    handler = make_handler(tts_url, model, args.system)

    if BACKEND == "rkllama" and KEEPALIVE_MIN > 0:
        threading.Thread(target=keep_warm, args=(model, KEEPALIVE_MIN), daemon=True).start()

    print("\033[32mLLM node ready.\033[0m")
    print(f"  Backend   : {backend_desc}")
    if BACKEND == "claude":
        print(f"  Agent mode: {CLAUDE_AGENT_MODE}  (native tools/permissions)")
    print(f"  LLM model : {model}")
    print(f"  TTS node  : {tts_url}")
    if BACKEND == "rkllama":
        print(f"  Keep-warm : {'every %d min' % KEEPALIVE_MIN if KEEPALIVE_MIN > 0 else 'off'}")
    print(f"  Listening : ws://{args.host}:{args.port}  (laptop connects to ws://{ROCK5C_IP}:{args.port})")
    with serve(handler, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
