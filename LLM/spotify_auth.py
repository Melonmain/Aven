#!/usr/bin/env python3
"""One-time Spotify sign-in for the play_music tool.

Reads SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET / SPOTIPY_REDIRECT_URI from the
environment (set them in ../.env.local) and writes the OAuth token cache next to
this file (.spotify_cache), which llm_server.py reuses. Re-run if the token is
revoked or the scopes change. Needs a Spotify Premium account.

Run on the LLM board:
    cd LLM
    set -a; . ../.env.local; set +a        # load the SPOTIPY_* vars
    uv run python spotify_auth.py
It prints a URL — open it in any browser (on any device), authorize, then paste
the URL you land on (http://127.0.0.1:8888/callback?code=...) back here.
"""
import os
import pathlib
import sys

from spotipy.oauth2 import SpotifyOAuth

if not (os.environ.get("SPOTIPY_CLIENT_ID") and os.environ.get("SPOTIPY_CLIENT_SECRET")):
    sys.exit("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET first (see ../.env.local).")

cache = str(pathlib.Path(__file__).resolve().parent / ".spotify_cache")
auth = SpotifyOAuth(
    # recently-played lets resume_music restart the last track (in its playlist/
    # album context) after the live session has expired — "continue" after a gap.
    scope="user-modify-playback-state user-read-playback-state "
          "user-read-recently-played",
    redirect_uri=os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
    cache_path=cache, open_browser=False)

print("1) Open this URL in a browser and authorize:\n\n   " + auth.get_authorize_url() + "\n")
redirect = input("2) Paste the full URL you were redirected to: ").strip()
auth.get_access_token(auth.parse_response_code(redirect), as_dict=False)
print(f"\nDone — token cached at {cache}. Restart llm_server.")
