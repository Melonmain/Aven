#!/usr/bin/env python3
"""Shared config loader for every Aven service.

The whole repo is checked out on each board, so each service imports this module
(by adding the repo root to sys.path) and reads the single root ``config.yaml``.
Only dependency is PyYAML, which every service project already pulls in.

Typical use::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from config import load_config, service_addr

    cfg = load_config()
    host, port = service_addr("tts")
"""

import functools
import pathlib

import yaml

CONFIG_NAME = "config.yaml"


@functools.lru_cache(maxsize=1)
def _config_path():
    """Walk up from this file to find config.yaml at the repo root."""
    here = pathlib.Path(__file__).resolve()
    for directory in (here.parent, *here.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {CONFIG_NAME} near {here}")


@functools.lru_cache(maxsize=1)
def load_config():
    """Load and cache the parsed config.yaml as a dict."""
    with open(_config_path(), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def service_addr(name):
    """Return (host, port) for a named service entry under ``services``."""
    svc = load_config()["services"][name]
    return svc["host"], svc["port"]
