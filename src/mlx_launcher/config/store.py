"""Load/save the on-disk config (~/.config/mlx-launcher/servers.json).

Atomic writes, corruption-tolerant loads (a bad file is backed up and a fresh
config is returned rather than crashing the app)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .models import AppSettings, ConfigFile, ServerConfig


def config_dir() -> Path:
    """`$XDG_CONFIG_HOME/mlx-launcher` or `~/.config/mlx-launcher`.

    We construct the XDG path explicitly rather than using platformdirs' macOS
    default (`~/Library/Application Support`) because the spec calls for ~/.config.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "mlx-launcher"


def config_path() -> Path:
    return config_dir() / "servers.json"


def load() -> ConfigFile:
    path = config_path()
    if not path.exists():
        return ConfigFile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConfigFile.model_validate(data)
    except Exception:
        # Back up the unreadable file so the user can recover it, then start fresh.
        try:
            backup = path.with_name(f"servers.corrupt-{int(time.time())}.json")
            path.rename(backup)
        except Exception:
            pass
        return ConfigFile()


def save(cfg: ConfigFile) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = config_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# --- convenience helpers -------------------------------------------------

def get_server(cfg: ConfigFile, server_id: str) -> Optional[ServerConfig]:
    return next((s for s in cfg.servers if s.id == server_id), None)


def upsert_server(cfg: ConfigFile, server: ServerConfig) -> None:
    for i, s in enumerate(cfg.servers):
        if s.id == server.id:
            cfg.servers[i] = server
            return
    cfg.servers.append(server)


def delete_server(cfg: ConfigFile, server_id: str) -> None:
    cfg.servers = [s for s in cfg.servers if s.id != server_id]
    if cfg.settings.last_used_id == server_id:
        cfg.settings.last_used_id = None


def find_server_by_id(server_id: str) -> Optional[ServerConfig]:
    """Used by the ACP entrypoint to resolve `--config-id`."""
    return get_server(load(), server_id)
