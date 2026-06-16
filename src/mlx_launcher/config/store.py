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
    except Exception:  # noqa: BLE001 — unparseable JSON: back up + start fresh
        _backup(path)
        return ConfigFile()
    try:
        return ConfigFile.model_validate(data)
    except Exception:  # noqa: BLE001
        # One bad server (e.g. an out-of-range numeric field) must NOT wipe every profile.
        # Back up the original, then salvage whichever servers still validate.
        _backup(path)
        return _salvage(data)


def _backup(path: Path) -> None:
    """Best-effort: move the unreadable file aside so the user can recover it."""
    try:
        path.rename(path.with_name(f"servers.corrupt-{int(time.time())}.json"))
    except Exception:  # noqa: BLE001
        pass


def _salvage(data) -> ConfigFile:
    """Rebuild a ConfigFile from a dict that failed whole-file validation, keeping every
    server that validates on its own and dropping only the bad ones."""
    if not isinstance(data, dict):
        return ConfigFile()
    servers = []
    for raw in data.get("servers") or []:
        try:
            servers.append(ServerConfig.model_validate(raw))
        except Exception:  # noqa: BLE001 — drop only this one profile
            pass
    try:
        settings = AppSettings.model_validate(data.get("settings") or {})
    except Exception:  # noqa: BLE001
        settings = AppSettings()
    sv = data.get("schema_version")
    return ConfigFile(schema_version=sv if isinstance(sv, int) else 1, servers=servers, settings=settings)


def save(cfg: ConfigFile) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = config_path()
    tmp = path.with_name(path.name + ".tmp")
    # flush + fsync before the atomic rename so a crash/power-loss can't leave the temp
    # (and thus the renamed file) truncated — os.replace alone doesn't guarantee the data
    # blocks are on disk before the directory entry is swapped.
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(cfg.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())
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
