"""Load/save the on-disk config (~/.config/mlx-launcher/servers.json).

Atomic writes, corruption-tolerant loads (a bad file is backed up and a fresh
config is returned rather than crashing the app)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from ...models.config import AppSettings, ConfigFile, ServerConfig


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


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: temp file → flush + fsync → os.replace. The fsync
    before the rename makes the data durable before the directory entry is swapped, so a
    crash/power-loss can't leave a truncated file. Shared by both JSON stores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_aside(path: Path, prefix: str) -> None:
    """Best-effort: move an unreadable file aside as `<prefix>-<unixtime>.json` so the user can
    recover it before we start fresh / salvage."""
    try:
        path.rename(path.with_name(f"{prefix}-{int(time.time())}.json"))
    except Exception:  # noqa: BLE001
        pass


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
    backup_aside(path, "servers.corrupt")


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
    path = config_path()
    atomic_write_text(path, cfg.model_dump_json(indent=2))
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


# --- single-writer mutation ---------------------------------------------

_write_lock = asyncio.Lock()


async def mutate(apply: Callable[[ConfigFile], Optional[ConfigFile]]) -> ConfigFile:
    """Read-modify-write the config store under a process-wide lock: load → apply → save.

    Once the backend service can be driven by several frontends at once, an unserialized
    load/modify/save (the previous "re-read before save" convention) can lose a concurrent
    writer's changes. Route every write through this so writers are serialized. ``apply`` mutates
    the loaded file in place (or returns a replacement). Returns the saved ``ConfigFile``."""
    async with _write_lock:
        data = load()
        replaced = apply(data)
        if replaced is not None:
            data = replaced
        save(data)
        return data
