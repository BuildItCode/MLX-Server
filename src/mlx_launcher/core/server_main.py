"""``lis-backend`` entrypoint + backend discovery.

Starts the HTTP+SSE service on an ephemeral ``127.0.0.1`` port and writes a discovery file at
``~/.config/mlx-launcher/backend.json`` (mode 0600) holding ``{pid, port, token, version}`` so a
frontend can find and authenticate to a running backend. The bearer token is minted here and never
logged.

The full spawn/daemon lifecycle (TUI-spawned child vs. shared ACP daemon, client ref-counting,
graceful model-server teardown) is wired when the frontends connect; this module provides the
runnable service + discovery primitives they build on."""

from __future__ import annotations

import json
import os
import secrets
import socket
from pathlib import Path
from typing import Optional

from .. import __version__
from .persistence.config import atomic_write_text, config_dir
from .service import create_app


def backend_info_path() -> Path:
    return config_dir() / "backend.json"


def read_backend_info() -> Optional[dict]:
    """The discovery record for a (possibly) running backend, or None if absent/unreadable."""
    try:
        return json.loads(backend_info_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_backend_info(port: int, token: str) -> Path:
    path = backend_info_path()
    atomic_write_text(path, json.dumps(
        {"pid": os.getpid(), "port": port, "token": token, "version": __version__}))
    try:
        os.chmod(path, 0o600)  # the token is a secret — owner-only
    except OSError:
        pass
    return path


def _free_port() -> int:
    """Pick a free loopback port. (Small race between close and uvicorn bind — fine for local.)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _remove_backend_info() -> None:
    try:
        backend_info_path().unlink()
    except OSError:
        pass


def main() -> None:
    import uvicorn

    token = secrets.token_hex(16)
    port = int(os.environ.get("MLX_BACKEND_PORT") or _free_port())
    # Remove the discovery file from the lifespan shutdown: on SIGTERM/SIGINT uvicorn runs the
    # lifespan shutdown but uvicorn.run() may not return, so a finally here wouldn't fire.
    app = create_app(token=token, on_shutdown=_remove_backend_info)
    write_backend_info(port, token)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        _remove_backend_info()  # belt-and-suspenders for a clean return


if __name__ == "__main__":
    main()
