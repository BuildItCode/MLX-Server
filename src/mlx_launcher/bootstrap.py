"""Dependency self-check and install helpers.

The app's own pure-Python deps are guaranteed by the install method; this module
targets the *external* runtime dependency `mlx-lm` (the `mlx_lm.server` binary)
and the optional "install me as a global command" flow."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from .server import discovery

LogCb = Callable[[str], None]


# --- mlx_lm.server detection / install -----------------------------------

def find_mlx_server(engine: str = "mlx-lm") -> Optional[str]:
    return discovery.find_server_binary(engine)


def mlx_server_available(engine: str = "mlx-lm") -> bool:
    return find_mlx_server(engine) is not None


def pip_install_argv(package: str = "mlx-lm") -> list[str]:
    """Install into the *current* interpreter's environment."""
    return [sys.executable, "-m", "pip", "install", "--upgrade", package]


def pipx_inject_argv(package: str = "mlx-lm", into: str = "mlx-launcher") -> list[str]:
    return ["pipx", "inject", into, package]


# --- global install ------------------------------------------------------

def pipx_available() -> bool:
    return shutil.which("pipx") is not None


def project_root() -> Optional[Path]:
    """Repo root (the dir containing pyproject.toml), if this is an editable/source
    checkout. Returns None for non-source installs (e.g. a wheel inside pipx)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def global_install_argv() -> Optional[list[str]]:
    """Command to install the launcher globally via pipx, or None if not possible
    from here (no pipx, or not a source checkout)."""
    root = project_root()
    if root is None or not pipx_available():
        return None
    return ["pipx", "install", "--force", str(root)]


# --- generic streamed subprocess runner ----------------------------------

async def run_streamed(argv: list[str], on_log: LogCb, env: Optional[dict] = None) -> int:
    """Run a command, streaming combined stdout+stderr line-by-line to on_log.
    Returns the exit code (or a negative pseudo-code on spawn failure)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except FileNotFoundError:
        on_log(f"command not found: {argv[0]}")
        return -1
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        on_log(line.decode(errors="replace").rstrip("\n"))
    return await proc.wait()
