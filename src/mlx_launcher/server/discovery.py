"""Locate the engine server binary and check port availability."""

from __future__ import annotations

import shutil
import socket
from typing import Optional

# Each engine ships its own console script. The text-LLM runtime is mlx-lm
# (`mlx_lm.server`); vision-language models are served by mlx-vlm
# (`mlx_vlm.server`); vllm-mlx is a vLLM-style MLX server (`vllm-mlx serve …`).
# Keyed by ServerConfig.engine.
SERVER_BINARIES: dict[str, str] = {
    "mlx-lm": "mlx_lm.server",
    "mlx-vlm": "mlx_vlm.server",
    "vllm-mlx": "vllm-mlx",
}
SERVER_BINARY = SERVER_BINARIES["mlx-lm"]  # back-compat default


def binary_name(engine: str = "mlx-lm") -> str:
    """The console-script name for an engine (defaults to mlx-lm)."""
    return SERVER_BINARIES.get(engine, SERVER_BINARY)


def find_server_binary(engine: str = "mlx-lm") -> Optional[str]:
    """Absolute path to the engine's server binary on PATH, or None.

    We resolve the console script (which carries its own interpreter) rather
    than assuming `python -m ...`, because the engine may be installed in a
    different environment than this app's venv.
    """
    return shutil.which(binary_name(engine))


def is_port_free(host: str, port: int) -> bool:
    """True if nothing is currently listening on host:port.

    Detects an existing listener via a short connect attempt — the common case
    is "I already have a server on 8080".
    """
    target = "127.0.0.1" if host in ("0.0.0.0", "", "*") else host
    family = socket.AF_INET6 if ":" in target else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            s.connect((target, port))
        return False  # connection accepted → something is listening
    except OSError:
        return True
