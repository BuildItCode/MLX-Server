"""Locate the engine server binary and check port availability."""

from __future__ import annotations

import glob
import os
import re
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
    "llama-cpp": "llama-server",  # llama.cpp's OpenAI-compatible server (brew install llama.cpp)
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


def resolve_gguf(model: str) -> str:
    """Resolve a llama.cpp model reference to the actual .gguf FILE to pass to `-m`.

    LM Studio / HuggingFace store a GGUF model as a DIRECTORY of one or more .gguf files
    (just like MLX models are folders), but `llama-server -m` needs a specific file — the
    first shard of a sharded model pulls in the rest. Returns `model` unchanged when it's
    already a file, an HF repo id (`org/repo`), or no .gguf can be found. The vision
    projector (`mmproj-*.gguf`) is skipped here — it's loaded via `--mmproj` (find_mmproj)."""
    path = os.path.expanduser(model or "")
    if not os.path.isdir(path):
        return model
    ggufs = (sorted(glob.glob(os.path.join(path, "*.gguf")))
             or sorted(glob.glob(os.path.join(path, "*", "*.gguf"))))
    main = [g for g in ggufs if "mmproj" not in os.path.basename(g).lower()]
    if not main:
        return model
    shards = [g for g in main if re.search(r"-0*1-of-\d+\.gguf$", os.path.basename(g))]
    return (shards or main)[0]


def find_mmproj(model: str) -> Optional[str]:
    """The multimodal projector (`mmproj-*.gguf`) sitting beside a llama.cpp model, if any
    — passed to `llama-server --mmproj` so vision/omni models can actually see images."""
    path = os.path.expanduser(model or "")
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isdir(folder):
        return None
    hits = sorted(glob.glob(os.path.join(folder, "*mmproj*.gguf")))
    return hits[0] if hits else None


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
