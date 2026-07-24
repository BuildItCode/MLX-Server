"""Small cross-platform / low-level helpers shared across modules, kept in one place to
avoid copy-paste drift (Apple-Silicon detection and the native-stderr mute today)."""

from __future__ import annotations

import contextlib
import os
import platform
import signal
import subprocess
import sys


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@contextlib.contextmanager
def silence_native_stderr():
    """Redirect the OS-level stderr fd to /dev/null for the duration, always restoring it.

    Native layers (e.g. hf_xet's Rust transfer used by huggingface_hub / mlx_whisper) print
    straight to fd 2, bypassing Python logging and `sys.stderr` — which would glitch the TUI.
    Callers run this in a worker thread while the screen renders on stdout, so muting fd 2
    briefly is safe; the fd is restored on every exit path (and the dup'd fd never leaks)."""
    saved = None
    devnull = None
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:  # noqa: BLE001 — if dup/dup2 isn't available, just don't silence
        if saved is not None:
            try:
                os.close(saved)
            except Exception:  # noqa: BLE001
                pass
            saved = None
    finally:
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:  # noqa: BLE001
                pass
    try:
        yield
    finally:
        if saved is not None:
            try:
                os.dup2(saved, 2)
            finally:
                os.close(saved)


# --- process groups (so a stop can kill the whole tree) ------------------

def process_group_kwargs() -> dict:
    """subprocess kwargs that put a child in its own process group. POSIX → a new session;
    Windows → a new process group."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_group(proc) -> None:
    """Graceful stop of `proc`'s process group: SIGTERM the POSIX group, or `.terminate()` on
    Windows (no POSIX signals / groups there). `proc` is a Popen / asyncio subprocess."""
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def kill_process_group(proc) -> None:
    """Force-kill `proc`'s whole tree: SIGKILL the POSIX group, or `taskkill /F /T` on Windows."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
