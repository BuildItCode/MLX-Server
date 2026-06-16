"""Manage the lifecycle of one mlx_lm.server subprocess.

Spawns the server in its own process group, streams stdout/stderr line-by-line,
detects readiness over HTTP, and stops it cleanly (TERM then KILL the group).

The manager buffers log lines and tracks status, and lets any number of UI
subscribers attach/detach. That way a launched server keeps running when you
navigate away from its screen, and re-opening it replays the buffered log."""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import signal
from collections import deque
from enum import Enum
from typing import Callable, Optional

from ..config.flags import build_argv
from ..config.models import ServerConfig
from . import discovery
from .readiness import wait_until_ready


class ServerStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    STOPPED = "stopped"
    ERROR = "error"


class BinaryNotFound(Exception):
    pass


class PortInUse(Exception):
    pass


LogCb = Callable[[str, str], None]  # (stream_name, line)
StatusCb = Callable[[ServerStatus, str], None]  # (status, message)


class ServerManager:
    def __init__(
        self,
        cfg: ServerConfig,
        *,
        mlx_override: Optional[str] = None,
        ready_timeout: float = 180.0,
        log_maxlen: int = 4000,
    ) -> None:
        self.cfg = cfg
        self._mlx_override = mlx_override
        self._ready_timeout = ready_timeout
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.status = ServerStatus.IDLE
        self.status_message = ""
        self.mlx_path: Optional[str] = None
        self.log_buffer: deque[tuple[str, str]] = deque(maxlen=log_maxlen)
        self._subs: dict[int, tuple[LogCb, StatusCb]] = {}
        self._sub_ids = itertools.count(1)
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    # --- subscriptions ---------------------------------------------------

    def subscribe(self, on_log: LogCb, on_status: StatusCb) -> int:
        token = next(self._sub_ids)
        self._subs[token] = (on_log, on_status)
        return token

    def unsubscribe(self, token: int) -> None:
        self._subs.pop(token, None)

    def _emit_log(self, stream: str, line: str) -> None:
        self.log_buffer.append((stream, line))
        for on_log, _ in list(self._subs.values()):
            try:
                on_log(stream, line)
            except Exception:
                pass

    def _set_status(self, status: ServerStatus, message: str = "") -> None:
        self.status = status
        self.status_message = message
        for _, on_status in list(self._subs.values()):
            try:
                on_status(status, message)
            except Exception:
                pass

    # --- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self.is_running:
            return  # already running — never overwrite a live process (it would orphan it)
        binary = discovery.binary_name(self.cfg.engine)
        mlx = self._mlx_override or self.cfg.mlx_server_path or discovery.find_server_binary(self.cfg.engine)
        if not mlx:
            self._set_status(ServerStatus.ERROR, f"{binary} not found on PATH")
            raise BinaryNotFound(f"{binary} not found on PATH")
        self.mlx_path = mlx

        if not discovery.is_port_free(self.cfg.host, self.cfg.port):
            msg = f"Port {self.cfg.port} is already in use"
            self._set_status(ServerStatus.ERROR, msg)
            raise PortInUse(msg)

        argv = build_argv(self.cfg, mlx)
        self._emit_log("meta", "$ " + " ".join(argv))
        self._set_status(ServerStatus.STARTING, f"launching {binary} …")

        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group → clean group kill
            env=os.environ.copy(),
        )
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._read_stream(self.proc.stdout, "stdout")),
            asyncio.create_task(self._read_stream(self.proc.stderr, "stderr")),
            asyncio.create_task(self._watch_exit()),
            asyncio.create_task(self._wait_ready()),
        ]

    async def _read_stream(self, stream: Optional[asyncio.StreamReader], name: str) -> None:
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except (asyncio.CancelledError, ValueError):
                break
            if not line:
                break
            self._emit_log(name, line.decode(errors="replace").rstrip("\n"))

    async def _wait_ready(self) -> None:
        # Keep probing as long as the process is alive: a large model can take
        # longer than one timeout window to load, and we must not leave the UI
        # stuck on STARTING forever when the probe window elapses.
        while self.is_running and not self._stopping and self.status == ServerStatus.STARTING:
            ok = await wait_until_ready(
                self.cfg.server_url(),
                timeout=self._ready_timeout,
                should_continue=lambda: self.is_running and not self._stopping,
            )
            if ok:
                if self.status == ServerStatus.STARTING:
                    self._set_status(ServerStatus.READY, "server is ready")
                return
            if not self.is_running or self._stopping:
                return
            # timed out but the process is still alive → keep waiting, but say so
            self._set_status(ServerStatus.STARTING, "still loading — readiness probe hasn't answered yet")

    async def _watch_exit(self) -> None:
        assert self.proc is not None
        rc = await self.proc.wait()
        await asyncio.sleep(0.15)  # let readers drain remaining output
        if self._stopping:
            self._set_status(ServerStatus.STOPPED, "server stopped")
        else:
            self._set_status(ServerStatus.ERROR, self._exit_message(rc))

    def _exit_message(self, rc: int) -> str:
        """Turn a crashed-on-startup traceback into a plain-English reason."""
        text = "\n".join(line for _, line in list(self.log_buffer)[-120:])
        m = re.search(r"Model type (\S+?) not supported", text)
        if m or re.search(r"No module named 'mlx_v?lm\.models\.", text):
            arch = m.group(1) if m else "this model"
            if self.cfg.engine == "mlx-vlm":
                return f"architecture '{arch}' isn't supported by your mlx-vlm — update it (pip install -U mlx-vlm)"
            if self.cfg.engine == "vllm-mlx":
                return f"architecture '{arch}' isn't supported by your vllm-mlx — update it (uv tool upgrade vllm-mlx)"
            return (
                f"architecture '{arch}' isn't supported by mlx-lm — update it "
                "(uv tool upgrade mlx-lm), or switch the engine to mlx-vlm if this is a vision model"
            )
        if "trust_remote_code" in text.lower():
            return "model needs trust-remote-code — enable it on the editor's Advanced tab"
        if re.search(r"(No such file or directory|does not exist|is not a directory|Repository Not Found)", text, re.I):
            return "could not load the model — check the model path/repo id"
        if "Address already in use" in text or "EADDRINUSE" in text:
            return f"port {self.cfg.port} is already in use"
        return f"server exited (code {rc}) — see the log above"

    async def stop(self) -> None:
        self._stopping = True
        proc = self.proc
        if proc is None or proc.returncode is not None:
            self._set_status(ServerStatus.STOPPED, "server stopped")
            self._cancel_tasks()
            return
        self._signal_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._signal_group(proc, signal.SIGKILL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
        self._cancel_tasks()

    def terminate(self) -> None:
        """Best-effort synchronous SIGTERM, for app shutdown (no await)."""
        self._stopping = True
        proc = self.proc
        if proc is not None and proc.returncode is None:
            self._signal_group(proc, signal.SIGTERM)

    def is_alive(self) -> bool:
        """Liveness checked via the OS — works even when the event loop is no longer
        reaping the subprocess (e.g. during app shutdown), unlike `is_running`."""
        proc = self.proc
        if proc is None:
            return False
        try:
            os.kill(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def kill_now(self) -> None:
        """Synchronous SIGKILL of the process group — last-resort shutdown."""
        self._stopping = True
        proc = self.proc
        if proc is not None:
            self._signal_group(proc, signal.SIGKILL)

    @staticmethod
    def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _cancel_tasks(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks = []
