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
import sys
from collections import deque
from enum import Enum
from typing import Callable, Optional

from ..._util import kill_process_group, process_group_kwargs, terminate_process_group
from .flags import build_argv
from ...models.config import ServerConfig
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
        self._starting = False  # guards against a concurrent double-start spawning two procs

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
        # never overwrite a live process (orphan) or spawn twice under concurrent callers
        # (e.g. a side-chat load racing a dashboard launch of the same profile).
        if self.is_running or self._starting:
            return
        self._starting = True
        try:
            await self._do_start()
        finally:
            self._starting = False

    async def _do_start(self) -> None:
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

        launch_cfg = self.cfg
        if launch_cfg.engine == "llama-cpp":
            # the model field may be a folder (LM Studio / HF layout) → resolve to the .gguf
            resolved = discovery.resolve_gguf(launch_cfg.model)
            if resolved != launch_cfg.model:
                self._emit_log("meta", f"resolved GGUF → {resolved}")
            launch_cfg = launch_cfg.model_copy(update={"model": resolved})
        argv = build_argv(launch_cfg, mlx)
        # vision/omni models: auto-load the sibling projector unless one is already set
        if launch_cfg.engine == "llama-cpp" and "--mmproj" not in argv:
            mmproj = discovery.find_mmproj(launch_cfg.model)
            if mmproj:
                argv += ["--mmproj", mmproj]
                self._emit_log("meta", f"vision projector → {mmproj}")
        self._emit_log("meta", "$ " + " ".join(argv))
        self._set_status(ServerStatus.STARTING, f"launching {binary} …")

        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            limit=2 ** 20,  # 1 MB line buffer — model servers emit long progress/JSON/traceback lines
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            **process_group_kwargs(),  # own process group → clean tree kill (POSIX + Windows)
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
            except asyncio.CancelledError:
                break
            except ValueError:
                # A single line exceeded the buffer limit. readline() drops it from the buffer
                # and raises — keep reading subsequent lines instead of going dark forever.
                self._emit_log(name, "⋯ (overlong log line skipped)")
                continue
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
        if self.cfg.engine == "llama-cpp" and re.search(
                r"(failed to load model|error loading model|llama_model_load|gguf)", text, re.I):
            return "couldn't load the GGUF — check the model path/quant and that llama-server supports this architecture"
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
        terminate_process_group(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            kill_process_group(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
        self._cancel_tasks()

    def terminate(self) -> None:
        """Best-effort synchronous graceful stop, for app shutdown (no await)."""
        self._stopping = True
        proc = self.proc
        if proc is not None and proc.returncode is None:
            terminate_process_group(proc)

    def is_alive(self) -> bool:
        """Liveness checked via the OS — works even when the event loop is no longer
        reaping the subprocess (e.g. during app shutdown), unlike `is_running`."""
        proc = self.proc
        if proc is None:
            return False
        if proc.returncode is not None:  # the loop's watcher already reaped it
            return False
        if sys.platform == "win32":
            # os.kill(pid, 0) *terminates* the process on Windows — never probe with it.
            return True
        # During the synchronous shutdown loop the event loop can't run, so a SIGTERM'd child
        # becomes a zombie that os.kill(pid, 0) still reports as alive — which would waste the
        # full grace period on every quit. Once we're stopping, reap it directly (WNOHANG).
        # Gated on `_stopping` so we don't race the loop's own reaper during normal operation.
        if self._stopping:
            try:
                reaped, _ = os.waitpid(proc.pid, os.WNOHANG)
            except ChildProcessError:
                return False  # already reaped elsewhere
            except OSError:
                return True
            return reaped == 0  # 0 → still running; nonzero → we just reaped it (it's dead)
        try:
            os.kill(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def kill_now(self) -> None:
        """Synchronous force-kill — last-resort shutdown."""
        self._stopping = True
        proc = self.proc
        if proc is not None:
            kill_process_group(proc)

    def _cancel_tasks(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks = []
