"""The thin wire client both frontends use to drive the backend over HTTP + SSE.

This is the *only* thing a frontend needs in order to reach the backend — it speaks the documented
REST + Server-Sent-Events contract and nothing else (no import of ``engine``/``core`` internals).
``BackendClient`` wraps the endpoints; ``discover``/``spawn``/``connect`` find or start a local
backend via the ``backend.json`` discovery file (the same file ``core.server_main`` writes).

A replacement frontend in any language only has to reimplement this surface against the same wire
contract."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx


class BackendError(RuntimeError):
    """A non-2xx response from the backend (carries status + body)."""


def backend_info_path() -> Path:
    """``$XDG_CONFIG_HOME/mlx-launcher/backend.json`` (or ``~/.config/...``) — the discovery file the
    backend writes and the client reads. Computed here so the client stays independent of core."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "mlx-launcher" / "backend.json"


def read_backend_info() -> Optional[dict]:
    try:
        return json.loads(backend_info_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class BackendClient:
    """Async client for the backend's REST + SSE API. Construct directly with ``(base_url, token)``,
    or via :func:`connect` for discovery/spawn. Pass ``transport`` (an ``httpx`` transport) in tests
    to drive an in-process ASGI app."""

    def __init__(self, base_url: str, token: Optional[str] = None, *, transport=None) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._transport = transport

    def _new(self, **kw) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        return httpx.AsyncClient(base_url=self.base_url, headers=headers, transport=self._transport, **kw)

    @staticmethod
    def _ok(resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            raise BackendError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp

    async def _get(self, path: str, **params) -> dict:
        async with self._new(timeout=30.0) as c:
            return self._ok(await c.get(path, params=params or None)).json()

    async def _post(self, path: str, body: Optional[dict] = None) -> dict:
        async with self._new(timeout=30.0) as c:
            r = self._ok(await c.post(path, json=body or {}))
            return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else {}

    async def _send(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        async with self._new(timeout=30.0) as c:
            r = self._ok(await c.request(method, path, json=body))
            return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else {}

    # --- handshake -------------------------------------------------------

    async def healthz(self) -> dict:
        return await self._get("/healthz")

    # --- sessions --------------------------------------------------------

    async def create_session(self, *, server_id: Optional[str] = None, project_id: Optional[str] = None) -> dict:
        return await self._post("/sessions", {"server_id": server_id, "project_id": project_id})

    async def list_sessions(self, project_id: Optional[str] = None) -> list:
        return (await self._get("/sessions", **({"project_id": project_id} if project_id else {}))).get("chats", [])

    async def get_session(self, session_id: str) -> dict:
        return (await self._get(f"/sessions/{session_id}")).get("chat", {})

    async def delete_session(self, session_id: str) -> None:
        await self._send("DELETE", f"/sessions/{session_id}")

    # --- runs ------------------------------------------------------------

    async def start_run(self, session_id: str, text: str, *, attachments: Optional[list] = None,
                        kind: str = "chat") -> str:
        out = await self._post(f"/sessions/{session_id}/runs",
                               {"text": text, "attachments": attachments or [], "kind": kind})
        return out["run_id"]

    async def stream_run(self, session_id: str, run_id: str) -> AsyncIterator[tuple[str, dict]]:
        """Yield ``(event_type, data)`` for a run's SSE stream until it closes."""
        async for item in self._sse(f"/sessions/{session_id}/runs/{run_id}/events"):
            yield item

    async def cancel_run(self, session_id: str, run_id: str) -> None:
        await self._post(f"/sessions/{session_id}/runs/{run_id}/cancel")

    async def answer_permission(self, session_id: str, run_id: str, permission_id: str, decision: str) -> None:
        await self._post(f"/sessions/{session_id}/runs/{run_id}/permissions/{permission_id}",
                         {"decision": decision})

    # --- resources (backed by the resource/lifecycle endpoints) ----------

    async def list_servers(self) -> list:
        return (await self._get("/servers")).get("servers", [])

    async def upsert_server(self, server: dict) -> dict:
        sid = server.get("id")
        return await self._send("PUT" if sid else "POST", f"/servers/{sid}" if sid else "/servers", server)

    async def delete_server(self, server_id: str) -> None:
        await self._send("DELETE", f"/servers/{server_id}")

    async def start_server(self, server_id: str, *, bump: bool = False) -> dict:
        """Start a model server; returns its status snapshot (incl. the actual ``base_url``).
        ``bump=True`` runs it alongside others on a free port (subagent side-pane)."""
        return await self._post(f"/servers/{server_id}/start", {"bump": bump})

    async def stop_server(self, server_id: str) -> None:
        await self._post(f"/servers/{server_id}/stop")

    async def restart_server(self, server_id: str) -> dict:
        return await self._post(f"/servers/{server_id}/restart")

    async def server_status(self, server_id: str) -> dict:
        return await self._get(f"/servers/{server_id}/status")

    async def server_models(self, server_id: str) -> dict:
        return await self._get(f"/servers/{server_id}/models")

    async def all_server_status(self) -> list:
        return (await self._get("/servers/status")).get("servers", [])

    async def stream_server_logs(self, server_id: str) -> AsyncIterator[tuple[str, dict]]:
        async for item in self._sse(f"/servers/{server_id}/logs"):
            yield item

    async def list_resource(self, kind: str) -> list:
        """Generic list for projects | mcp-servers | subagents | skills."""
        return (await self._get(f"/{kind}")).get(kind.replace("-", "_"), [])

    async def upsert_resource(self, kind: str, obj: dict) -> dict:
        """Create/update a projects | mcp-servers | subagents entry (PUT when it has an id)."""
        oid = obj.get("id")
        return await self._send("PUT" if oid else "POST", f"/{kind}/{oid}" if oid else f"/{kind}", obj)

    async def delete_resource(self, kind: str, resource_id: str) -> None:
        await self._send("DELETE", f"/{kind}/{resource_id}")

    async def patch_session(self, session_id: str, changes: dict) -> dict:
        return (await self._send("PATCH", f"/sessions/{session_id}", changes)).get("chat", {})

    async def session_context(self, session_id: str) -> dict:
        return await self._get(f"/sessions/{session_id}/context")

    async def get_settings(self) -> dict:
        return await self._get("/settings")

    async def patch_settings(self, changes: dict) -> dict:
        return await self._send("PATCH", "/settings", changes)

    # --- SSE -------------------------------------------------------------

    async def _sse(self, path: str) -> AsyncIterator[tuple[str, dict]]:
        async with self._new(timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)) as c:
            async with c.stream("GET", path) as resp:
                self._ok(resp)
                event = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            data = {"raw": raw}
                        yield (event or "message", data)
                        event = None


# --- discovery / spawn ---------------------------------------------------

async def _healthy(info: dict, *, timeout: float = 2.0) -> bool:
    try:
        client = BackendClient(f"http://127.0.0.1:{info['port']}", info.get("token"))
        async with client._new(timeout=timeout) as c:
            r = await c.get("/healthz")
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def discover() -> Optional[BackendClient]:
    """Return a client for a healthy already-running backend, or None."""
    info = read_backend_info()
    if info and "port" in info and await _healthy(info):
        return BackendClient(f"http://127.0.0.1:{info['port']}", info.get("token"))
    return None


async def spawn(*, wait: float = 20.0) -> BackendClient:
    """Start ``lis-backend`` as a child process and wait for it to become healthy. Uses
    ``python -m mlx_launcher.core.server_main`` so it works without the console script on PATH."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mlx_launcher.core.server_main",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            raise BackendError(f"lis-backend exited early (code {proc.returncode})")
        client = await discover()
        if client is not None:
            return client
        await asyncio.sleep(0.2)
    proc.terminate()
    raise BackendError("lis-backend did not become healthy in time")


async def connect() -> BackendClient:
    """Connect to a running backend, or spawn one if none is healthy."""
    return await discover() or await spawn()
