"""Tests for the wire client (mlx_launcher.client.BackendClient) against the in-process service.

Validates the REST + SSE round-trip end-to-end: a frontend creates a session, starts a run, and
consumes the event stream — exactly what the TUI/ACP frontends do over the wire."""

import httpx

from mlx_launcher.client import BackendClient, backend_info_path, read_backend_info
from mlx_launcher.core.persistence import config as config_store
from mlx_launcher.core.service import create_app
from mlx_launcher.models import ServerConfig


class FakeEngine:
    def __init__(self, stream_script):
        self.stream_script = list(stream_script)

    async def chat(self, messages, tools=None, *, read_timeout=600.0):
        return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    async def stream_chat(self, messages, *, cancel=None):
        for item in self.stream_script:
            yield item


def _client_for(app):
    return BackendClient("http://t", token=None, transport=httpx.ASGITransport(app=app))


async def test_client_drives_a_streaming_run(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await config_store.mutate(lambda f: config_store.upsert_server(
        f, ServerConfig(id="s1", name="srv", model="llama", engine="llama-cpp", host="127.0.0.1", port=8080)))
    eng = FakeEngine([("content", "hi "), ("content", "there"), ("finish", "stop")])
    client = _client_for(create_app(engine_factory=lambda s: eng))

    assert (await client.healthz())["status"] == "ok"
    sess = await client.create_session(server_id="s1")
    sid = sess["session_id"]
    assert sess["chat"]["model"] == "llama"

    listed = await client.list_sessions()
    assert any(c["id"] == sid for c in listed)

    rid = await client.start_run(sid, "hello")
    events = [e async for e in client.stream_run(sid, rid)]
    types = [t for t, _ in events]
    assert "content" in types and "finish" in types
    assert "".join(d["text"] for t, d in events if t == "content") == "hi there"

    chat = await client.get_session(sid)
    assert chat["messages"][-1]["text"] == "hi there"


async def test_resource_crud_and_session_patch(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    client = _client_for(create_app())
    # create a server profile over the wire
    created = await client.upsert_server(
        {"name": "prof", "model": "llama", "engine": "llama-cpp", "host": "127.0.0.1", "port": 8080})
    server_id = created["server"]["id"]
    assert any(s["name"] == "prof" for s in await client.list_servers())
    # a session bound to it, then patch chat settings
    sid = (await client.create_session(server_id=server_id))["session_id"]
    chat = (await client._send("PATCH", f"/sessions/{sid}", {"mode": "plan", "coding": True}))["chat"]
    assert chat["mode"] == "plan" and chat["coding"] is True
    # settings round-trip
    assert "theme" in await client.get_settings()
    assert (await client.patch_settings({"theme": "custom"}))["theme"] == "custom"
    # a project over the wire
    await client._send("POST", "/projects", {"name": "P", "working_dir": str(tmp_path)})
    assert any(p["name"] == "P" for p in await client.list_resource("projects"))


def test_backend_info_path_uses_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert backend_info_path() == tmp_path / "mlx-launcher" / "backend.json"
    assert read_backend_info() is None  # absent → None, not a crash
