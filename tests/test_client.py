"""Tests for the wire client (omnicode.client.BackendClient) against the in-process service.

Validates the REST + SSE round-trip end-to-end: a frontend creates a session, starts a run, and
consumes the event stream — exactly what the TUI/ACP frontends do over the wire."""

import httpx

from omnicode.client import BackendClient, backend_info_path, read_backend_info
from omnicode.core.persistence import config as config_store
from omnicode.core.service import create_app
from omnicode.models import ServerConfig


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
    chat = (await client._send("PATCH", f"/sessions/{sid}", {"mode": "plan"}))["chat"]
    assert chat["mode"] == "plan"
    # settings round-trip
    assert "theme" in await client.get_settings()
    assert (await client.patch_settings({"theme": "custom"}))["theme"] == "custom"
    # a project over the wire
    await client._send("POST", "/projects", {"name": "P", "working_dir": str(tmp_path)})
    assert any(p["name"] == "P" for p in await client.list_resource("projects"))


def test_backend_info_path_uses_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert backend_info_path() == tmp_path / "omnicode" / "backend.json"
    assert read_backend_info() is None  # absent → None, not a crash


# --- Ctrl+Q kills the backend (stale-backend prevention) -------------------

def _write_backend_info(tmp_path, pid: int, port: int = 64479):
    import json

    path = tmp_path / "omnicode" / "backend.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "port": port, "token": "t", "version": "0.1.0"}))


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_discover_client_carries_pid_and_port(tmp_path, monkeypatch):
    """The client must remember WHICH backend it connected to so Ctrl+Q can kill exactly
    that one — a stale backend outliving code changes is the recurring 'fix didn't work'
    trap."""
    import importlib
    import os

    import omnicode.client as client_mod

    # conftest's autouse fixture stubs discover/connect to the in-process backend; this
    # test needs the REAL discovery path — reload the module to restore it, then re-stub
    # only the health probe.
    real_discover = importlib.reload(client_mod).discover
    monkeypatch.setattr(client_mod, "discover", real_discover)

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    port = _free_port()
    _write_backend_info(tmp_path, os.getpid(), port=port)
    # Nothing listens on that port → no client at all with the real health probe.
    assert await real_discover() is None
    # Fake a healthy hit → pid/port are carried from the discovery file.
    async def _always_healthy(info, timeout=2.0):
        return True

    monkeypatch.setattr(client_mod, "_healthy", _always_healthy)
    client = await real_discover()
    assert client is not None
    assert client.pid == os.getpid() and client.port == port


async def test_terminate_backend_kills_the_named_process(tmp_path, monkeypatch):
    """SIGTERM the backend we connected to; wait until it's gone. Uses a real sleeping
    child process so the signal/wait logic runs for real. Liveness is probed via the
    health endpoint (a SIGTERMed child lingers as a zombie, so a pid probe would lie)."""
    import subprocess
    import sys

    from omnicode.client import BackendClient

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _write_backend_info(tmp_path, proc.pid, port=port)
    client = BackendClient(f"http://127.0.0.1:{port}", "t")
    client.pid, client.port = proc.pid, port
    assert await client.terminate_backend(timeout=5.0) is True
    assert proc.wait(timeout=5) is not None  # actually dead


async def test_terminate_backend_refuses_a_replaced_backend(tmp_path, monkeypatch):
    """If backend.json now names a DIFFERENT pid+port than the one we connected to, a newer
    backend replaced ours — killing it would orphan the new TUI. Refuse."""
    import subprocess
    import sys

    from omnicode.client import BackendClient

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    our_port, new_port = _free_port(), _free_port()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _write_backend_info(tmp_path, proc.pid, port=new_port)
    client = BackendClient(f"http://127.0.0.1:{our_port}", "t")
    client.pid, client.port = proc.pid, our_port  # connected to the OLD port
    assert await client.terminate_backend(timeout=0.5) is False
    assert proc.poll() is None  # untouched
    proc.terminate()
    # And never signal anything when we don't know a pid at all.
    unknown = BackendClient(f"http://127.0.0.1:{our_port}", "t")
    assert await unknown.terminate_backend(timeout=0.1) is False
