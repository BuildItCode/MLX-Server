"""Service-level tests for the HTTP+SSE backend (mlx_launcher.core.service).

Drives the ASGI app in-process via httpx with a fake engine injected through ``engine_factory`` —
no real model server. Covers auth, session creation, a streaming run (SSE content + finish +
persistence), and the permission round-trip (request over SSE → POST decision → denied tool)."""

import asyncio
import json

import httpx

from mlx_launcher.core.persistence import chats as chats_store
from mlx_launcher.core.persistence import config as config_store
from mlx_launcher.core.service import create_app
from mlx_launcher.models import ServerConfig


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _native_call(name, args, call_id="c1"):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        "finish_reason": "tool_calls"}]}


def _final(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class FakeEngine:
    def __init__(self, chat_responses=None, stream_script=None):
        self.chat_responses = list(chat_responses or [])
        self.stream_script = list(stream_script or [])

    async def chat(self, messages, tools=None, *, read_timeout=600.0):
        return self.chat_responses.pop(0)

    async def stream_chat(self, messages, *, cancel=None):
        for item in self.stream_script:
            yield item


async def _seed_server(server_id="s1"):
    await config_store.mutate(lambda f: config_store.upsert_server(
        f, ServerConfig(id=server_id, name="srv", model="llama", engine="llama-cpp",
                        host="127.0.0.1", port=8080)))


async def _consume_sse(client, url, collected, pid_holder, stop_at=("finish", "error")):
    async with client.stream("GET", url, timeout=10.0) as resp:
        assert resp.status_code == 200
        ev_name = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
                collected.append((ev_name, data))
                if ev_name == "permission_request":
                    pid_holder["id"] = json.loads(data)["id"]
                if ev_name in stop_at:
                    break
                ev_name = None


async def test_healthz_needs_no_auth_but_others_do(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    app = create_app(token="secret")
    async with _client(app) as c:
        assert (await c.get("/healthz")).status_code == 200
        assert (await c.get("/sessions")).status_code == 401  # missing token
        ok = await c.get("/sessions", headers={"authorization": "Bearer secret"})
        assert ok.status_code == 200


async def test_streaming_run_emits_events_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    eng = FakeEngine(stream_script=[("reason", "mm "), ("content", "hello "), ("content", "world"),
                                    ("finish", "stop")])
    app = create_app(engine_factory=lambda s: eng)
    async with _client(app) as c:
        sid = (await c.post("/sessions", json={"server_id": "s1"})).json()["session_id"]
        rid = (await c.post(f"/sessions/{sid}/runs", json={"text": "hi"})).json()["run_id"]
        collected, pid = [], {}
        await _consume_sse(c, f"/sessions/{sid}/runs/{rid}/events", collected, pid)
    types = [t for t, _ in collected]
    assert "content" in types and "finish" in types
    content = "".join(json.loads(d)["text"] for t, d in collected if t == "content")
    assert content == "hello world"
    chat = chats_store.get_chat(chats_store.load(), sid)
    assert chat.messages[-1].role == "assistant" and chat.messages[-1].text == "hello world"
    assert chat.messages[0].role == "user" and chat.messages[0].text == "hi"


async def test_permission_request_round_trip_deny(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    # a project whose working dir exists → the sandboxed fs tools (incl. mutating write_file) are offered
    work = tmp_path / "proj"
    work.mkdir()
    from mlx_launcher.models import Project
    proj = Project(name="p", working_dir=str(work))
    await chats_store.mutate(lambda f: chats_store.upsert_project(f, proj))

    from mlx_launcher.core import events as ev
    eng = FakeEngine(chat_responses=[_native_call("write_file", {"path": "a.py", "content": "x"}),
                                     _final("ok, I asked first")])
    app = create_app(engine_factory=lambda s: eng)
    async with _client(app) as ctrl:
        sid = (await ctrl.post("/sessions", json={"server_id": "s1", "project_id": proj.id})).json()["session_id"]
        rid = (await ctrl.post(f"/sessions/{sid}/runs", json={"text": "make a.py"})).json()["run_id"]
        handle = app.state.runs[rid]
        # Drain the run's event queue directly (SSE serialization is covered by the streaming test);
        # answer the permission through the real HTTP endpoint when it's requested.
        collected = []
        while True:
            item = await asyncio.wait_for(handle.queue.get(), timeout=10)
            if item is None:
                break
            collected.append(item)
            if isinstance(item, ev.PermissionRequest):
                r = await ctrl.post(f"/sessions/{sid}/runs/{rid}/permissions/{item.id}",
                                    json={"decision": "deny"})
                assert r.status_code == 202
    assert any(isinstance(e, ev.PermissionRequest) for e in collected), collected
    assert any(isinstance(e, ev.ToolFinished) and e.status == "denied" for e in collected)
    assert any(isinstance(e, ev.TurnFinished) for e in collected)


async def test_cancel_during_permission_prompt_unblocks_the_run(tmp_path, monkeypatch):
    """Cancelling while a permission prompt is outstanding must resolve it as deny so the loop
    doesn't hang on the permission future (the queue draining to the None sentinel within the
    timeout proves it finished; a TimeoutError here would mean the regression is back)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    work = tmp_path / "proj"
    work.mkdir()
    from mlx_launcher.core import events as ev
    from mlx_launcher.models import Project
    proj = Project(name="p", working_dir=str(work))
    await chats_store.mutate(lambda f: chats_store.upsert_project(f, proj))

    eng = FakeEngine(chat_responses=[_native_call("write_file", {"path": "a.py", "content": "x"}),
                                     _final("unused — cancelled before this")])
    app = create_app(engine_factory=lambda s: eng)
    async with _client(app) as ctrl:
        sid = (await ctrl.post("/sessions", json={"server_id": "s1", "project_id": proj.id})).json()["session_id"]
        rid = (await ctrl.post(f"/sessions/{sid}/runs", json={"text": "make a.py"})).json()["run_id"]
        handle = app.state.runs[rid]
        collected = []
        while True:
            item = await asyncio.wait_for(handle.queue.get(), timeout=10)
            if item is None:
                break
            collected.append(item)
            if isinstance(item, ev.PermissionRequest):
                # Cancel INSTEAD of answering — the fail-safe must deny the pending prompt.
                assert (await ctrl.post(f"/sessions/{sid}/runs/{rid}/cancel")).status_code == 202
    assert any(isinstance(e, ev.PermissionRequest) for e in collected), collected
    assert any(isinstance(e, ev.ToolFinished) and e.status == "denied" for e in collected)
    fin = [e for e in collected if isinstance(e, ev.TurnFinished)]
    assert fin and fin[-1].reason == "cancelled"
    assert not (work / "a.py").exists()  # the denied write never touched disk


async def test_auto_mode_runs_mutating_tools_without_prompting(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    work = tmp_path / "proj"
    work.mkdir()
    from mlx_launcher.core import events as ev
    from mlx_launcher.models import Project
    proj = Project(name="p", working_dir=str(work))
    await chats_store.mutate(lambda f: chats_store.upsert_project(f, proj))

    eng = FakeEngine(chat_responses=[_native_call("write_file", {"path": "a.txt", "content": "hi"}),
                                     _final("done")])
    app = create_app(engine_factory=lambda s: eng)
    async with _client(app) as c:
        sid = (await c.post("/sessions", json={"server_id": "s1", "project_id": proj.id})).json()["session_id"]
        await c.patch(f"/sessions/{sid}", json={"mode": "auto"})  # auto → no permission prompt
        rid = (await c.post(f"/sessions/{sid}/runs", json={"text": "make a.txt"})).json()["run_id"]
        handle = app.state.runs[rid]
        collected = []
        while True:
            item = await asyncio.wait_for(handle.queue.get(), timeout=10)
            if item is None:
                break
            collected.append(item)
    assert not any(isinstance(e, ev.PermissionRequest) for e in collected)  # auto-approved, no prompt
    assert any(isinstance(e, ev.ToolFinished) and e.status == "ok" for e in collected)
    assert (work / "a.txt").read_text() == "hi"  # the tool actually ran server-side


async def test_server_lifecycle_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    app = create_app()
    async with _client(app) as c:
        # nothing started yet
        assert (await c.get("/servers/status")).json() == {"servers": []}
        assert (await c.get("/servers/s1/status")).json()["is_running"] is False
        assert (await c.post("/servers/nope/start")).status_code == 404  # unknown profile
        # the llama-cpp binary isn't on PATH in the test env → BinaryNotFound → 409 (not a crash)
        r = await c.post("/servers/s1/start")
        assert r.status_code == 409 and "not found" in r.json()["error"].lower()
        assert (await c.post("/servers/s1/stop")).status_code == 202  # stop is a no-op, never errors


async def test_compaction_run_replaces_history(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    await _seed_server()
    eng = FakeEngine(stream_script=[("content", "Summary: did X and Y."), ("finish", "stop")])
    app = create_app(engine_factory=lambda s: eng)
    async with _client(app) as c:
        sid = (await c.post("/sessions", json={"server_id": "s1"})).json()["session_id"]
        # seed a couple of turns, then compact
        await c.post(f"/sessions/{sid}/runs", json={"text": "hello"})  # 1 real turn (engine streams)
        rid = (await c.post(f"/sessions/{sid}/runs", json={"kind": "compact"})).json()["run_id"]
        handle = app.state.runs[rid]
        while await asyncio.wait_for(handle.queue.get(), timeout=10) is not None:
            pass
    chat = chats_store.get_chat(chats_store.load(), sid)
    # history replaced by the compaction marker + the summary
    assert chat.messages[0].role == "user" and "compacted" in chat.messages[0].text.lower()
    assert chat.messages[-1].role == "assistant" and "Summary" in chat.messages[-1].text
