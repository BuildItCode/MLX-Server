"""The local HTTP + SSE backend service — the wire boundary between the backend and any frontend.

REST for control (create/list/open sessions, send a prompt, cancel, answer a permission prompt)
and a per-run Server-Sent-Events stream for everything the agent loop emits (content, reason, tool
events, permission requests, finish). A *session* is 1:1 with a persisted Chat.

Security posture (local-first): bind ``127.0.0.1`` only, require a bearer token (minted at startup,
see :mod:`mlx_launcher.core.server_main`) on every route except ``/healthz``, and treat a dropped
SSE connection as an implicit cancel so a runaway loop can't keep burning the model. The token is
never logged; model-server output forwarded over the wire is treated as untrusted text.

This module is engine- and frontend-agnostic: ``app.state.engine_factory`` builds the engine for a
session (overridable in tests), and every frontend speaks the same JSON over the wire."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

from .. import __version__
from ..models import Attachment, Chat, ChatMessage, McpServer, Project, ServerConfig, Subagent
from . import compaction
from . import events as ev
from . import skills
from .agent import AgentRunner, RunPolicy
from .persistence import chats as chats_store
from .persistence import config as config_store
from .server import discovery
from .server.manager import BinaryNotFound, PortInUse, ServerManager, ServerStatus
from .session import Session
from .tools.phrasing import _perm_prompt

PROTOCOL_VERSION = 1


class RunHandle:
    """In-flight state for one run: its event queue (drained by the SSE endpoint), a cancel flag
    polled by the loop, and the futures awaiting permission decisions."""

    def __init__(self, run_id: str, session_id: str, kind: str = "chat") -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.kind = kind  # "chat" | "compact"
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancel = asyncio.Event()
        self.perms: dict[str, asyncio.Future] = {}
        self.done = False
        self.task: Optional[asyncio.Task] = None  # strong ref so the run task isn't GC'd mid-flight


def _deny_pending_perms(handle: RunHandle) -> None:
    """Resolve any outstanding permission prompts as ``deny`` — the fail-safe for when a run is
    cancelled or its SSE stream drops while a prompt is in flight, so the loop (blocked on the
    permission future) never hangs and its tool/MCP resources are released."""
    for fut in list(handle.perms.values()):
        if not fut.done():
            fut.set_result("deny")


# --- auth ----------------------------------------------------------------

def _auth_error(request: Request) -> Optional[Response]:
    token = request.app.state.token
    if not token:
        return None  # auth disabled (tests)
    if request.headers.get("authorization", "") != f"Bearer {token}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


# --- handshake -----------------------------------------------------------

async def healthz(request: Request) -> Response:
    """Liveness + identity (no auth) so a client can confirm it found the right backend."""
    return JSONResponse({"status": "ok", "version": __version__,
                         "protocol": PROTOCOL_VERSION, "pid": os.getpid()})


# --- sessions ------------------------------------------------------------

def _chat_summary(chat: Chat) -> dict:
    return {"id": chat.id, "title": chat.title, "model": chat.model,
            "project_id": chat.project_id, "updated": chat.updated}


async def create_session(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    body = await request.json()
    from .persistence import config as config_store
    chat = Chat(project_id=body.get("project_id"))
    cfg = config_store.find_server_by_id(body["server_id"]) if body.get("server_id") else None
    if cfg is not None:
        from ..engine import capabilities
        chat.server_id = cfg.id
        chat.base_url = cfg.base_url()
        chat.model = cfg.model
        chat.reasoning = capabilities.supports_reasoning(cfg.model)
    await chats_store.mutate(lambda f: chats_store.upsert_chat(f, chat))
    return JSONResponse({"session_id": chat.id, "chat": chat.model_dump()})


async def list_sessions(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    data = chats_store.load()
    project_id = request.query_params.get("project_id")
    chats = chats_store.chats_in(data, project_id if project_id else None)
    return JSONResponse({"chats": [_chat_summary(c) for c in chats]})


async def get_session(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    chat = chats_store.get_chat(chats_store.load(), request.path_params["session_id"])
    if chat is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    return JSONResponse({"chat": chat.model_dump()})


async def delete_session(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    sid = request.path_params["session_id"]
    await chats_store.mutate(lambda f: chats_store.delete_chat(f, sid))
    return Response(status_code=204)


# --- runs ----------------------------------------------------------------

def _append_user(file, chat_id: str, msg: ChatMessage) -> None:
    chat = chats_store.get_chat(file, chat_id)
    if chat is not None:
        if chat.title == "New chat" and msg.text:  # title a fresh chat from its first message
            chat.title = msg.text[:40]
        chat.messages.append(msg)
        chat.updated = msg.ts


def _append_assistant(file, chat_id: str, fin: ev.TurnFinished) -> None:
    chat = chats_store.get_chat(file, chat_id)
    if chat is not None:
        m = ChatMessage(role="assistant", text=fin.text, reasoning=fin.reasoning,
                        n_tokens=fin.n_tool_calls or None, elapsed=fin.elapsed or None)
        chat.messages.append(m)
        chat.updated = m.ts


def _replace_history(file, chat_id: str, turns: list) -> None:
    chat = chats_store.get_chat(file, chat_id)
    if chat is not None:
        chat.messages = list(turns)


async def create_run(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    sid = request.path_params["session_id"]
    chat = chats_store.get_chat(chats_store.load(), sid)
    if chat is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    body = await request.json()
    kind = body.get("kind", "chat")
    if kind != "compact":  # a normal turn: append the user message (compaction summarizes history)
        atts = [Attachment(**a) for a in (body.get("attachments") or [])]
        user_msg = ChatMessage(role="user", text=body.get("text", ""), attachments=atts)
        await chats_store.mutate(lambda f: _append_user(f, sid, user_msg))
        chat = chats_store.get_chat(chats_store.load(), sid)  # reload with the new turn

    run_id = uuid.uuid4().hex
    handle = RunHandle(run_id, sid, kind=kind)
    request.app.state.runs[run_id] = handle
    driver = _drive_compaction if kind == "compact" else _drive_run
    # Keep a strong reference: a bare create_task() can be garbage-collected mid-run.
    handle.task = asyncio.create_task(driver(request.app, handle, chat))
    return JSONResponse({"run_id": run_id})


async def _drive_compaction(app: Starlette, handle: RunHandle, chat: Chat) -> None:
    """Summarize the conversation and REPLACE its history with the summary (a user→assistant pair),
    streaming the summary as content events. Manual /compact or the >95% auto-trigger."""
    q = handle.queue
    try:
        session = Session.resolve(chat)
        engine = app.state.engine_factory(session)
        # summarize, not plan/review: drop plan/coding framing for the summary request
        summary_chat = chat.model_copy(update={"mode": "build", "coding": False})
        ssn = Session(chat=summary_chat, server=session.server, project=session.project,
                      skill_instructions=session.skill_instructions, mcp_servers=session.mcp_servers)
        parts: list[str] = []
        async for k, chunk in engine.stream_chat(compaction.summary_request(ssn.messages()),
                                                  cancel=handle.cancel.is_set):
            if k == "content":
                parts.append(chunk)
                await q.put(ev.ContentDelta(chunk))
        text = "".join(parts).strip()
        if text and not handle.cancel.is_set():
            turns = [ChatMessage(role=t["role"], text=t["content"]) for t in compaction.replacement_turns(text)]
            await chats_store.mutate(lambda f: _replace_history(f, handle.session_id, turns))
        await q.put(ev.TurnFinished(text, reason="cancelled" if handle.cancel.is_set() else "stop"))
    except Exception as exc:  # noqa: BLE001
        await q.put(ev.TurnFailed(str(exc)))
    finally:
        handle.done = True
        await q.put(None)


async def _drive_run(app: Starlette, handle: RunHandle, chat: Chat) -> None:
    """Run the agent loop for one turn, forwarding its events onto the run's queue and persisting
    the assistant turn at the end. Permission + open-url become SSE events answered out-of-band."""
    q = handle.queue
    try:
        session = Session.resolve(chat)
        engine = app.state.engine_factory(session)

        async def permission(name: str, args: dict) -> str:
            if chat.mode == "auto":  # auto mode runs mutating tools without asking
                return "all"
            if handle.cancel.is_set():  # already cancelling — don't surface a prompt no one can answer
                return "deny"
            pid = uuid.uuid4().hex
            fut = asyncio.get_running_loop().create_future()
            handle.perms[pid] = fut
            summary, detail = _perm_prompt(name, args)
            await q.put(ev.PermissionRequest(pid, name, summary, detail))
            try:
                return await fut
            finally:
                handle.perms.pop(pid, None)

        async def open_url(url: str) -> bool:
            await q.put(ev.OpenUrlRequested(url))
            return True

        async with AsyncExitStack() as stack:
            tools = await session.build_toolset(stack, open_url=open_url)
            root = session.fs_root()
            policy = RunPolicy(
                max_iters=24 if root else 8,
                max_tool_calls=None if root else 8,
                native_tools=(chat.server_id or "") not in app.state.prompted_servers,
            )
            runner = AgentRunner(engine, tools=tools, policy=policy, permission=permission,
                                 system_note=session.system_note(), cancel=handle.cancel.is_set)
            final: Optional[ev.TurnFinished] = None
            async for event in runner.run(session.messages()):
                await q.put(event)
                if isinstance(event, ev.TurnFinished):
                    final = event
            if runner.used_prompted:
                app.state.prompted_servers.add(chat.server_id or "")
            if final is not None and final.reason != "cancelled":
                await chats_store.mutate(lambda f: _append_assistant(f, handle.session_id, final))
    except Exception as exc:  # noqa: BLE001
        await q.put(ev.TurnFailed(str(exc)))
    finally:
        handle.done = True
        await q.put(None)  # sentinel → close the SSE stream


async def run_events(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    handle: Optional[RunHandle] = request.app.state.runs.get(request.path_params["run_id"])
    if handle is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)

    async def gen():
        try:
            while True:
                item = await handle.queue.get()
                if item is None:
                    break
                yield {"event": item.type, "data": json.dumps(ev.to_dict(item))}
        finally:
            # A dropped connection (or normal close) cancels the run as a fail-safe so a runaway
            # loop can't keep generating with no one listening — and any outstanding permission
            # prompt resolves to deny so the loop doesn't hang waiting for an answer that can't come.
            handle.cancel.set()
            _deny_pending_perms(handle)
            request.app.state.runs.pop(handle.run_id, None)

    return EventSourceResponse(gen())


async def cancel_run(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    handle = request.app.state.runs.get(request.path_params["run_id"])
    if handle is not None:
        handle.cancel.set()
        _deny_pending_perms(handle)  # unblock the loop if it's waiting on a permission answer
    return Response(status_code=202)


async def answer_permission(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    handle = request.app.state.runs.get(request.path_params["run_id"])
    if handle is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    body = await request.json()
    fut = handle.perms.get(request.path_params["permission_id"])
    if fut is not None and not fut.done():
        fut.set_result(body.get("decision", "deny"))
    return Response(status_code=202)


# --- session settings ----------------------------------------------------

_SESSION_STR_FIELDS = ("project_id", "skill_id", "reasoning_effort", "mode", "title")
_SESSION_BOOL_FIELDS = ("reasoning", "web_search", "tools", "coding")


async def patch_session(request: Request) -> Response:
    """Update chat-scoped settings (mode/effort/reasoning/web/tools/coding/skill/server/title)."""
    if (e := _auth_error(request)):
        return e
    sid = request.path_params["session_id"]
    body = await request.json()

    def apply(f):
        chat = chats_store.get_chat(f, sid)
        if chat is None:
            return
        for k in _SESSION_STR_FIELDS:
            if k in body:
                setattr(chat, k, body[k])
        for k in _SESSION_BOOL_FIELDS:
            if k in body:
                setattr(chat, k, bool(body[k]))
        if "server_id" in body:  # repoint the model + base_url from the new profile
            chat.server_id = body["server_id"]
            cfg = config_store.find_server_by_id(body["server_id"]) if body["server_id"] else None
            if cfg is not None:
                chat.base_url = cfg.base_url()
                chat.model = cfg.model

    out = await chats_store.mutate(apply)
    chat = chats_store.get_chat(out, sid)
    if chat is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    return JSONResponse({"chat": chat.model_dump()})


# --- resource CRUD --------------------------------------------------------

async def list_servers(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    return JSONResponse({"servers": [s.model_dump() for s in config_store.load().servers]})


async def upsert_server(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    body = await request.json()
    if "server_id" in request.path_params:
        body = {**body, "id": request.path_params["server_id"]}
    srv = ServerConfig.model_validate(body)
    await config_store.mutate(lambda f: config_store.upsert_server(f, srv))
    return JSONResponse({"server": srv.model_dump()})


async def delete_server(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    await config_store.mutate(lambda f: config_store.delete_server(f, request.path_params["server_id"]))
    return Response(status_code=204)


def _list_chat_resource(attr: str):
    async def handler(request: Request) -> Response:
        if (e := _auth_error(request)):
            return e
        items = getattr(chats_store.load(), attr)
        return JSONResponse({attr: [i.model_dump() for i in items]})
    return handler


def _upsert_chat_resource(model, upsert, id_param: str):
    async def handler(request: Request) -> Response:
        if (e := _auth_error(request)):
            return e
        body = await request.json()
        if id_param in request.path_params:
            body = {**body, "id": request.path_params[id_param]}
        obj = model.model_validate(body)
        await chats_store.mutate(lambda f: upsert(f, obj))
        return JSONResponse(obj.model_dump())
    return handler


def _delete_chat_resource(delete, id_param: str):
    async def handler(request: Request) -> Response:
        if (e := _auth_error(request)):
            return e
        await chats_store.mutate(lambda f: delete(f, request.path_params[id_param]))
        return Response(status_code=204)
    return handler


async def get_settings(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    return JSONResponse(config_store.load().settings.model_dump())


async def patch_settings(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    body = await request.json()

    def apply(f):
        for k, v in body.items():
            if hasattr(f.settings, k):
                setattr(f.settings, k, v)

    out = await config_store.mutate(apply)
    return JSONResponse(out.settings.model_dump())


async def list_skills(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    out = [{"id": getattr(s, "id", ""), "name": getattr(s, "name", ""),
            "description": getattr(s, "description", ""), "origin": getattr(s, "origin", "")}
           for s in skills.all_skills()]
    return JSONResponse({"skills": out})


async def session_context(request: Request) -> Response:
    """Estimated context-window usage for the chat (for the usage bar)."""
    if (e := _auth_error(request)):
        return e
    chat = chats_store.get_chat(chats_store.load(), request.path_params["session_id"])
    if chat is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    usage = Session.resolve(chat).context_usage()
    return JSONResponse({"used": usage[0], "window": usage[1]} if usage else {})


# --- model-server lifecycle (the backend owns the subprocesses) -----------

def _status_snapshot(server_id: str, mgr: Optional[ServerManager]) -> dict:
    return {
        "id": server_id,
        "status": mgr.status.value if mgr else ServerStatus.IDLE.value,
        "message": mgr.status_message if mgr else "",
        "is_running": bool(mgr and mgr.is_running),
        "host": mgr.cfg.host if mgr else None,
        "port": mgr.cfg.port if mgr else None,
        "base_url": mgr.cfg.base_url() if mgr else None,  # the ACTUAL address (may be bumped)
        "pid": (mgr.proc.pid if mgr and mgr.proc else None),
    }


async def start_server(request: Request) -> Response:
    """Start a model server. ``{"bump": true}`` runs it ALONGSIDE others (a free port is chosen if
    the profile's is taken — for the subagent side-pane); otherwise the profile's port is freed by
    stopping any other server bound to it. Returns the status snapshot incl. the actual base_url."""
    if (e := _auth_error(request)):
        return e
    sid = request.path_params["server_id"]
    cfg = config_store.find_server_by_id(sid)
    if cfg is None:
        return JSONResponse({"error": "unknown server"}, status_code=404)
    body = await request.json() if await request.body() else {}
    managers = request.app.state.managers
    mgr = managers.get(sid)
    if mgr is not None and mgr.is_running:
        return JSONResponse(_status_snapshot(sid, mgr))  # already up (reuse)

    reserved = {(m.cfg.host, m.cfg.port) for m in managers.values() if m.is_running}
    if body.get("bump"):  # run side-by-side: pick a free port if the profile's collides
        if (cfg.host, cfg.port) in reserved or not discovery.is_port_free(cfg.host, cfg.port):
            free = next((p for p in range(cfg.port + 1, cfg.port + 64)
                         if (cfg.host, p) not in reserved and discovery.is_port_free(cfg.host, p)), None)
            if free is None:
                return JSONResponse({"error": "no free port for the subagent server"}, status_code=409)
            cfg = cfg.model_copy(update={"port": free})
    else:  # evict any other server on this host:port
        for other_id, m in list(managers.items()):
            if other_id != sid and m.is_running and (m.cfg.host, m.cfg.port) == (cfg.host, cfg.port):
                await m.stop()
    mlx_override = config_store.load().settings.mlx_server_path or None
    mgr = ServerManager(cfg, mlx_override=mlx_override)
    managers[sid] = mgr
    try:
        await mgr.start()
    except (BinaryNotFound, PortInUse) as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse(_status_snapshot(sid, mgr))


async def stop_server(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    mgr = request.app.state.managers.get(request.path_params["server_id"])
    if mgr is not None:
        await mgr.stop()
    return Response(status_code=202)


async def restart_server(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    mgr = request.app.state.managers.get(request.path_params["server_id"])
    if mgr is not None:
        await mgr.stop()
    return await start_server(request)


async def server_status(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    sid = request.path_params["server_id"]
    return JSONResponse(_status_snapshot(sid, request.app.state.managers.get(sid)))


async def all_server_status(request: Request) -> Response:
    if (e := _auth_error(request)):
        return e
    managers = request.app.state.managers
    return JSONResponse({"servers": [_status_snapshot(sid, m) for sid, m in managers.items()]})


async def server_models(request: Request) -> Response:
    """Probe a server profile's ``/v1/models`` (used by the Xcode help screen to test the provider).
    Returns ``{"models": [...]}`` or ``{"error": ...}`` — never raises to the frontend."""
    if (e := _auth_error(request)):
        return e
    from ..engine.openai import fetch_models
    cfg = config_store.find_server_by_id(request.path_params["server_id"])
    if cfg is None:
        return JSONResponse({"error": "unknown server"}, status_code=404)
    try:
        return JSONResponse({"models": await fetch_models(cfg.base_url())})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)})


async def server_logs(request: Request) -> Response:
    """SSE stream of a model server's logs + status — replays the buffer, then streams live."""
    if (e := _auth_error(request)):
        return e
    mgr = request.app.state.managers.get(request.path_params["server_id"])
    if mgr is None:
        return JSONResponse({"error": "server not started"}, status_code=404)
    queue: asyncio.Queue = asyncio.Queue()
    for stream, line in list(mgr.log_buffer):  # replay first
        queue.put_nowait(("log", {"stream": stream, "line": line}))
    queue.put_nowait(("status", {"status": mgr.status.value, "message": mgr.status_message}))
    token = mgr.subscribe(
        lambda stream, line: queue.put_nowait(("log", {"stream": stream, "line": line})),
        lambda status, message: queue.put_nowait(("status", {"status": status.value, "message": message})),
    )

    async def gen():
        try:
            while True:
                etype, data = await queue.get()
                yield {"event": etype, "data": json.dumps(data)}
        finally:
            mgr.unsubscribe(token)

    return EventSourceResponse(gen())


async def _stop_all_managers(app: Starlette) -> None:
    """On backend shutdown, don't orphan model-server subprocesses."""
    for mgr in list(getattr(app.state, "managers", {}).values()):
        try:
            await mgr.stop()
        except Exception:  # noqa: BLE001
            pass


# --- app -----------------------------------------------------------------

def create_app(*, token: Optional[str] = None, engine_factory=None, on_shutdown=None) -> Starlette:
    """Build the ASGI app. ``token`` is the bearer token required on every route except
    ``/healthz`` (None disables auth, for tests). ``engine_factory(session) -> Engine`` builds the
    engine for a run (defaults to the session's own OpenAI engine; tests inject a fake).
    ``on_shutdown()`` runs at the end of the lifespan AFTER model servers are stopped — used by
    ``server_main`` to remove the discovery file. It runs on SIGTERM/SIGINT (the lifespan shutdown
    fires) even though ``uvicorn.run()`` itself may not return, so it can't live in ``main()``."""
    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/sessions", create_session, methods=["POST"]),
        Route("/sessions", list_sessions, methods=["GET"]),
        Route("/sessions/{session_id}", get_session, methods=["GET"]),
        Route("/sessions/{session_id}", patch_session, methods=["PATCH"]),
        Route("/sessions/{session_id}", delete_session, methods=["DELETE"]),
        Route("/sessions/{session_id}/runs", create_run, methods=["POST"]),
        Route("/sessions/{session_id}/runs/{run_id}/events", run_events, methods=["GET"]),
        Route("/sessions/{session_id}/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/sessions/{session_id}/runs/{run_id}/permissions/{permission_id}",
              answer_permission, methods=["POST"]),
        # resource CRUD
        Route("/servers", list_servers, methods=["GET"]),
        Route("/servers", upsert_server, methods=["POST"]),
        Route("/servers/{server_id}", upsert_server, methods=["PUT"]),
        Route("/servers/{server_id}", delete_server, methods=["DELETE"]),
        Route("/projects", _list_chat_resource("projects"), methods=["GET"]),
        Route("/projects", _upsert_chat_resource(Project, chats_store.upsert_project, "project_id"), methods=["POST"]),
        Route("/projects/{project_id}", _upsert_chat_resource(Project, chats_store.upsert_project, "project_id"), methods=["PUT"]),
        Route("/projects/{project_id}", _delete_chat_resource(chats_store.delete_project, "project_id"), methods=["DELETE"]),
        Route("/mcp-servers", _list_chat_resource("mcp_servers"), methods=["GET"]),
        Route("/mcp-servers", _upsert_chat_resource(McpServer, chats_store.upsert_mcp, "mcp_id"), methods=["POST"]),
        Route("/mcp-servers/{mcp_id}", _upsert_chat_resource(McpServer, chats_store.upsert_mcp, "mcp_id"), methods=["PUT"]),
        Route("/mcp-servers/{mcp_id}", _delete_chat_resource(chats_store.delete_mcp, "mcp_id"), methods=["DELETE"]),
        Route("/subagents", _list_chat_resource("subagents"), methods=["GET"]),
        Route("/subagents", _upsert_chat_resource(Subagent, chats_store.upsert_subagent, "sub_id"), methods=["POST"]),
        Route("/subagents/{sub_id}", _upsert_chat_resource(Subagent, chats_store.upsert_subagent, "sub_id"), methods=["PUT"]),
        Route("/subagents/{sub_id}", _delete_chat_resource(chats_store.delete_subagent, "sub_id"), methods=["DELETE"]),
        Route("/skills", list_skills, methods=["GET"]),
        Route("/settings", get_settings, methods=["GET"]),
        Route("/settings", patch_settings, methods=["PATCH"]),
        Route("/sessions/{session_id}/context", session_context, methods=["GET"]),
        # model-server lifecycle (the backend owns the subprocesses)
        Route("/servers/{server_id}/start", start_server, methods=["POST"]),
        Route("/servers/{server_id}/stop", stop_server, methods=["POST"]),
        Route("/servers/{server_id}/restart", restart_server, methods=["POST"]),
        Route("/servers/{server_id}/status", server_status, methods=["GET"]),
        Route("/servers/{server_id}/models", server_models, methods=["GET"]),
        Route("/servers/status", all_server_status, methods=["GET"]),
        Route("/servers/{server_id}/logs", server_logs, methods=["GET"]),
    ]
    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        await _stop_all_managers(app)  # don't orphan model servers on backend shutdown
        if on_shutdown is not None:
            try:
                on_shutdown()
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.token = token
    app.state.runs = {}
    app.state.managers = {}  # server_id -> ServerManager (the backend owns model-server subprocesses)
    app.state.prompted_servers = set()
    app.state.engine_factory = engine_factory or (lambda s: s.engine())
    return app
