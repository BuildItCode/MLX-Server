"""Connect to MCP servers (stdio or SSE), list their tools as OpenAI tool specs,
and route tool calls to them. Sessions are opened per turn via an AsyncExitStack."""

from __future__ import annotations

import asyncio
import re
import shlex
from contextlib import AsyncExitStack
from typing import Callable, Optional

_CONNECT_TIMEOUT = 20.0  # per-server cap on connect+initialize+list_tools, so one
# unresponsive MCP server can't hang the whole chat turn


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "srv"


def _parse_env(text: str) -> Optional[dict]:
    env = {}
    for part in (text or "").split():
        if "=" in part:
            key, value = part.split("=", 1)
            env[key] = value
    return env or None


async def open_sessions(
    stack: AsyncExitStack,
    servers: list,
    on_error: Optional[Callable[[str, str], None]] = None,
) -> tuple[dict, list, dict]:
    """Open sessions to all enabled servers. Returns (sessions, tool_specs, router),
    where router maps the OpenAI tool name -> (server_name, real_tool_name)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    sessions: dict = {}
    specs: list = []
    router: dict = {}

    async def _connect(srv):
        """Open a session and list its tools — bounded so one bad server can't hang."""
        if srv.transport == "sse":
            read, write = await stack.enter_async_context(sse_client(srv.url))
        else:
            params = StdioServerParameters(
                command=srv.command,
                args=shlex.split(srv.args or ""),
                env=_parse_env(srv.env),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session, await session.list_tools()

    for srv in servers:
        if not getattr(srv, "enabled", True):
            continue
        try:
            session, listed = await asyncio.wait_for(_connect(srv), timeout=_CONNECT_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — bad/unreachable server (timeout included)
            if on_error:
                reason = (f"timed out after {int(_CONNECT_TIMEOUT)}s"
                          if isinstance(exc, asyncio.TimeoutError) else str(exc))
                on_error(srv.name, reason)
            continue
        sessions[srv.name] = session
        for tool in listed.tools:
            base = f"mcp__{slug(srv.name)}__{tool.name}"[:64]
            fq, n = base, 1
            while fq in router:  # truncation/name collision → disambiguate within 64 chars
                tag = f"_{n}"
                fq = base[: 64 - len(tag)] + tag
                n += 1
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": fq,
                        "description": (tool.description or tool.name)[:1024],
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
            router[fq] = (srv.name, tool.name)
    return sessions, specs, router


async def call_mcp(sessions: dict, router: dict, fq_name: str, arguments: dict) -> str:
    # Return an error string rather than raising (matches the web_search / fs-tool
    # contract) so a hallucinated name or a server that failed to open feeds the model a
    # result instead of aborting the agent loop.
    if fq_name not in router:
        return f"unknown MCP tool: {fq_name}"
    server_name, real_name = router[fq_name]
    session = sessions.get(server_name)
    if session is None:
        return f"MCP server not connected: {server_name}"
    try:
        result = await session.call_tool(real_name, arguments or {})
    except Exception as exc:  # noqa: BLE001
        return f"ERROR calling {fq_name}: {exc}"
    parts = []
    for chunk in result.content or []:
        if getattr(chunk, "type", None) == "text":
            parts.append(chunk.text)
        else:
            parts.append(f"[{getattr(chunk, 'type', 'content')} content]")
    text = "\n".join(parts) or "(no output)"
    if getattr(result, "isError", False):
        text = "ERROR: " + text
    return text
