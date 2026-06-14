"""Connect to MCP servers (stdio or SSE), list their tools as OpenAI tool specs,
and route tool calls to them. Sessions are opened per turn via an AsyncExitStack."""

from __future__ import annotations

import re
import shlex
from contextlib import AsyncExitStack
from typing import Callable, Optional


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
    for srv in servers:
        if not getattr(srv, "enabled", True):
            continue
        try:
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
            listed = await session.list_tools()
            sessions[srv.name] = session
            for tool in listed.tools:
                fq = f"mcp__{slug(srv.name)}__{tool.name}"[:64]
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
        except Exception as exc:  # noqa: BLE001
            if on_error:
                on_error(srv.name, str(exc))
    return sessions, specs, router


async def call_mcp(sessions: dict, router: dict, fq_name: str, arguments: dict) -> str:
    server_name, real_name = router[fq_name]
    session = sessions[server_name]
    result = await session.call_tool(real_name, arguments or {})
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
