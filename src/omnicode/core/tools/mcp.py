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


def _clean_arguments(args: dict) -> dict:
    """Drop arguments explicitly set to null before the call hits the MCP server.

    The deepagents wrapper (core/deepagents/tools.py) already strips Nones, but
    call_mcp is also reached from other paths (chat/mcp_client, ACP) — and a model
    can still emit an explicit null for an optional param. MCP servers like
    @playwright/mcp type optional params as plain "string" in their schema, so an
    explicit JSON null is a schema violation ("expected string, received null").
    In JSON Schema an optional argument means OMITTED, never null.
    """
    return {k: v for k, v in args.items() if v is not None}


def _missing_file_url(args: dict) -> str | None:
    """The url arg of an MCP call, if it's a file:// URL pointing at a nonexistent path."""
    import os
    from urllib.parse import unquote, urlparse

    url = args.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    path = unquote(urlparse(url).path or "")
    return None if os.path.exists(path) else url


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
    arguments = _clean_arguments(arguments or {})
    # A hand-built file:// URL pointing at a nonexistent path (e.g. the classic fused
    # '/Users/nick/DesktopTest/…' for '/Users/nick/Desktop/Test/…') would navigate the
    # browser to a broken page — turn the slip into a self-correcting error instead.
    missing = _missing_file_url(arguments)
    if missing is not None:
        return (
            f"ERROR: {missing} does not exist. Check the path for typos (fused "
            "segments like 'DesktopTest' instead of 'Desktop/Test' are a common slip). "
            "To preview a file from the working directory, call open_in_browser with a "
            "RELATIVE path instead of building a file:// URL yourself."
        )
    try:
        result = await session.call_tool(real_name, arguments)
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
