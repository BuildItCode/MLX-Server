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
_CALL_TIMEOUT = 120.0   # per-tool-call cap, so a hung MCP tool can't block the turn indefinitely
_MAX_RESULT_CHARS = 12_000  # cap the text fed back to the model — a large browser_snapshot or
# web-fetch HTML can be 50–200 KB, blowing the context window. ~3K tokens leaves headroom for the
# model's reply and the next tool call.


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "srv"


def _parse_env(text: str) -> Optional[dict]:
    # shlex.split handles quoted values with spaces (KEY="hello world"), which .split()
    # would break. Splits into tokens, then keeps only KEY=VALUE pairs.
    env = {}
    for part in shlex.split(text or ""):
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
                try:
                    on_error(srv.name, reason)
                except Exception:  # noqa: BLE001 — a faulty callback must not abort discovery
                    pass
            continue
        sessions[srv.name] = session
        is_playwright = "playwright" in slug(srv.name)
        for tool in listed.tools:
            if not is_playwright and _is_browser_tool(tool.name):
                # Browser/web actions are playwright-only: a non-playwright server that
                # ships its own browser_* tools never exposes them here.
                continue
            base = f"mcp__{slug(srv.name)}__{tool.name}"[:64]
            fq, n = base, 1
            while fq in router:  # truncation/name collision → disambiguate within 64 chars
                tag = f"_{n}"
                fq = base[: 64 - len(tag)] + tag
                n += 1
            description = tool.description or tool.name
            if is_playwright:
                description = _harden_playwright_description(tool.name, tool.inputSchema, description)
            # Cap description to 1024 chars, but when a note was appended by the hardener,
            # shrink the ORIGINAL description first so the note is always preserved.
            if len(description) > 1024:
                description = description[:1024]
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": fq,
                        "description": description,
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
            router[fq] = (srv.name, tool.name)
    return sessions, specs, router


# Tool names that act on a web page/browser. Non-playwright MCP servers never expose
# these — every web/browser action must go through the playwright server.
_BROWSER_TOOL_RE = re.compile(
    r"(?:^|_)(?:browser|navigate|click|scroll|snapshot|screenshot|type|press|hover|"
    r"select_option|drag|evaluate|tab|wait_for|go_back|go_forward|reload)(?:_|$)",
    re.IGNORECASE,
)


def _is_browser_tool(name: str) -> bool:
    """True when an MCP tool name is a browser/page-action tool (playwright-only zone)."""
    return bool(_BROWSER_TOOL_RE.search(name or ""))


# The current @playwright/mcp renames the element-reference argument from the
# long-standing `ref` to `target` (and startRef/endRef similarly). Models trained on
# the older API still emit `ref` — alias it to the canonical name rather than failing
# with "expected string, received undefined at target".
_REF_ALIASES = {"ref": "target", "startRef": "startTarget", "endRef": "endTarget"}
_TARGET_KEYS = ("target", "startTarget", "endTarget")

# Extra guidance appended to playwright tool descriptions before they go to the model.
_EVALUATE_NOTE = (
    " IMPORTANT: the function gets NO Playwright page object. For page-level JS use"
    " `() => document…`; for element JS pass `element` + `target` and use"
    " `(element) => element…`. Never reference `page` — it does not exist here."
)
_TARGET_NOTE = (
    " Pass ONLY the element reference id in the `target` argument (older docs call it"
    " `ref`) — e.g. target=\"e6\" copied from the snapshot's `[ref=e6]` marker. Never"
    " paste the whole snapshot line ('link \"Text\" [ref=e6]') and never invent"
    " attribute selectors like '[text=…]'; if you have no ref, use a real CSS selector"
    " or Playwright's text engine (text=Some Text)."
)


def _harden_playwright_description(tool_name: str, input_schema: dict, description: str) -> str:
    """Append usage notes to a playwright tool's description so the model picks the
    right calling convention (canonical arg names; no page object in evaluate)."""
    props = (input_schema or {}).get("properties") or {}
    if tool_name == "browser_evaluate":
        description += _EVALUATE_NOTE
    elif "target" in props or "startTarget" in props:
        description += _TARGET_NOTE
    return description


# Markers in a pasted snapshot line ("- link \"Text\" [ref=e6]") or an invented
# attribute selector ("\"link\" [text=Text]") that a model sometimes sends as the
# target instead of just the ref id — playwright then parses it as CSS and dies with
# "Error while parsing selector".
_SNAPSHOT_REF_RE = re.compile(r"\[\s*ref\s*=\s*([A-Za-z0-9_-]+)\s*\]")
_TEXT_ATTR_RE = re.compile(
    r"\[\s*text\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\]\s][^\]]*?))\s*\]"
)


def _unwrap_wrapping_quotes(value: str) -> str:
    """Strip one pair of matching wrapping quotes — models sometimes copy the ref
    out of a snapshot wrapped in literal quotes, and "'e3'" parses as a broken CSS
    selector ("Unexpected token while parsing css selector")."""
    if len(value) > 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _repair_target(value: str) -> str:
    """Salvage a usable target from common model mis-formattings.

    1. A pasted snapshot line ('link "Text" [ref=e6]') → the bare ref 'e6'.
    2. An invented attribute selector ('"link" [text=Text]') → Playwright's text
       engine ('text=Text'), which the server accepts as a real selector.
    Plain refs and real CSS selectors pass through unchanged.
    """
    m = _SNAPSHOT_REF_RE.search(value)
    if m:
        return m.group(1)
    m = _TEXT_ATTR_RE.search(value)
    if m:
        text = next(g for g in m.groups() if g is not None).strip()
        if text:
            return f"text={text}"
    return value


def _clean_arguments(args: dict) -> dict:
    """Normalize arguments before the call hits the MCP server.

    1. Drop arguments explicitly set to null. The deepagents wrapper
       (core/deepagents/tools.py) already strips Nones, but call_mcp is also reached
       from other paths (chat/mcp_client, ACP) — and a model can still emit an
       explicit null for an optional param. MCP servers like @playwright/mcp type
       optional params as plain "string" in their schema, so an explicit JSON null is
       a schema violation ("expected string, received null"). In JSON Schema an
       optional argument means OMITTED, never null.
    2. Alias legacy `ref`-style arg names to the canonical `target`-style ones.
    3. Unwrap quotes wrapped around element targets.
    4. Repair targets mangled into pasted snapshot lines or invented selectors.
    """
    cleaned = {k: v for k, v in args.items() if v is not None}
    for old, new in _REF_ALIASES.items():
        if old in cleaned:
            cleaned.setdefault(new, cleaned.pop(old))
    for key in _TARGET_KEYS:
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = _repair_target(_unwrap_wrapping_quotes(value))
    return cleaned


def _missing_file_url(args: dict) -> str | None:
    """The url arg of an MCP call, if it's a file:// URL pointing at a nonexistent path."""
    import os
    from urllib.parse import unquote, urlparse

    url = args.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    path = unquote(urlparse(url).path or "")
    # isfile, not exists: browsing a directory renders a listing or errors in headless mode
    return None if os.path.isfile(path) else url


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
    if not isinstance(arguments, dict):
        return f"ERROR: MCP arguments must be an object, got {type(arguments).__name__}"
    arguments = _clean_arguments(arguments)
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
        result = await asyncio.wait_for(
            session.call_tool(real_name, arguments), timeout=_CALL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"ERROR: {fq_name} timed out after {int(_CALL_TIMEOUT)}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR calling {fq_name}: {exc}"
    # Parse result defensively — a malformed MCP response must not crash the agent loop.
    try:
        parts = []
        for chunk in (getattr(result, "content", None) or []):
            if getattr(chunk, "type", None) == "text":
                parts.append(getattr(chunk, "text", str(chunk)))
            else:
                parts.append(f"[{getattr(chunk, 'type', 'content')} content]")
        text = "\n".join(parts) or "(no output)"
        if getattr(result, "isError", False):
            text = "ERROR: " + text
        # Cap the result so a large snapshot or HTML dump doesn't blow the context window.
        if len(text) > _MAX_RESULT_CHARS:
            text = text[:_MAX_RESULT_CHARS] + f"\n… (truncated {len(text) - _MAX_RESULT_CHARS} chars)"
    except Exception as exc:  # noqa: BLE001 — malformed result shape
        text = f"ERROR parsing result from {fq_name}: {exc}"
    return text
