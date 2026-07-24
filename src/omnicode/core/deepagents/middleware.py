"""Middleware for the deepagents integration.

:class:`PermissionMiddleware` is kept as an option but the recommended approach is to
wrap the executor closure itself with permission gating (see
:func:`adapter.build_agent`), so LangGraph's ``on_tool_start`` / ``on_tool_end`` events
fire for every tool call — even denied ones — keeping the omnicode event stream complete.

The permission gate is built into the tool executor rather than as a separate middleware
because LangChain middleware that short-circuits (doesn't call ``handler``) prevents
``on_tool_start`` / ``on_tool_end`` from firing, which would drop omnicode ``ToolStarted`` /
``ToolFinished`` events for denied tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage


# (name, args) -> "once" | "all" | "deny"
PermissionPolicy = Callable[[str, dict], Awaitable[str]]


@dataclass
class _ToolEventCapture:
    """Accumulates tool-call observations (reserved for future use; currently the adapter
    reads events from ``astream_events`` directly)."""

    started: list[dict] = field(default_factory=list)
    finished: list[dict] = field(default_factory=list)
    auto_approved: bool = False


class PermissionMiddleware(AgentMiddleware):
    """Gates mutating tools behind the injected permission callback.

    .. deprecated::
        Prefer wrapping the executor closure with permission gating instead (see
        :func:`adapter.build_agent`). This middleware is kept for back-compat but note
        that short-circuiting (deny) prevents ``on_tool_start`` / ``on_tool_end`` from
        firing, so omnicode ``ToolStarted`` / ``ToolFinished`` events are NOT emitted for
        denied tools when using this middleware.
    """

    def __init__(
        self,
        permission: PermissionPolicy,
        mutating: frozenset[str],
        capture: _ToolEventCapture,
    ) -> None:
        self._permission = permission
        self._mutating = mutating
        self._capture = capture

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args") or {}
        if name not in self._mutating or self._capture.auto_approved:
            return await handler(request)
        decision = await self._permission(name, args)
        if decision == "all":
            self._capture.auto_approved = True
            return await handler(request)
        if decision == "once":
            return await handler(request)
        # deny → feed the model a denial instead of running the tool
        return ToolMessage(
            content="The user DENIED this action. Do not retry it; ask how to proceed.",
            tool_call_id=request.tool_call.get("id") or "",
            name=name,
        )


# The built-in fs/execute tools that MUTATE state → gated behind the permission prompt.
# Read-only built-ins (ls, read_file, glob, grep) run without asking.
_LIS_FS_MUTATING = frozenset({"write_file", "edit_file", "execute"})

# Built-in fs tools whose path args are interpreted relative to the backend's virtual root.
_LIS_FS_TOOLS = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})


def _strip_fs_root(fs_root: str, value: str) -> str | None:
    """Map an absolute HOST path inside ``fs_root`` back to its virtual path.

    Returns the virtual path for absolute paths inside the root, ``value`` unchanged
    for non-absolute input, and ``None`` for absolute paths OUTSIDE the root.

    The system note names the working directory by its absolute path, so models
    naturally emit absolute paths like ``/work/proj/index.html`` — but the
    ``LocalShellBackend(virtual_mode=True)`` joins EVERY incoming path under
    ``root_dir`` (it only rejects ``..``/``~``), so an absolute path lands at the
    doubled ``{root}/abs/path/...`` — even one outside the root (e.g. a model typo
    like ``/Users/nick/DesktopTest/index.html`` with a fused ``Desktop/Test`` →
    ``DesktopTest`` creates a whole garbage tree under the working dir). Callers
    must therefore treat outside-root absolutes as an error, not pass them through.
    """
    import os

    root = os.path.normpath(os.path.expanduser(fs_root))
    if not value or not value.startswith("/"):
        return value
    norm = os.path.normpath(value)
    if norm == root:
        return "/"
    if norm.startswith(root + os.sep):
        return norm[len(root):]
    return None


class _LISFileToolsMiddleware(AgentMiddleware):
    """Permission-gates deepagents' built-in filesystem + ``execute`` tools.

    When a chat has a working dir, ``build_agent`` wires a ``LocalShellBackend`` so the
    built-ins do the real file work. This middleware prompts the user before any
    MUTATING built-in runs (write_file / edit_file / execute), honouring the same
    once / all / deny contract as the old executor-closure gate. Unlike
    :class:`PermissionMiddleware`, a deny RETURNS a ``Command(goto=END)`` *through* the
    normal tool node, so ``on_tool_start`` / ``on_tool_end`` still fire and the omnicode
    ``ToolStarted`` / ``ToolFinished`` events stay complete (LangChain drops the tool
    events entirely when a middleware returns a bare ``ToolMessage``).
    """

    def __init__(self, permission: PermissionPolicy, capture: _ToolEventCapture, fs_root: str | None = None) -> None:
        self._permission = permission
        self._capture = capture
        self._fs_root = fs_root

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args") or {}
        # Models trained on the old omnicode fs schema (or generic tool-calling data) often
        # emit ``path`` where deepagents' built-ins require ``file_path`` — normalize
        # before anything (validation or the permission prompt) sees the call.
        if (
            name in ("read_file", "write_file", "edit_file")
            and isinstance(args, dict)
            and "file_path" not in args
            and "path" in args
        ):
            args = {**args, "file_path": args["path"]}
            request = request.override(tool_call={**request.tool_call, "args": args})
        # Models often emit absolute HOST paths (the system note names the working dir
        # by its absolute path), but virtual_mode joins EVERY path under root — an
        # absolute path would land at the doubled ``{root}/abs/path/...``. Paths inside
        # the root are mapped back to virtual; absolute paths OUTSIDE the root are
        # rejected outright with guidance (otherwise they silently create a garbage
        # directory tree under the working dir).
        if self._fs_root and name in _LIS_FS_TOOLS and isinstance(args, dict):
            changed = dict(args)
            outside: str | None = None
            for key in ("file_path", "path"):
                if isinstance(changed.get(key), str):
                    stripped = _strip_fs_root(self._fs_root, changed[key])
                    if stripped is None:
                        outside = changed[key]
                        break
                    changed[key] = stripped
            if outside is not None:
                return ToolMessage(
                    content=(
                        f"ERROR: {outside} is outside the working directory. "
                        "Pass paths RELATIVE to the working directory "
                        "(e.g. \"index.html\", \"src/app.py\")."
                    ),
                    tool_call_id=request.tool_call.get("id") or "",
                    name=name,
                    status="error",
                )
            if changed != args:
                args = changed
                request = request.override(tool_call={**request.tool_call, "args": args})
        if (
            name not in _LIS_FS_MUTATING
            or self._permission is None
            or self._capture.auto_approved
        ):
            return await handler(request)
        decision = await self._permission(name, args)
        if decision == "all":
            self._capture.auto_approved = True
            return await handler(request)
        if decision == "once":
            return await handler(request)
        # Deny: emit ToolStarted+ToolFinished ourselves (LangChain suppresses the tool
        # node's events for middleware-short-circuited calls, so omnicode would otherwise
        # see no tool activity for the denied call), then route straight to END with a
        # denial message for the model.
        from langgraph.graph import END
        from langgraph.types import Command

        denial_text = "The user DENIED this action. Do not retry it; ask how to proceed."
        try:
            from langchain_core.callbacks.manager import adispatch_custom_event

            await adispatch_custom_event("lis_tool_denied", {
                "id": request.tool_call.get("id") or "",
                "name": name,
                "args": args,
                "text": denial_text,
            })
        except Exception:  # noqa: BLE001 — no runnable config (e.g. unit tests) → skip
            pass
        denial = ToolMessage(
            content=denial_text,
            tool_call_id=request.tool_call.get("id") or "",
            name=name,
        )
        return Command(update={"messages": [denial]}, goto=END)
