"""An ACP (Agent Client Protocol) agent that bridges an editor (e.g. Xcode 27) to a local model.

The agent loop itself is the shared :class:`mlx_launcher.core.agent.AgentRunner` — the ACP agent is
now a thin translator: it owns the ACP protocol (sessions, streaming session updates, permission
prompts) and supplies the runner with (a) a tool executor that **delegates filesystem + terminal
work to the editor** via the ACP client (so edits flow through Xcode's buffers/terminals, the
faithful ACP model) and (b) a permission policy backed by ``request_permission``. This removes the
duplicated tool loop that used to live here and gives ACP the runner's format-recovery + truncation
handling for free.

Two shapes, chosen per prompt: with no editor fs/terminal access (or ``--no-tools``) the runner
streams a plain chat answer; otherwise it runs the tool loop over the editor-delegated tools."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

import acp
from acp import (
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message_text,
    update_agent_thought_text,
    update_tool_call,
)
from acp.schema import (
    AgentCapabilities,
    Implementation,
    PermissionOption,
    PromptCapabilities,
    ToolCallUpdate,
)

from .. import __version__
from ..core import events as ev
from ..core.agent import AgentRunner, RunPolicy, ToolOutcome, ToolSet
from ..engine.openai import OpenAIEngine

log = logging.getLogger("mlx-acp-agent")

# OpenAI finish_reason / TurnFinished.reason -> ACP StopReason
_FINISH_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "cancelled": "cancelled",
    "content_filter": "refusal",
    "tool_calls": "end_turn",
}
_TOOL_KIND = {"read_file": "read", "write_file": "edit", "run_command": "execute"}

_MAX_TOOL_ITERS = 12
_TOOL_OUTPUT_LIMIT = 4000
_TERMINAL_TIMEOUT = 120.0  # cap a single run_command so a hung process can't block the turn


def _truncate(text: str, limit: int = _TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} chars]"


def _blocks_to_text(blocks: list) -> str:
    """Flatten ACP content blocks (text, embedded resources, links) into a prompt."""
    parts: list[str] = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text:
            parts.append(text)
            continue
        resource = getattr(b, "resource", None)
        if resource is not None:
            rtext = getattr(resource, "text", None)
            uri = getattr(resource, "uri", None)
            if rtext:
                parts.append(f'<context uri="{uri}">\n{rtext}\n</context>' if uri else rtext)
                continue
        uri = getattr(b, "uri", None)
        if uri:
            parts.append(f"[attached resource: {uri}]")
    return "\n".join(parts)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


class MlxAcpAgent:
    """Implements the subset of the ACP Agent protocol we support, driving core.AgentRunner."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", use_tools: bool = True) -> None:
        self.base_url = base_url
        self.model = model
        self.use_tools = use_tools
        # match the chat UI's budget so reasoning models aren't cut off at the server's 512 default
        self.engine = OpenAIEngine(base_url, model, api_key, max_tokens=16384)
        self._client: Optional[acp.Client] = None
        self._sessions: dict[str, list[dict]] = {}
        self._cancelled: set[str] = set()
        # One lock per session so two overlapping prompts can't interleave appends into one history.
        self._locks: dict[str, asyncio.Lock] = {}
        self._fs_read = False
        self._fs_write = False
        self._terminal = False

    def on_connect(self, conn: acp.Client) -> None:
        self._client = conn

    # --- protocol --------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        fs = getattr(client_capabilities, "fs", None) if client_capabilities else None
        self._fs_read = bool(getattr(fs, "read_text_file", False)) if fs else False
        self._fs_write = bool(getattr(fs, "write_text_file", False)) if fs else False
        self._terminal = bool(getattr(client_capabilities, "terminal", False)) if client_capabilities else False
        log.info("client caps: fs_read=%s fs_write=%s terminal=%s", self._fs_read, self._fs_write, self._terminal)
        return acp.InitializeResponse(
            protocol_version=min(protocol_version, acp.PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(image=False, audio=False, embedded_context=True),
            ),
            agent_info=Implementation(name="mlx-acp-agent", version=__version__),
            auth_methods=[],
        )

    async def new_session(self, cwd: str, additional_directories: Any = None, mcp_servers: Any = None, **kwargs: Any) -> acp.NewSessionResponse:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = []
        return acp.NewSessionResponse(session_id=session_id)

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._cancelled.add(session_id)

    async def prompt(self, prompt: list, session_id: str, message_id: Optional[str] = None, **kwargs: Any) -> acp.PromptResponse:
        if session_id not in self._sessions:
            raise acp.RequestError.invalid_params({"session_id": session_id, "reason": "unknown session"})
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._run(prompt, session_id)

    # --- the run (translate AgentRunner events ⇄ ACP) --------------------

    async def _run(self, prompt: list, session_id: str) -> acp.PromptResponse:
        history = self._sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": _blocks_to_text(prompt)})
        self._cancelled.discard(session_id)

        tools = self._build_toolset(session_id) if self.use_tools else ToolSet()
        runner = AgentRunner(
            self.engine,
            tools=tools,
            policy=RunPolicy(max_iters=_MAX_TOOL_ITERS, native_tools=True),
            permission=self._permission_policy(session_id),
            cancel=lambda: session_id in self._cancelled,
        )

        streamed = False
        finish = "stop"
        try:
            async for event in runner.run(history):
                if isinstance(event, ev.ContentDelta):
                    streamed = True
                    await self._emit(session_id, update_agent_message_text(event.text))
                elif isinstance(event, ev.ReasonDelta):
                    await self._emit(session_id, update_agent_thought_text(event.text))
                elif isinstance(event, ev.ToolStarted):
                    kind = _TOOL_KIND.get(event.name, "other")
                    await self._emit(session_id, start_tool_call(event.tool_id, event.phrase,
                                                                  kind=kind, status="in_progress"))
                elif isinstance(event, ev.ToolFinished):
                    status = "completed" if event.status == "ok" else "failed"
                    await self._emit(session_id, update_tool_call(
                        event.tool_id, status=status,
                        content=[tool_content(text_block(_truncate(event.preview or event.result)))]))
                elif isinstance(event, ev.TurnFinished):
                    finish = event.reason
                    if event.text and not streamed:  # tool loop doesn't stream — emit the answer now
                        await self._emit(session_id, update_agent_message_text(event.text))
                elif isinstance(event, ev.TurnFailed):
                    await self._emit(session_id, update_agent_message_text(self._unreachable_msg(event.error)))
                    history.extend(runner.turns)
                    return acp.PromptResponse(stop_reason="end_turn")
        except (OSError,) as exc:  # transport-level failure outside the runner's own handling
            await self._emit(session_id, update_agent_message_text(self._unreachable_msg(str(exc))))
            return acp.PromptResponse(stop_reason="end_turn")

        history.extend(runner.turns)  # keep the exchange (assistant + tool turns) for the next prompt
        if session_id in self._cancelled:
            self._cancelled.discard(session_id)
            return acp.PromptResponse(stop_reason="cancelled")
        return acp.PromptResponse(stop_reason=_FINISH_MAP.get(finish, "end_turn"))

    # --- editor-delegated tools -----------------------------------------

    def _build_toolset(self, session_id: str) -> ToolSet:
        specs: list[dict] = []
        if self._fs_read:
            specs.append(_fn("read_file", "Read a UTF-8 text file from the user's workspace.",
                             {"path": {"type": "string", "description": "File path"}}, ["path"]))
        if self._fs_write:
            specs.append(_fn("write_file", "Create or overwrite a UTF-8 text file in the workspace.",
                             {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]))
        if self._terminal:
            specs.append(_fn("run_command", "Run a shell command in the workspace and return its output.",
                             {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command"]))

        async def execute(name: str, args: dict) -> ToolOutcome:
            try:
                if name == "read_file" and self._fs_read:
                    resp = await self._client.read_text_file(path=args["path"], session_id=session_id)
                    return ToolOutcome(resp.content)
                if name == "write_file" and self._fs_write:
                    await self._client.write_text_file(content=args.get("content", ""),
                                                       path=args["path"], session_id=session_id)
                    return ToolOutcome(f"Wrote {args['path']}.")
                if name == "run_command" and self._terminal:
                    return ToolOutcome(await self._run_terminal(session_id, args.get("command", ""), args.get("cwd")))
                return ToolOutcome(f"Tool '{name}' is not available.", ok=False)
            except KeyError as exc:
                return ToolOutcome(f"Tool '{name}' missing argument {exc}.", ok=False)

        return ToolSet(specs=specs, execute=execute, mutating=frozenset({"write_file"}))

    def _permission_policy(self, session_id: str):
        async def permission(name: str, args: dict) -> str:
            return "once" if await self._request_write_permission(session_id, args.get("path", "")) else "deny"
        return permission

    async def _request_write_permission(self, session_id: str, path: str) -> bool:
        if self._client is None:
            return True
        options = [
            PermissionOption(kind="allow_once", name="Allow", option_id="allow"),
            PermissionOption(kind="reject_once", name="Reject", option_id="reject"),
        ]
        tool_call = ToolCallUpdate(tool_call_id=uuid.uuid4().hex, title=f"Write {path}", kind="edit", status="pending")
        try:
            resp = await self._client.request_permission(options=options, session_id=session_id, tool_call=tool_call)
        except Exception:  # noqa: BLE001 — client may not support permissions; proceed
            return True
        outcome = getattr(resp, "outcome", None)
        return getattr(outcome, "outcome", None) == "selected" and getattr(outcome, "option_id", "") == "allow"

    async def _run_terminal(self, session_id: str, command: str, cwd: Optional[str]) -> str:
        resp = await self._client.create_terminal(command="/bin/sh", session_id=session_id, args=["-c", command], cwd=cwd)
        tid = resp.terminal_id
        try:
            try:
                await asyncio.wait_for(
                    self._client.wait_for_terminal_exit(session_id=session_id, terminal_id=tid),
                    timeout=_TERMINAL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                return f"command timed out after {int(_TERMINAL_TIMEOUT)}s"
            out = await self._client.terminal_output(session_id=session_id, terminal_id=tid)
            return out.output
        finally:
            try:
                await self._client.release_terminal(session_id=session_id, terminal_id=tid)
            except Exception:  # noqa: BLE001
                pass

    # --- helpers ---------------------------------------------------------

    async def _emit(self, session_id: str, update: Any) -> None:
        if self._client is not None:
            await self._client.session_update(session_id=session_id, update=update)

    def _unreachable_msg(self, error: str) -> str:
        return f"\n\n⚠️ Could not reach the MLX server at {self.base_url}.\n{error}"
