"""An ACP (Agent Client Protocol) agent that bridges an editor (e.g. Xcode 27)
to a running mlx_lm.server.

Two modes, chosen per prompt:
  * chat    — stream assistant text (used when the client exposes no fs/terminal
              tools, or tools are disabled).
  * agentic — an OpenAI tool-calling loop where the model can read_file /
              write_file / run_command, executed via the ACP client's filesystem
              and terminal methods (with a permission prompt before writes), and
              surfaced as ACP tool_call / tool_call_update notifications."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

import acp
import httpx
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
from .bridge import MlxBridge

log = logging.getLogger("mlx-acp-agent")

# OpenAI finish_reason -> ACP StopReason
_FINISH_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "cancelled": "cancelled",
    "content_filter": "refusal",
    "tool_calls": "end_turn",
}

_MAX_TOOL_ITERS = 12
_TOOL_OUTPUT_LIMIT = 4000


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


class MlxAcpAgent:
    """Implements the subset of the ACP Agent protocol we support."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", use_tools: bool = True) -> None:
        self.base_url = base_url
        self.model = model
        self.use_tools = use_tools
        # match the chat UI's budget so reasoning models aren't cut off at the
        # server's 512-token default (see chat/client.py:DEFAULT_MAX_TOKENS)
        self.bridge = MlxBridge(base_url, model, api_key, max_tokens=16384)
        self._client: Optional[acp.Client] = None
        self._sessions: dict[str, list[dict]] = {}
        self._cancelled: set[str] = set()
        # client capabilities (learned at initialize)
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
            protocol_version=acp.PROTOCOL_VERSION,
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
        if self.use_tools and (self._fs_read or self._fs_write or self._terminal):
            return await self._agentic_prompt(prompt, session_id)
        return await self._chat_prompt(prompt, session_id)

    # --- chat mode (streaming) ------------------------------------------

    async def _chat_prompt(self, prompt: list, session_id: str) -> acp.PromptResponse:
        history = self._sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": _blocks_to_text(prompt)})
        self._cancelled.discard(session_id)

        assistant: list[str] = []
        finish = "stop"
        try:
            async for kind, text in self.bridge.stream_chat(history, cancel=lambda: session_id in self._cancelled):
                if kind == "content":
                    assistant.append(text)
                    await self._emit(session_id, update_agent_message_text(text))
                elif kind == "reason":
                    await self._emit(session_id, update_agent_thought_text(text))
                elif kind == "finish":
                    finish = text
        except (httpx.HTTPError, OSError) as exc:
            await self._emit(session_id, update_agent_message_text(self._unreachable_msg(exc)))
            self._finalize(history, assistant)
            return acp.PromptResponse(stop_reason="end_turn")

        self._finalize(history, assistant)
        if session_id in self._cancelled:
            self._cancelled.discard(session_id)
            return acp.PromptResponse(stop_reason="cancelled")
        return acp.PromptResponse(stop_reason=_FINISH_MAP.get(finish, "end_turn"))

    # --- agentic mode (tool calling) ------------------------------------

    async def _agentic_prompt(self, prompt: list, session_id: str) -> acp.PromptResponse:
        history = self._sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": _blocks_to_text(prompt)})
        self._cancelled.discard(session_id)
        tools = self._tool_specs()

        for _ in range(_MAX_TOOL_ITERS):
            if session_id in self._cancelled:
                self._cancelled.discard(session_id)
                return acp.PromptResponse(stop_reason="cancelled")
            try:
                data = await self.bridge.chat(history, tools=tools or None)
            except (httpx.HTTPError, OSError) as exc:
                await self._emit(session_id, update_agent_message_text(self._unreachable_msg(exc)))
                return acp.PromptResponse(stop_reason="end_turn")

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                history.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            else:
                history.append({"role": "assistant", "content": content})
            if content:
                await self._emit(session_id, update_agent_message_text(content))

            if not tool_calls:
                return acp.PromptResponse(stop_reason=_FINISH_MAP.get(choice.get("finish_reason", "stop"), "end_turn"))

            for call in tool_calls:
                result = await self._run_tool(session_id, call)
                history.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})

        return acp.PromptResponse(stop_reason="max_turn_requests")

    def _tool_specs(self) -> list[dict]:
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
        return specs

    async def _run_tool(self, session_id: str, call: dict) -> str:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_id = call.get("id") or name or "tool"
        kind = {"read_file": "read", "write_file": "edit", "run_command": "execute"}.get(name, "other")
        title = {
            "read_file": f"Read {args.get('path', '')}",
            "write_file": f"Edit {args.get('path', '')}",
            "run_command": f"Run: {args.get('command', '')}",
        }.get(name, name)
        await self._emit(session_id, start_tool_call(tool_id, title, kind=kind, status="in_progress"))
        try:
            if name == "read_file" and self._fs_read:
                resp = await self._client.read_text_file(path=args["path"], session_id=session_id)
                await self._done_tool(session_id, tool_id, _truncate(resp.content))
                return resp.content
            if name == "write_file" and self._fs_write:
                if not await self._request_write_permission(session_id, tool_id, args.get("path", "")):
                    await self._fail_tool(session_id, tool_id, "permission denied")
                    return "The user denied permission to write this file."
                await self._client.write_text_file(content=args.get("content", ""), path=args["path"], session_id=session_id)
                await self._done_tool(session_id, tool_id, f"wrote {args['path']}")
                return f"Wrote {args['path']}."
            if name == "run_command" and self._terminal:
                out = await self._run_terminal(session_id, args.get("command", ""), args.get("cwd"))
                await self._done_tool(session_id, tool_id, _truncate(out))
                return out
            await self._fail_tool(session_id, tool_id, "tool unavailable")
            return f"Tool '{name}' is not available."
        except KeyError as exc:
            await self._fail_tool(session_id, tool_id, f"missing argument {exc}")
            return f"Tool '{name}' missing argument {exc}."
        except Exception as exc:  # noqa: BLE001
            await self._fail_tool(session_id, tool_id, str(exc))
            return f"Tool '{name}' failed: {exc}"

    async def _request_write_permission(self, session_id: str, tool_id: str, path: str) -> bool:
        if self._client is None:
            return True
        options = [
            PermissionOption(kind="allow_once", name="Allow", option_id="allow"),
            PermissionOption(kind="reject_once", name="Reject", option_id="reject"),
        ]
        tool_call = ToolCallUpdate(tool_call_id=tool_id, title=f"Write {path}", kind="edit", status="pending")
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
            await self._client.wait_for_terminal_exit(session_id=session_id, terminal_id=tid)
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

    async def _done_tool(self, session_id: str, tool_id: str, text: str) -> None:
        await self._emit(session_id, update_tool_call(tool_id, status="completed", content=[tool_content(text_block(text))]))

    async def _fail_tool(self, session_id: str, tool_id: str, text: str) -> None:
        await self._emit(session_id, update_tool_call(tool_id, status="failed", content=[tool_content(text_block(text))]))

    def _unreachable_msg(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"\n\n⚠️ MLX server returned {exc.response.status_code} at {self.base_url}."
        return f"\n\n⚠️ Could not reach the MLX server at {self.base_url}.\n{exc}"

    @staticmethod
    def _finalize(history: list[dict], assistant: list[str]) -> None:
        if assistant:
            history.append({"role": "assistant", "content": "".join(assistant)})


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
