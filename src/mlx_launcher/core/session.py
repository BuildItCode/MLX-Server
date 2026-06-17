"""A chat session: a :class:`~mlx_launcher.models.chat.Chat` plus its resolved server profile,
project, skill, and MCP servers, with the pure computations the backend needs to drive a turn —
the engine config + token budget, the OpenAI message list, the filesystem root + system note, the
tool set, and context-window usage.

Textual-free; ported from the ChatScreen helpers (``_client``, ``_sampling_of``,
``_context_cap_of``, ``_effective_context``, ``_context_usage``, ``_fs_root``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from ..engine import capabilities
from ..engine.base import Engine, EngineConfig, build_engine
from ..models import Chat, McpServer, Project, ServerConfig
from . import skills
from .agent import ToolOutcome, ToolSet
from .messages import build_openai_messages, scaled_max_tokens
from .persistence import chats as chats_store
from .persistence import config as config_store
from .tools import fs, mcp, web

# (url) -> opened? — a client-side action the backend asks a frontend to perform (open_in_browser).
OpenUrlHandler = Callable[[str], Awaitable[bool]]


def context_cap_of(cfg: Optional[ServerConfig]) -> Optional[int]:
    """The context size the user explicitly configured on a profile (mlx-vlm/vllm-mlx
    ``--max-kv-size``, llama-cpp ``-c``), or None. mlx-lm can't cap context, so its setting is
    ignored."""
    engine = getattr(cfg, "engine", None) if cfg else None
    if engine in ("mlx-vlm", "vllm-mlx"):
        return cfg.max_kv_size or None
    if engine == "llama-cpp":
        return cfg.ctx or None
    return None


def sampling_of(cfg: Optional[ServerConfig]) -> dict:
    """A profile's sampling settings as OpenAI request params (only values the user set). Sent in
    the request body, so they work on every OpenAI-compatible engine."""
    out: dict = {}
    if cfg is None:
        return out
    if cfg.temp is not None:
        out["temperature"] = cfg.temp
    if cfg.top_p is not None:
        out["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        out["top_k"] = cfg.top_k
    if cfg.min_p is not None:
        out["min_p"] = cfg.min_p
    return out


@dataclass
class Session:
    chat: Chat
    server: Optional[ServerConfig] = None
    project: Optional[Project] = None
    skill_instructions: Optional[str] = None
    mcp_servers: list[McpServer] = field(default_factory=list)

    @classmethod
    def resolve(cls, chat: Chat) -> "Session":
        """Build a Session for ``chat`` from the on-disk stores (server profile, project, skill,
        MCP servers)."""
        cfg = config_store.find_server_by_id(chat.server_id) if chat.server_id else None
        cfile = chats_store.load()
        project = chats_store.get_project(cfile, chat.project_id) if chat.project_id else None
        return cls(
            chat=chat,
            server=cfg,
            project=project,
            skill_instructions=skills.instructions_for(chat.skill_id),
            mcp_servers=list(cfile.mcp_servers),
        )

    # --- engine + budget -------------------------------------------------

    def max_tokens(self) -> int:
        """The profile's explicit ``--max-tokens`` if set, else a budget scaled to the context
        window — never the server's truncating 512-token default."""
        if self.server and self.server.max_tokens:
            return self.server.max_tokens
        return scaled_max_tokens(self.chat.model, context_cap_of(self.server))

    def engine_config(self) -> EngineConfig:
        ctk = capabilities.reasoning_template_kwargs(self.chat.model, self.chat.reasoning_effort)
        return EngineConfig(
            base_url=self.chat.base_url,
            model=self.chat.model,
            max_tokens=self.max_tokens(),
            chat_template_kwargs=ctk or None,
            sampling=sampling_of(self.server) or None,
        )

    def engine(self) -> Engine:
        return build_engine(self.engine_config())

    # --- messages + filesystem ------------------------------------------

    def messages(self) -> list[dict]:
        return build_openai_messages(self.chat, self.project, self.skill_instructions)

    def fs_root(self) -> Optional[str]:
        """The project's working directory if it exists on disk, else None."""
        if self.project and self.project.working_dir:
            path = os.path.expanduser(self.project.working_dir)
            if os.path.isdir(path):
                return path
        return None

    def system_note(self) -> Optional[str]:
        root = self.fs_root()
        return fs.system_note(root) if root else None

    # --- context metering -----------------------------------------------

    def effective_context(self) -> Optional[int]:
        model_max = capabilities.context_window(self.chat.model)
        cap = context_cap_of(self.server)
        if cap and model_max:
            return min(cap, model_max)
        return cap or model_max

    def context_usage(self) -> Optional[tuple[int, int]]:
        """(estimated tokens used, context window), or None when the window is unknown."""
        window = self.effective_context()
        if not window:
            return None
        used = capabilities.estimate_prompt_tokens(self.messages())
        root = self.fs_root()
        if root:
            used += capabilities.approx_tokens(fs.system_note(root))
        return used, window

    # --- tools -----------------------------------------------------------

    async def build_toolset(self, stack, *, open_url: Optional[OpenUrlHandler] = None,
                            on_mcp_error: Optional[Callable[[str, str], None]] = None) -> ToolSet:
        """Assemble the tools this chat allows: web_search, the sandboxed fs tools (when a working
        dir is set), and the connected MCP servers' tools. MCP sessions are opened on ``stack``
        (closed when it exits). The executor runs web/fs/MCP server-side; ``open_in_browser`` is a
        client action delegated to ``open_url``."""
        specs: list[dict] = []
        if self.chat.web_search:
            specs.append(web.web_search_spec())
        root = self.fs_root()
        if root:
            specs += fs.fs_specs()
        sessions: dict = {}
        router: dict = {}
        if self.chat.tools and self.mcp_servers:
            sessions, mcp_specs, router = await mcp.open_sessions(stack, self.mcp_servers, on_error=on_mcp_error)
            specs += mcp_specs

        async def execute(name: str, args: dict) -> ToolOutcome:
            if name == "web_search":
                return ToolOutcome(await web.run_web_search(args.get("query", ""), args.get("max_results", 6)))
            if root and name in fs.FS_TOOL_NAMES:
                if name == "open_in_browser":
                    return await self._open_in_browser(root, args, open_url)
                return ToolOutcome(await fs.run_fs_tool(root, name, args))
            if name in router:
                return ToolOutcome(await mcp.call_mcp(sessions, router, name, args))
            return ToolOutcome(f"Unknown tool: {name}", ok=False)

        mutating = fs.MUTATING_TOOLS if root else frozenset()
        return ToolSet(specs=specs, execute=execute, mutating=frozenset(mutating))

    @staticmethod
    async def _open_in_browser(root: str, args: dict, open_url: Optional[OpenUrlHandler]) -> ToolOutcome:
        """Resolve the model's target (a file in the working dir or an http(s) URL, confined to the
        root) and ask the frontend to open it. Headless backends without an ``open_url`` handler
        just report the resolved URL."""
        target = args.get("path") or args.get("url") or ""
        try:
            url = fs.resolve_browser_target(root, target)
        except ValueError as exc:
            return ToolOutcome(f"error: {exc}", ok=False)
        if open_url is None:
            return ToolOutcome(f"Open this URL to view it: {url}", ok=True)
        try:
            ok = await open_url(url)
        except Exception as exc:  # noqa: BLE001
            return ToolOutcome(f"error: couldn't open the browser: {exc}", ok=False)
        return ToolOutcome(f"Opened {url} in the browser." if ok else f"Could not open {url}.", ok=ok)
