"""A chat session: a :class:`~omnicode.models.chat.Chat` plus its resolved server profile,
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
from ..models import Chat, McpServer, ServerConfig, Subagent
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
        """Build a Session for ``chat`` from the on-disk stores (server profile, skill,
        MCP servers). Projects were removed — working_dir + instructions are now on the chat."""
        cfg = config_store.find_server_by_id(chat.server_id) if chat.server_id else None
        cfile = chats_store.load()
        return cls(
            chat=chat,
            server=cfg,
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
        """The chat's working directory if it exists on disk, else None. Checks BOTH the
        legacy project link and the chat's own working_dir, so old chats (still carrying
        only ``project_id``) and new chats (working_dir set directly) both get their
        sandboxed fs tools."""
        proj = chats_store.get_project(chats_store.load(), self.chat.project_id)
        candidates = []
        if proj and proj.working_dir:
            candidates.append(proj.working_dir)
        if self.chat.working_dir:
            candidates.append(self.chat.working_dir)
        for cand in candidates:
            path = os.path.expanduser(cand)
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
        """Assemble the tools this chat allows: web_search, ``open_in_browser`` (when a
        working dir is set), and the connected MCP servers' tools. MCP sessions are
        opened on ``stack`` (closed when it exits).

        The actual filesystem work (read/write/edit/ls/grep/execute) is done by
        deepagents' built-in tools driven by the ``LocalShellBackend`` that
        ``build_agent`` wires for this working dir — those names are deliberately NOT in
        this user-tool set (they'd collide with the built-ins). ``open_in_browser`` is
        the one fs-adjacent action deepagents doesn't provide, so it stays a user tool,
        delegated to ``open_url``."""
        specs: list[dict] = []
        if self.chat.web_search:
            specs.append(web.web_search_spec())
        root = self.fs_root()
        if root:
            # Only the browser-open action stays a user tool; the fs read/write/edit/
            # ls/grep/execute work is deepagents' built-ins (wired via the backend).
            specs += [s for s in fs.fs_specs()
                      if (s.get("function") or {}).get("name") == "open_in_browser"]
        sessions: dict = {}
        router: dict = {}
        # MCP tools are always available when connectors are configured (was a toggle, now on).
        if self.mcp_servers:
            sessions, mcp_specs, router = await mcp.open_sessions(stack, self.mcp_servers, on_error=on_mcp_error)
            specs += mcp_specs

        async def execute(name: str, args: dict) -> ToolOutcome:
            if name == "web_search":
                return ToolOutcome(await web.run_web_search(args.get("query", ""), args.get("max_results", 6)))
            if root and name == "open_in_browser":
                return await self._open_in_browser(root, args, open_url)
            if name in router:
                return ToolOutcome(await mcp.call_mcp(sessions, router, name, args))
            return ToolOutcome(f"Unknown tool: {name}", ok=False)

        # open_in_browser mutates (acts outward) → permission-gated. The built-in fs/execute
        # tools are gated by _LISFileToolsMiddleware, not here.
        mutating = {"open_in_browser"} if root else frozenset()
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

    # --- deepagents subagents / memory / skills ---------------------------

    def subagent_specs(self) -> list[dict]:
        """Build deepagents ``SubAgent`` dicts for the subagents defined in the store.

        Each becomes a ``task`` tool subagent_type the main model can delegate to. The
        subagent's model comes from its ``server_id`` (resolved to an EngineChatModel), or
        the chat's ``subagent_server_id``, or the main model (inherit).
        """
        import re

        cfile = chats_store.load()
        if not cfile.subagents:
            return []
        # The default subagent model: the chat's subagent_server_id, or the main server_id
        default_sub_model = self.chat.subagent_server_id or self.chat.server_id
        specs: list[dict] = []
        seen_names: set[str] = set()
        for sub in cfile.subagents:
            name = re.sub(r"[^A-Za-z0-9_]+", "_", sub.name).strip("_").lower() or sub.id[:8]
            # Ensure unique names for the task tool
            base, n = name, 2
            while name in seen_names:
                name = f"{base}_{n}"
                n += 1
            seen_names.add(name)
            model_id = sub.server_id or default_sub_model
            sub_cfg = config_store.find_server_by_id(model_id) if model_id else None
            base_url = sub_cfg.base_url() if sub_cfg else self.chat.base_url
            model_name = sub_cfg.model if sub_cfg else self.chat.model
            # Build a minimal EngineChatModel for the subagent (deepagents resolves it)
            from .deepagents.model import EngineChatModel
            from ..engine.base import EngineConfig, build_engine

            sub_engine = build_engine(EngineConfig(
                base_url=base_url,
                model=model_name,
                max_tokens=self.max_tokens(),
                chat_template_kwargs=capabilities.reasoning_template_kwargs(model_name, None) or None,
                sampling=sampling_of(sub_cfg) or None,
            ))
            sub_model = EngineChatModel(engine=sub_engine, tool_specs=[], tool_names=[], model_name=model_name)
            specs.append({
                "name": name,
                "description": sub.description or sub.name,
                "system_prompt": sub.system_prompt or f"You are {sub.name}, a specialist assistant.",
                "model": sub_model,
            })
        return specs

    def memory_sources(self) -> list[str]:
        """File paths to load as agent memory (AGENTS.md in the working dir)."""
        root = self.fs_root()
        if root:
            import os

            agents_md = os.path.join(root, "AGENTS.md")
            if os.path.isfile(agents_md):
                return [agents_md]
        return []

    def skill_source_paths(self) -> list[str]:
        """Skill directories to make available for on-demand skill loading."""
        from .skills import _root_for, ORIGIN_BUNDLED, ORIGIN_CUSTOM, ORIGIN_BMAD

        paths: list[str] = []
        for origin in (ORIGIN_BUNDLED, ORIGIN_CUSTOM, ORIGIN_BMAD):
            root = _root_for(origin)
            if root and root.is_dir():
                paths.append(str(root))
        return paths
