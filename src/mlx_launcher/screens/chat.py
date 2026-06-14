"""A Claude-app-like chat front-end for talking to a running MLX server.

Sidebar of projects + chats, a streaming transcript that renders Markdown
(headings, lists, and syntax-highlighted code / JSON) live as it arrives, a
collapsible "thinking" block for reasoning models, a server/model picker, a
reasoning toggle, file attachments, a multiline prompt (Enter sends,
Shift+Enter / Ctrl+J for a newline), plus regenerate / edit-last / export."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from contextlib import AsyncExitStack
from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    Switch,
    TextArea,
)

from ..chat import capabilities, fs_tools, mcp_client, prompted_tools, skills, store
from ..chat import tools as chat_tools
from ..chat.blocks import split_blocks
from ..chat.client import (
    DEFAULT_MAX_TOKENS,
    ChatClient,
    build_openai_messages,
    parse_harmony,
    parse_harmony_tool_calls,
    prepend_system,
)
from ..chat.models import Attachment, Chat, ChatMessage, Project
from ..config.models import ServerConfig
from ..server import discovery
from ..server.manager import BinaryNotFound, PortInUse, ServerStatus
from ..widgets.code_block import CodeBlock
from ..widgets.path_input import resolve_path
from ..widgets.safe_content import plain, title_sub
from ..widgets.thinking import ThinkingIndicator
from textual.content import Content


def _candidate_paths(text: str) -> list[str]:
    """Existing absolute file paths in a pasted/dropped string (handles quotes and
    backslash-escaped spaces from terminal drag-and-drop)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = [text]
    paths: list[str] = []
    for tok in tokens:
        p = resolve_path(tok)
        if os.path.isabs(p) and os.path.isfile(p):
            paths.append(p)
    return paths


def _fmt_k(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k".replace(".0k", "k")


def _context_bar_markup(used: int, window: int, width: int = 10) -> str:
    frac = max(0.0, min(1.0, used / window)) if window else 0.0
    filled = round(frac * width)
    pct = int(round(frac * 100))
    color = "#7fb069" if frac < 0.7 else "#d19a66" if frac < 0.9 else "#e06c75"
    bar = f"[{color}]" + "█" * filled + "[/][dim]" + "░" * (width - filled) + "[/]"
    return f"[dim]ctx[/] {bar} [dim]{_fmt_k(used)}/{_fmt_k(window)} · {pct}%[/]"


class PromptArea(TextArea):
    """Multiline prompt: Enter sends, Shift+Enter / Ctrl+J insert a newline.
    Dropping files onto it (a paste of file paths) attaches them instead of typing."""

    class Submitted(Message):
        pass

    class FilesDropped(Message):
        def __init__(self, paths: list[str]) -> None:
            self.paths = paths
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted())
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

    async def _on_paste(self, event: events.Paste) -> None:
        paths = _candidate_paths(event.text)
        if paths:
            event.stop()
            event.prevent_default()
            self.post_message(self.FilesDropped(paths))
            return
        await super()._on_paste(event)


class ConfirmModal(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "no", "No")]

    def __init__(self, prompt: str, confirm_label: str = "OK") -> None:
        super().__init__()
        self._prompt = prompt
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self._prompt)
            with Horizontal(id="modal-buttons"):
                yield Button(self._confirm_label, id="yes", variant="error")
                yield Button("Cancel", id="no", variant="primary")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_no(self) -> None:
        self.dismiss(False)


def _perm_prompt(name: str, args: dict) -> tuple[str, str]:
    """A human summary + detail preview for a file/command permission prompt."""
    if name == "write_file":
        content = args.get("content", "")
        return f"Write file  {args.get('path', '?')}  ({len(content)} chars)", content[:500]
    if name == "edit_file":
        return (f"Edit file  {args.get('path', '?')}",
                f"- {args.get('old_text', '')[:200]}\n+ {args.get('new_text', '')[:200]}")
    if name == "delete_path":
        return f"Delete  {args.get('path', '?')}", ""
    if name == "run_command":
        return "Run command", args.get("command", "")[:500]
    return name, json.dumps(args)[:500]


class PermissionModal(ModalScreen[str]):
    """Ask the user to allow a mutating file/command operation.
    Returns 'once' | 'all' | 'deny'."""

    BINDINGS = [Binding("escape", "deny", "Deny")]

    def __init__(self, summary: str, detail: str = "") -> None:
        super().__init__()
        self._summary = summary
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("[b]Allow this action?[/]")
            yield Label(plain(self._summary), classes="perm-summary")
            if self._detail:
                yield Static(plain(self._detail), classes="perm-detail")
            with Horizontal(id="modal-buttons"):
                yield Button("Approve", id="once", variant="success")
                yield Button("Approve all", id="all", variant="primary")
                yield Button("Deny", id="deny", variant="error")

    @on(Button.Pressed, "#once")
    def _once(self) -> None:
        self.dismiss("once")

    @on(Button.Pressed, "#all")
    def _all(self) -> None:
        self.dismiss("all")

    @on(Button.Pressed, "#deny")
    def _deny(self) -> None:
        self.dismiss("deny")

    def action_deny(self) -> None:
        self.dismiss("deny")


class TextPromptModal(ModalScreen[Optional[str]]):
    """A tiny centered text prompt (used for naming projects)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "") -> None:
        super().__init__()
        self._title = title
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self._title)
            yield Input(value=self._value, id="modal-input")
            with Horizontal(id="modal-buttons"):
                yield Button("OK", id="ok", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted, "#modal-input")
    def _ok(self) -> None:
        self.dismiss(self.query_one("#modal-input", Input).value.strip() or None)

    @on(Button.Pressed, "#cancel")
    def _cancel_btn(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProjectItem(ListItem):
    def __init__(self, project_id: Optional[str], label) -> None:
        super().__init__(Label(label))
        self.project_id = project_id


class ChatItem(ListItem):
    def __init__(self, chat: Chat, subtitle: str) -> None:
        super().__init__(Label(title_sub(chat.title, subtitle or "—")))
        self.chat_id = chat.id


class ChatScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back / Stop"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("ctrl+o", "toggle_attach", "Attach"),
        Binding("ctrl+l", "clear_attach", "Clear files"),
        Binding("ctrl+r", "regenerate", "Regenerate"),
        Binding("ctrl+t", "theme", "Theme"),
        Binding("ctrl+g", "mcp", "MCP servers"),
        Binding("ctrl+k", "skills", "Skills"),
        Binding("ctrl+e", "edit_project", "Edit project"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self, server_id: Optional[str] = None) -> None:
        super().__init__()
        self.data = store.load()
        self.chat: Optional[Chat] = None
        self.project_filter: Optional[str] = None
        self._initial_server = server_id  # when opened from a running server
        self._pending: list[Attachment] = []
        self._cancel = False
        self._streaming = False
        self._auto_approve_fs = False  # "approve all" for file/command ops this session
        self._prompted_servers: set[str] = set()  # servers whose native tools failed → prompted mode

    # --- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="chat-body"):
            with Vertical(id="chat-sidebar"):
                yield Label("Projects", classes="section")
                yield ListView(id="projects")
                yield Label("Chats", classes="section")
                yield ListView(id="chats")
                with Vertical(id="sidebar-buttons"):
                    yield Button("+ Chat", id="new-chat")
                    yield Button("+ Project", id="new-project")
                    yield Button("✕ Delete", id="delete-item")
            with Vertical(id="chat-main"):
                with Horizontal(id="chat-topbar"):
                    yield Static("", id="chat-title")
                    yield Select([], id="skill-select", prompt="skill", allow_blank=True)
                    yield Select([], id="server-select", prompt="server", allow_blank=True)
                yield VerticalScroll(id="transcript")
                with Horizontal(id="chat-toggles"):
                    yield Static("", id="context-bar", classes="ctx-bar")
                    yield Static(classes="actions-spacer")
                    yield Label("plan", id="plan-label", classes="toggle-label")
                    yield Switch(id="plan")
                    yield Label("reason", id="reason-label", classes="toggle-label")
                    yield Switch(id="reasoning")
                    yield Label("web", id="web-label", classes="toggle-label")
                    yield Switch(id="web")
                    yield Label("tools", id="tools-label", classes="toggle-label")
                    yield Switch(id="tools")
                with Horizontal(id="chat-actions"):
                    yield Button("↻ Regenerate", id="regenerate")
                    yield Button("✎ Edit last", id="edit-last")
                    yield Button("⤓ Export", id="export")
                yield Static("", id="attachments", classes="hidden")
                yield Input(id="attach", placeholder="paste a file path, Enter to attach", classes="hidden")
                with Horizontal(id="chat-inputrow"):
                    yield PromptArea(id="prompt", soft_wrap=True)
                    yield Button("+ Attach", id="attach-btn")
                    yield Button("Send", id="send", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", PromptArea).border_title = "Enter to send · Shift+Enter for newline"
        self._refresh_servers()
        self._refresh_skills()
        self._refresh_projects()
        self._refresh_chats()
        if self._initial_server:
            # opened from a running server: continue its most recent chat, else start one
            existing = [c for c in store.chats_in(self.data, None) if c.server_id == self._initial_server]
            if existing:
                self._open_chat(existing[0])
            else:
                self._create_chat(self._server_by_id(self._initial_server))
        elif self.data.chats:
            self._open_chat(store.chats_in(self.data, None)[0])
        else:
            self._create_chat()
        self.query_one("#prompt", PromptArea).focus()

    # --- sidebar ---------------------------------------------------------

    def _refresh_servers(self) -> None:
        options = [(plain(s.name), s.id) for s in self.app.config.servers]  # names may contain markup chars
        # set_options resets the value (→ a spurious Changed); suppress it
        with self.prevent(Select.Changed):
            self.query_one("#server-select", Select).set_options(options)

    def _refresh_skills(self) -> None:
        """Populate the skill picker, marking each skill's origin (★ = custom)."""
        marker = {skills.ORIGIN_CUSTOM: "★ ", skills.ORIGIN_BMAD: "◆ ", skills.ORIGIN_BUNDLED: ""}
        options = [(plain(f"{marker.get(s.origin, '')}{s.name}"), s.id) for s in skills.all_skills()]
        sel = self.query_one("#skill-select", Select)
        keep = self.chat.skill_id if self.chat else None
        ids = {s.id for s in skills.all_skills()}
        # set_options resets the value to blank, firing a spurious Changed that would
        # wipe the chat's skill_id; suppress Changed while we repopulate and restore.
        with self.prevent(Select.Changed):
            sel.set_options(options)
            if keep in ids:
                sel.value = keep

    def _refresh_projects(self) -> None:
        lv = self.query_one("#projects", ListView)
        lv.clear()
        lv.append(ProjectItem(None, plain("All chats")))
        for p in self.data.projects:
            lv.append(ProjectItem(p.id, Content.assemble("▪ ", p.name)))

    def _refresh_chats(self) -> None:
        lv = self.query_one("#chats", ListView)
        lv.clear()
        chats = store.chats_in(self.data, self.project_filter)
        if not chats:
            lv.append(ListItem(Label("[dim]No chats — ctrl+n[/]")))
            return
        for c in chats:
            lv.append(ChatItem(c, self._server_label_for(c)))

    # --- opening / creating ---------------------------------------------

    def _default_server(self) -> Optional[ServerConfig]:
        servers = self.app.config.servers
        if not servers:
            return None
        last = self.app.config.settings.last_used_id
        return next((s for s in servers if s.id == last), servers[0])

    def _server_by_id(self, sid: Optional[str]) -> Optional[ServerConfig]:
        return next((s for s in self.app.config.servers if s.id == sid), None)

    def _client(self) -> ChatClient:
        """A chat client whose token budget respects the profile's --max-tokens
        (if the user set one) and otherwise falls back to a generous default —
        never the server's truncating 512-token default."""
        cfg = self._server_by_id(self.chat.server_id) if (self.chat and self.chat.server_id) else None
        max_tokens = (cfg.max_tokens if cfg and cfg.max_tokens else DEFAULT_MAX_TOKENS)
        return ChatClient(self.chat.base_url, self.chat.model, max_tokens=max_tokens)

    def _server_label_for(self, chat: Optional[Chat]) -> str:
        """The user-given profile name for a chat's server (falls back to the model)."""
        cfg = self._server_by_id(chat.server_id) if chat and chat.server_id else None
        if cfg and cfg.name:
            return cfg.name
        return (chat.model if chat else "") or "model"

    def _create_chat(self, server: Optional[ServerConfig] = None) -> None:
        cfg = server or self._default_server()
        chat = Chat(project_id=self.project_filter)
        if cfg is not None:
            chat.server_id = cfg.id
            chat.base_url = cfg.base_url()
            chat.model = cfg.model
            chat.reasoning = capabilities.supports_reasoning(cfg.model)
        store.upsert_chat(self.data, chat)
        store.save(self.data)
        self._refresh_chats()
        self._open_chat(chat)

    def _open_chat(self, chat: Chat) -> None:
        self.chat = chat
        self._pending = []
        self._refresh_attachments()
        select = self.query_one("#server-select", Select)
        ids = [s.id for s in self.app.config.servers]
        select.value = chat.server_id if chat.server_id in ids else Select.NULL
        skill_sel = self.query_one("#skill-select", Select)
        skill_ids = {s.id for s in skills.all_skills()}
        skill_sel.value = chat.skill_id if chat.skill_id in skill_ids else Select.NULL
        self._sync_reasoning_switch()
        self.query_one("#web", Switch).value = bool(self.chat and self.chat.web_search)
        self.query_one("#tools", Switch).value = bool(self.chat and self.chat.tools)
        self.query_one("#plan", Switch).value = bool(self.chat and self.chat.plan_mode)
        self._update_topbar()
        self._update_context_bar()
        self._render_transcript()
        self.query_one("#prompt", PromptArea).focus()

    def _sync_reasoning_switch(self) -> None:
        sw = self.query_one("#reasoning", Switch)
        supported = bool(self.chat and capabilities.supports_reasoning(self.chat.model))
        sw.disabled = not supported
        sw.value = bool(self.chat and self.chat.reasoning and supported)

    def _update_topbar(self) -> None:
        if not self.chat:
            return
        name = "(no server)" if (not self.chat.server_id and not self.chat.model) else self._server_label_for(self.chat)
        dim = name
        if capabilities.supports_vision(self.chat.model):
            dim += " · ◉ vision"
        skill = skills.get_skill(self.chat.skill_id)
        if skill:
            dim += f" · ▸ {'★ ' if skill.is_custom else ''}{skill.name}"
        proj = self._current_project()
        if proj and proj.working_dir:
            dim += " · ▣ " + proj.working_dir.replace(os.path.expanduser("~"), "~")
        parts: list = [(self.chat.title, "bold"), "   ", (dim, "dim")]
        if self.chat.plan_mode:
            parts += ["   ", ("● PLAN MODE", "bold #d19a66")]
        self.query_one("#chat-title", Static).update(Content.assemble(*parts))

    def _update_context_bar(self) -> None:
        """Show how much of the model's context window the conversation uses.
        Hidden when the context window can't be determined ('if available')."""
        bar = self.query_one("#context-bar", Static)
        window = capabilities.context_window(self.chat.model) if self.chat else None
        if not self.chat or not window:
            bar.update("")
            return
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        used = capabilities.estimate_prompt_tokens(messages)
        fs = self._fs_root()
        if fs:
            used += capabilities.approx_tokens(fs_tools.system_note(fs))
        bar.update(_context_bar_markup(used, window))

    # --- transcript / message widgets -----------------------------------

    def _assemble_row(self, label: str, role_class: str, body_widgets: list, *, right: bool = False) -> Horizontal:
        role = plain(label) if isinstance(label, str) else label  # names may contain markup chars
        bubble = Vertical(Label(role, classes="msg-role"), *body_widgets, classes=f"msg {role_class}")
        spacer = Static(classes="msg-spacer")
        children = (spacer, bubble) if right else (bubble, spacer)
        return Horizontal(*children, classes="msg-row")

    def _bubble(self, role_label: str, role_class: str, body, *, right: bool = False) -> tuple[Horizontal, Static]:
        body_widget = Static(body, classes="msg-body")
        return self._assemble_row(role_label, role_class, [body_widget], right=right), body_widget

    async def _mount_thinking(self, transcript) -> Horizontal:
        """Mount an animated 'thinking' bubble at the bottom of the transcript and
        return its row so the caller can remove it once a reply/tool result lands."""
        row = self._assemble_row(
            self._server_label_for(self.chat), "msg-assistant",
            [ThinkingIndicator(classes="msg-body thinking-indicator")],
        )
        await transcript.mount(row)
        self._scroll_end()
        return row

    def _assistant_body_widgets(self, text: str) -> list:
        """Prose runs as Markdown + each fenced code block as a copyable CodeBlock."""
        widgets: list = []
        for block in split_blocks(text):
            if block[0] == "code":
                widgets.append(CodeBlock(block[2], block[1]))
            elif block[1].strip():
                widgets.append(Static(RichMarkdown(block[1]), classes="msg-body"))
        return widgets or [Static("[dim](no content)[/]", classes="msg-body")]

    @staticmethod
    def _stats_text(tps: float, tokens: int, elapsed: float) -> str:
        return f"↯ {tps:.1f} tok/s · {tokens} tok · {elapsed:.1f}s"

    def _message_widget(self, m: ChatMessage) -> Horizontal:
        if m.role == "assistant":
            widgets = self._assistant_body_widgets(m.text)
            if m.tps:
                widgets.append(Static(self._stats_text(m.tps, m.n_tokens or 0, m.elapsed or 0.0), classes="msg-stats"))
            return self._assemble_row(self._server_label_for(self.chat), "msg-assistant", widgets)
        chips = "  ".join(f"{a.name or a.path} ({a.kind})" for a in m.attachments) if m.attachments else ""
        if m.text and chips:
            body = Content.assemble(m.text, "\n", (chips, "dim"))
        elif m.text:
            body = plain(m.text)
        elif chips:
            body = Content.assemble((chips, "dim"))
        else:
            body = plain("(empty)")
        row, _ = self._bubble("You", "msg-user", body, right=True)
        return row

    def _think_widget(self, text: str) -> Horizontal:
        row, _ = self._bubble("◌ thinking", "msg-think", plain(text))
        return row

    def _render_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.remove_children()
        if not self.chat:
            return
        for m in self.chat.messages:
            if m.role == "assistant" and m.reasoning:
                transcript.mount(self._think_widget(m.reasoning))
            transcript.mount(self._message_widget(m))
        self.call_after_refresh(self._scroll_end)

    def _scroll_end(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
        except Exception:  # noqa: BLE001 — transcript may be gone if the screen was dismissed
            pass

    # --- events ----------------------------------------------------------

    @on(ListView.Selected, "#projects")
    def _project_selected(self, event: ListView.Selected) -> None:
        self.project_filter = getattr(event.item, "project_id", None)
        self._refresh_chats()

    @on(ListView.Selected, "#chats")
    def _chat_selected(self, event: ListView.Selected) -> None:
        cid = getattr(event.item, "chat_id", None)
        chat = store.get_chat(self.data, cid) if cid else None
        if chat:
            self._open_chat(chat)

    @on(Select.Changed, "#server-select")
    def _server_changed(self, event: Select.Changed) -> None:
        # event.value == current → programmatic sync, re-select, or a revert: ignore.
        if not self.chat or event.value is Select.NULL or event.value == self.chat.server_id:
            return
        cfg = self._server_by_id(event.value)
        if cfg is None:
            return
        if self._streaming:
            self.notify("Stop the current response before switching models", severity="warning")
            self._revert_server_select()
            return
        self.run_worker(self._confirm_and_switch(cfg), exclusive=True, group="server-switch")

    @on(Select.Changed, "#skill-select")
    def _skill_changed(self, event: Select.Changed) -> None:
        if not self.chat:
            return
        new = None if event.value is Select.NULL else event.value
        if new == self.chat.skill_id:
            return
        self.chat.skill_id = new
        self._update_topbar()
        self._persist()
        skill = skills.get_skill(new)
        self.notify(f"Skill: {skill.name}" if skill else "Skill cleared")

    def _revert_server_select(self) -> None:
        sel = self.query_one("#server-select", Select)
        sel.value = self.chat.server_id if (self.chat and self.chat.server_id) else Select.NULL

    def _port_blockers(self, cfg: ServerConfig) -> list:
        """Running servers that must be stopped before `cfg` can bind its port:
        this chat's previous server PLUS anything else already on that host:port.
        All profiles default to :8080, so the occupant is often a *different*
        profile than the chat's previous one — stopping only `old` left the port
        taken and the new model failed with "already in use"."""
        old = self._server_by_id(self.chat.server_id) if self.chat.server_id else None
        by_id: dict[str, object] = {}
        old_mgr = self.app.get_manager(old.id) if old else None
        if old_mgr and old_mgr.is_running:
            by_id[old.id] = old_mgr
        for m in self.app.running_managers():
            if m.cfg.host == cfg.host and m.cfg.port == cfg.port:
                by_id[m.cfg.id] = m
        by_id.pop(cfg.id, None)  # never stop the model we're about to (re)use
        return list(by_id.values())

    async def _confirm_and_switch(self, cfg: ServerConfig) -> None:
        blockers = self._port_blockers(cfg)
        if blockers:
            names = ", ".join(m.cfg.name for m in blockers)
            unload = f"This unloads {names}"
        else:
            unload = "This starts the selected model"
        prompt = (
            f"Switch this chat to '{cfg.name}'?\n\n"
            f"{unload} and loads '{cfg.name}' on the server — the current model's "
            "loaded context (weights + KV cache) will be lost."
        )
        ok = await self.app.push_screen_wait(ConfirmModal(prompt, confirm_label="Switch & reload"))
        if not ok:
            self._revert_server_select()
            return

        # 1) unload every server holding the target port (not just the chat's old one)
        for mgr in blockers:
            self.notify(f"Unloading {mgr.cfg.name} …")
            await mgr.stop()

        # 2) repoint the chat at the new profile
        self.chat.server_id = cfg.id
        self.chat.base_url = cfg.base_url()
        self.chat.model = cfg.model
        self.chat.reasoning = self.chat.reasoning and capabilities.supports_reasoning(cfg.model)
        self.app.config.settings.last_used_id = cfg.id
        self.app.save_config()
        self._sync_reasoning_switch()
        self._update_topbar()
        self._persist()
        self._refresh_chats()

        # 3) load the new model
        await self._load_server(cfg)

    async def _load_server(self, cfg: ServerConfig) -> None:
        mgr = self.app.get_manager(cfg.id)
        if mgr is not None and mgr.is_running:
            self.notify(f"{cfg.name} is already running")
            return
        # A server we just stopped may not have released the port yet; give the OS
        # a moment so start()'s bind-check doesn't race and report "already in use".
        for _ in range(20):  # ~2s
            if discovery.is_port_free(cfg.host, cfg.port):
                break
            await asyncio.sleep(0.1)
        mgr = self.app.create_manager(cfg)
        self.notify(f"Loading {cfg.name} — model load can take a while …")
        try:
            await mgr.start()
        except BinaryNotFound:
            self.notify(
                f"{discovery.binary_name(cfg.engine)} not found — press p on the dashboard to install",
                severity="error",
            )
            return
        except PortInUse as exc:
            self.notify(str(exc), severity="error")
            return
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Failed to start: {exc}", severity="error")
            return
        # surface readiness/failure (model load happens after spawn). Poll status,
        # not is_running — the process can exit a beat before ERROR is recorded.
        for _ in range(900):  # ~270s budget for a slow model load
            status = mgr.status
            if status is ServerStatus.READY:
                self.notify(f"{cfg.name} ready")
                return
            if status in (ServerStatus.ERROR, ServerStatus.STOPPED):
                self.notify(mgr.status_message or "model failed to load", severity="error")
                return
            await asyncio.sleep(0.3)
        self.notify(f"{cfg.name}: still loading — open it from the dashboard to watch", severity="warning")

    @on(Switch.Changed, "#reasoning")
    def _reasoning_changed(self, event: Switch.Changed) -> None:
        if self.chat and not self.query_one("#reasoning", Switch).disabled:
            self.chat.reasoning = event.value
            self._persist()

    @on(Switch.Changed, "#web")
    def _web_changed(self, event: Switch.Changed) -> None:
        if self.chat:
            self.chat.web_search = event.value
            self._persist()

    @on(Switch.Changed, "#tools")
    def _tools_changed(self, event: Switch.Changed) -> None:
        if self.chat:
            self.chat.tools = event.value
            self._persist()

    @on(Switch.Changed, "#plan")
    def _plan_changed(self, event: Switch.Changed) -> None:
        if self.chat:
            self.chat.plan_mode = event.value
            self._update_topbar()
            self._persist()
            self.notify("Plan mode on — I'll propose a plan to approve, not make changes"
                        if event.value else "Plan mode off")

    @on(Button.Pressed, "#new-chat")
    def _new_chat_btn(self) -> None:
        self.action_new_chat()

    @on(Button.Pressed, "#new-project")
    def _new_project_btn(self) -> None:
        self._new_project()

    @on(Button.Pressed, "#delete-item")
    def _delete_btn(self) -> None:
        self.action_delete()

    @on(PromptArea.FilesDropped)
    def _files_dropped(self, event: PromptArea.FilesDropped) -> None:
        self._attach_paths(event.paths)

    def on_paste(self, event) -> None:
        # files dropped while a non-prompt widget is focused bubble up here
        paths = _candidate_paths(getattr(event, "text", ""))
        if paths:
            event.stop()
            self._attach_paths(paths)

    def _attach_paths(self, paths: list[str]) -> None:
        added = 0
        for p in paths:
            kind = capabilities.classify(p)
            if kind == "image" and self.chat and not capabilities.supports_vision(self.chat.model):
                self.notify("Model may not support images — attaching anyway.", severity="warning")
            self._pending.append(Attachment(path=p, name=os.path.basename(p), kind=kind))
            added += 1
        if added:
            self._refresh_attachments()
            self.notify(f"Attached {added} file" + ("s" if added != 1 else ""))

    @on(Button.Pressed, "#attach-btn")
    def _attach_btn(self) -> None:
        self.action_toggle_attach()

    @on(Button.Pressed, "#regenerate")
    def _regen_btn(self) -> None:
        self.action_regenerate()

    @on(Button.Pressed, "#edit-last")
    def _edit_btn(self) -> None:
        self.action_edit_last()

    @on(Button.Pressed, "#export")
    def _export_btn(self) -> None:
        self.action_export()

    @on(Input.Submitted, "#attach")
    def _attach_submit(self, event: Input.Submitted) -> None:
        path = resolve_path(event.value)
        field = self.query_one("#attach", Input)
        if not path:
            return
        if not os.path.exists(path):
            self.notify(f"Not found: {path}", severity="error")
            return
        kind = capabilities.classify(path)
        if kind == "image" and self.chat and not capabilities.supports_vision(self.chat.model):
            self.notify("This model may not support images — sending anyway.", severity="warning")
        self._pending.append(Attachment(path=path, name=os.path.basename(path), kind=kind))
        field.value = ""
        field.add_class("hidden")
        self._refresh_attachments()
        self.query_one("#prompt", PromptArea).focus()

    @on(Button.Pressed, "#send")
    def _send_pressed(self) -> None:
        if self._streaming:
            self.action_stop()
        else:
            self.action_send()

    @on(PromptArea.Submitted)
    def _prompt_submitted(self) -> None:
        self.action_send()

    def _refresh_attachments(self) -> None:
        bar = self.query_one("#attachments", Static)
        if self._pending:
            parts: list = []
            for a in self._pending:
                if parts:
                    parts.append("   ")
                parts.append(a.name)
                parts.append((f" ({a.kind})", "dim"))
            parts.append(("   · ctrl+l clears", "dim"))
            bar.update(Content.assemble(*parts))
            bar.remove_class("hidden")
        else:
            bar.update("")
            bar.add_class("hidden")

    def _persist(self) -> None:
        if self.chat:
            store.upsert_chat(self.data, self.chat)
            store.save(self.data)
        try:
            self._update_context_bar()
        except Exception:  # noqa: BLE001 — bar is best-effort
            pass

    def _set_generating(self, on: bool) -> None:
        self._streaming = on
        try:
            send = self.query_one("#send", Button)
            send.label = "■ Stop" if on else "Send"
            send.variant = "error" if on else "primary"
        except Exception:  # noqa: BLE001
            pass

    # --- actions ---------------------------------------------------------

    def action_back(self) -> None:
        if self._streaming:
            self._cancel = True
            self.notify("Stopped")
            return
        self.app.pop_screen()

    def action_stop(self) -> None:
        if self._streaming:
            self._cancel = True

    def action_new_chat(self) -> None:
        self._create_chat()

    def action_theme(self) -> None:
        from .theme_picker import ThemeScreen

        self.app.push_screen(ThemeScreen())

    def action_mcp(self) -> None:
        from .mcp_manager import McpManagerScreen

        self.app.push_screen(McpManagerScreen())

    def action_skills(self) -> None:
        from .skills_manager import SkillsManagerScreen

        self.app.push_screen(SkillsManagerScreen())

    def on_screen_resume(self) -> None:
        # returning from a manager/editor: pick up new skills and project edits
        self._refresh_skills()
        self._refresh_projects()
        self._refresh_chats()
        self._update_topbar()

    def action_toggle_attach(self) -> None:
        field = self.query_one("#attach", Input)
        if field.has_class("hidden"):
            field.remove_class("hidden")
            field.focus()
        else:
            field.add_class("hidden")

    def action_clear_attach(self) -> None:
        self._pending = []
        self._refresh_attachments()

    def action_delete(self) -> None:
        self._delete_flow()

    def _resolve_delete(self):
        """What the delete action targets: a focused list's highlighted item, else
        a highlighted project, else the open chat."""
        chats_lv = self.query_one("#chats", ListView)
        projects_lv = self.query_one("#projects", ListView)
        if projects_lv.has_focus:
            pid = getattr(projects_lv.highlighted_child, "project_id", None)
            if pid:
                proj = store.get_project(self.data, pid)
                return ("project", pid, proj.name if proj else "project")
            return None
        if chats_lv.has_focus:
            cid = getattr(chats_lv.highlighted_child, "chat_id", None)
            if cid:
                c = store.get_chat(self.data, cid)
                return ("chat", cid, c.title if c else "chat")
            return None
        pid = getattr(projects_lv.highlighted_child, "project_id", None)
        if pid:
            proj = store.get_project(self.data, pid)
            return ("project", pid, proj.name if proj else "project")
        if self.chat is not None:
            return ("chat", self.chat.id, self.chat.title)
        return None

    @work
    async def _delete_flow(self) -> None:
        target = self._resolve_delete()
        if target is None:
            self.notify("Nothing selected to delete", severity="warning")
            return
        kind, ident, label = target
        ok = await self.app.push_screen_wait(
            ConfirmModal(f'Delete this {kind}?\n"{label}"', confirm_label="Delete")
        )
        if not ok:
            return
        if kind == "project":
            store.delete_project(self.data, ident)
            store.save(self.data)
            self.project_filter = None
            self._refresh_projects()
            self._refresh_chats()
            self.notify("Project deleted")
        else:
            store.delete_chat(self.data, ident)
            store.save(self.data)
            self._refresh_chats()
            remaining = store.chats_in(self.data, self.project_filter)
            if remaining:
                self._open_chat(remaining[0])
            else:
                self._create_chat()
            self.notify("Chat deleted")

    def action_regenerate(self) -> None:
        if self._streaming:
            self.notify("Still generating — Esc to stop", severity="warning")
            return
        if not self.chat or not self.chat.messages:
            return
        if self.chat.messages[-1].role != "assistant":
            self.notify("Nothing to regenerate", severity="warning")
            return
        self.chat.messages.pop()
        self._persist()
        self._render_transcript()
        self._generate()

    def action_edit_last(self) -> None:
        if self._streaming or not self.chat:
            return
        idx = next((i for i in range(len(self.chat.messages) - 1, -1, -1)
                    if self.chat.messages[i].role == "user"), None)
        if idx is None:
            self.notify("No message to edit", severity="warning")
            return
        self.query_one("#prompt", PromptArea).load_text(self.chat.messages[idx].text)
        self.chat.messages = self.chat.messages[:idx]
        self._persist()
        self._render_transcript()
        self.query_one("#prompt", PromptArea).focus()
        self.notify("Editing — change it and send again")

    def action_export(self) -> None:
        if not self.chat or not self.chat.messages:
            self.notify("Nothing to export", severity="warning")
            return
        markdown = self._export_markdown()
        safe = re.sub(r"[^\w.-]+", "_", self.chat.title).strip("_") or "chat"
        path = os.path.expanduser(f"~/{safe}.md")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown)
        except OSError as exc:
            self.notify(f"Export failed: {exc}", severity="error")
            return
        self.app.copy_text(markdown)
        self.notify(f"Exported to {path} (also copied to clipboard)")

    def _export_markdown(self) -> str:
        assert self.chat is not None
        lines = [f"# {self.chat.title}", "", f"*model: {self.chat.model or 'unknown'}*", ""]
        for m in self.chat.messages:
            who = "You" if m.role == "user" else (self.chat.model or "assistant")
            lines.append(f"## {who}")
            lines.append("")
            for a in m.attachments:
                lines.append(f"- ⊕ `{a.name or a.path}` ({a.kind})")
            if m.role == "assistant" and m.reasoning:
                lines.append("> **thinking**")
                for ln in m.reasoning.splitlines():
                    lines.append(f"> {ln}")
                lines.append("")
            lines.append(m.text)
            lines.append("")
        return "\n".join(lines)

    def _new_project(self) -> None:
        from .project_editor import ProjectEditorScreen

        self.app.push_screen(ProjectEditorScreen(self.data))

    def action_edit_project(self) -> None:
        item = self.query_one("#projects", ListView).highlighted_child
        pid = getattr(item, "project_id", None)
        proj = store.get_project(self.data, pid) if pid else None
        if proj is None:
            self.notify("Select a project in the sidebar to edit it", severity="warning")
            return
        from .project_editor import ProjectEditorScreen

        self.app.push_screen(ProjectEditorScreen(self.data, proj))

    def action_send(self) -> None:
        if self._streaming:
            self.notify("Still generating — Esc to stop", severity="warning")
            return
        if not self.chat:
            return
        text = self.query_one("#prompt", PromptArea).text.strip()
        if not text and not self._pending:
            return
        if not self.chat.base_url or not self.chat.model:
            self.notify("Pick a server first (or create one in the dashboard).", severity="warning")
            return

        msg = ChatMessage(role="user", text=text, attachments=list(self._pending))
        self.chat.messages.append(msg)
        if self.chat.title == "New chat" and text:
            self.chat.title = text[:40]
        self.query_one("#prompt", PromptArea).load_text("")
        self._pending = []
        self._refresh_attachments()
        self._persist()
        self._refresh_chats()

        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(self._message_widget(msg))
        self.call_after_refresh(self._scroll_end)
        self._generate()

    def _fs_root(self) -> Optional[str]:
        """The project's working directory if it exists on disk, else None."""
        proj = self._current_project()
        if proj and proj.working_dir:
            path = os.path.expanduser(proj.working_dir)
            if os.path.isdir(path):
                return path
        return None

    @work
    async def _generate(self) -> None:
        if self.chat and (self.chat.web_search or self.chat.tools or self._fs_root()):
            await self._generate_tools()
        else:
            await self._generate_stream()

    async def _bridge_chat(self, client: ChatClient, messages: list, specs) -> Optional[dict]:
        """A non-streaming completion that aborts within ~0.1s when the user hits
        Stop (the raw call blocks for the whole response, so we poll _cancel and
        cancel the request). Returns the response, or None if cancelled; bridge
        errors propagate to the caller."""
        task = asyncio.ensure_future(client.bridge.chat(messages, tools=specs or None))
        while not task.done():
            if self._cancel:
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 — swallow CancelledError / late errors
                    pass
                return None
            await asyncio.sleep(0.1)
        return task.result()  # re-raises a bridge error for the caller's try/except

    async def _generate_stream(self) -> None:
        assert self.chat is not None
        transcript = self.query_one("#transcript", VerticalScroll)
        # The answer bubble animates ("Thinking…") until the first content token,
        # then we stop the spinner and stream the reply into the same widget.
        assistant_body = ThinkingIndicator(classes="msg-body thinking-indicator")
        assistant_box = self._assemble_row(self._server_label_for(self.chat), "msg-assistant", [assistant_body])
        await transcript.mount(assistant_box)
        think_body: Optional[Static] = None

        client = self._client()
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        reason_acc: list[str] = []
        content_acc: list[str] = []
        last_render = 0
        tokens = 0
        t_first: Optional[float] = None
        errored = False
        self._cancel = False
        self._set_generating(True)
        try:
            async for kind, chunk in client.stream(messages, cancel=lambda: self._cancel):
                if kind in ("reason", "content"):
                    if t_first is None:
                        t_first = time.monotonic()
                    tokens += 1
                if kind == "reason":
                    reason_acc.append(chunk)
                    if self.chat.reasoning:
                        if think_body is None:
                            think_box, think_body = self._bubble("◌ thinking", "msg-think", "")
                            await transcript.mount(think_box, before=assistant_box)
                        think_body.update(plain("".join(reason_acc)))
                elif kind == "content":
                    if not content_acc:  # first real token → stop the spinner, take over
                        assistant_body.stop()
                    content_acc.append(chunk)
                    joined = "".join(content_acc)
                    # Render Markdown live, but throttle to newline / ~24-char deltas.
                    if last_render == 0 or "\n" in chunk or len(joined) - last_render >= 24:
                        assistant_body.update(RichMarkdown(joined))
                        last_render = len(joined)
                self._scroll_end()
        except Exception as exc:  # noqa: BLE001
            errored = True
            try:
                assistant_body.stop()
                assistant_body.update(f"[#e06c75]▲ {escape(str(exc))}[/]")
            except Exception:  # noqa: BLE001
                pass

        final = "".join(content_acc)
        elapsed = (time.monotonic() - t_first) if t_first else 0.0
        tps = (tokens / elapsed) if elapsed > 0 else 0.0
        if not errored:
            # re-render into prose + copyable code blocks, with a tok/s footer
            try:
                bubble = assistant_body.parent
                await assistant_body.remove()
                for widget in self._assistant_body_widgets(final):
                    await bubble.mount(widget)
                if tps > 0:
                    await bubble.mount(Static(self._stats_text(tps, tokens, elapsed), classes="msg-stats"))
            except Exception:  # noqa: BLE001
                pass
        self.chat.messages.append(ChatMessage(
            role="assistant",
            text=final,
            reasoning="".join(reason_acc),
            tps=round(tps, 1) if tps > 0 else None,
            n_tokens=tokens or None,
            elapsed=round(elapsed, 1) if elapsed > 0 else None,
        ))
        self.chat.updated = self.chat.messages[-1].ts
        self._persist()
        self._set_generating(False)
        try:
            self._refresh_chats()
        except Exception:  # noqa: BLE001
            pass
        self._scroll_end()

    async def _generate_tools(self) -> None:
        """Function-calling loop: offer web_search + connected MCP tools, execute the
        model's tool calls, then render the final answer."""
        assert self.chat is not None
        transcript = self.query_one("#transcript", VerticalScroll)
        client = self._client()
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        fs_root = self._fs_root()
        if fs_root:
            # ONE leading system message only — see prepend_system (templates 500 on two)
            prepend_system(messages, fs_tools.system_note(fs_root))
        self._cancel = False
        self._set_generating(True)
        t0 = time.monotonic()
        n_calls = 0
        final_text = ""
        thinking = None  # the live "thinking…" bubble between turns (cleaned up below)
        max_iters = 24 if fs_root else 8
        try:
            async with AsyncExitStack() as stack:
                specs = []
                if self.chat.web_search:
                    specs.append(chat_tools.web_search_spec())
                if fs_root:
                    specs += fs_tools.fs_specs()
                sessions, router = {}, {}
                if self.chat.tools:
                    servers = store.load().mcp_servers  # pick up edits from the MCP manager
                    sessions, mcp_specs, router = await mcp_client.open_sessions(
                        stack,
                        servers,
                        on_error=lambda name, err: self.notify(f"MCP {name}: {err}", severity="warning"),
                    )
                    specs += mcp_specs
                # native function-calling, falling back to prompted tools (text protocol)
                # for any model whose template rejects the `tools` param.
                prompted = (self.chat.server_id or "") in self._prompted_servers
                if prompted and specs:
                    prepend_system(messages, prompted_tools.tool_instructions(specs))
                for _ in range(max_iters):
                    if self._cancel:
                        break
                    thinking = await self._mount_thinking(transcript)
                    if prompted:
                        data = await self._bridge_chat(client, messages, None)
                        await thinking.remove()
                        thinking = None
                        if data is None:  # stopped
                            break
                        msg = (data.get("choices") or [{}])[0].get("message") or {}
                        content, _r = parse_harmony(msg.get("content") or "")
                        calls = prompted_tools.parse_tool_calls(content) if specs else []
                        if not calls:
                            final_text = prompted_tools.strip_tool_calls(content) or content
                            messages.append({"role": "assistant", "content": content})
                            break
                        messages.append({"role": "assistant", "content": content})
                        for call in calls:
                            n_calls += 1
                            result = await self._exec_tool(call["name"], call["arguments"], sessions, router, transcript, fs_root)
                            messages.append({"role": "user", "content": prompted_tools.tool_response(call["name"], result[:8000])})
                        self._scroll_end()
                        continue
                    try:
                        data = await self._bridge_chat(client, messages, specs)
                    except Exception as exc:  # native tools rejected → switch this server to prompted
                        await thinking.remove()
                        thinking = None
                        if specs and not prompted:
                            prompted = True
                            self._prompted_servers.add(self.chat.server_id or "")
                            prepend_system(messages, prompted_tools.tool_instructions(specs))
                            self.notify("Native tool-calling failed — using prompted tools for this model.",
                                        severity="warning", timeout=8)
                            continue
                        raise
                    await thinking.remove()
                    thinking = None
                    if data is None:  # stopped
                        break
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    raw = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls") or []
                    content, reason = parse_harmony(raw)
                    # gpt-oss puts calls in the Harmony commentary channel; mlx_lm
                    # returns them as text, not native tool_calls — recover them.
                    harmony_calls = parse_harmony_tool_calls(raw) if not tool_calls else []
                    if not tool_calls and not harmony_calls:
                        messages.append({"role": "assistant", "content": content})
                        final_text = content
                        break
                    if harmony_calls:
                        # Echo a CLEAN assistant turn (raw Harmony tokens would nest
                        # channels and confuse the template), then feed each result
                        # back as a user message — the round-trip gpt-oss renders.
                        messages.append({"role": "assistant", "content": reason or content or "Calling tools."})
                        for call in harmony_calls:
                            n_calls += 1
                            result = await self._exec_tool(call["name"], call["arguments"], sessions, router, transcript, fs_root)
                            messages.append({"role": "user", "content": prompted_tools.tool_response(call["name"], result[:8000])})
                        self._scroll_end()
                        continue
                    messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})
                    for call in tool_calls:
                        n_calls += 1
                        fn = call.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        result = await self._exec_tool(fn.get("name", ""), args, sessions, router, transcript, fs_root)
                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result[:8000]})
                    self._scroll_end()
        except Exception as exc:  # noqa: BLE001
            final_text = f"▲ {exc}"
        if thinking is not None:  # never leave a spinner spinning
            try:
                await thinking.remove()
            except Exception:  # noqa: BLE001
                pass

        elapsed = time.monotonic() - t0
        widgets = self._assistant_body_widgets(final_text or ("(stopped)" if self._cancel else "(no answer)"))
        plural = "" if n_calls == 1 else "s"
        widgets.append(Static(f"▸ {n_calls} tool call{plural} · {elapsed:.1f}s", classes="msg-stats"))
        await transcript.mount(self._assemble_row(self._server_label_for(self.chat), "msg-assistant", widgets))
        self.chat.messages.append(
            ChatMessage(role="assistant", text=final_text, n_tokens=n_calls or None, elapsed=round(elapsed, 1))
        )
        self.chat.updated = self.chat.messages[-1].ts
        self._persist()
        self._set_generating(False)
        try:
            self._refresh_chats()
        except Exception:  # noqa: BLE001
            pass
        self._scroll_end()

    async def _exec_tool(self, name: str, args: dict, sessions: dict, router: dict, transcript, fs_root: Optional[str] = None) -> str:
        row, body = self._bubble("▸ tool", "msg-tool",
                                 Content.assemble((name, "bold"), "  ", (json.dumps(args)[:80], "dim")))
        await transcript.mount(row)
        self._scroll_end()
        try:
            if name == "web_search":
                result = await chat_tools.run_web_search(args.get("query", ""), args.get("max_results", 6))
            elif fs_root and name in fs_tools.FS_TOOL_NAMES:
                if name in fs_tools.MUTATING_TOOLS and not self._auto_approve_fs:
                    summary, detail = _perm_prompt(name, args)
                    decision = await self.app.push_screen_wait(PermissionModal(summary, detail))
                    if decision == "all":
                        self._auto_approve_fs = True
                    elif decision != "once":
                        body.update(Content.assemble((name, "bold"), "\n", ("✕ denied by the user", "#e06c75")))
                        return "The user DENIED this action. Do not retry it; ask how to proceed."
                result = await fs_tools.run_fs_tool(fs_root, name, args)
            elif name in router:
                result = await mcp_client.call_mcp(sessions, router, name, args)
            else:
                result = f"Unknown tool: {name}"
        except Exception as exc:  # noqa: BLE001
            result = f"tool error: {exc}"
        preview = result if len(result) <= 500 else result[:500] + " …"
        body.update(Content.assemble((name, "bold"), "\n", (preview, "dim")))
        return result

    def _current_project(self) -> Optional[Project]:
        if self.chat and self.chat.project_id:
            return store.get_project(self.data, self.chat.project_id)
        return None

    def _skill_instructions(self) -> Optional[str]:
        return skills.instructions_for(self.chat.skill_id) if self.chat else None
