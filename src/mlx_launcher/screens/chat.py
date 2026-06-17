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
    Markdown,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from ..chat import capabilities, fs_tools, knowledge, mcp_client, skills, store, voice
from ..chat import tools as chat_tools
from ..chat.blocks import linkify_urls, split_blocks
from ..core.tools.phrasing import _perm_prompt, _tool_phrase  # moved to core; re-exported for tests
from ..core import events as core_events
from ..core.agent import AgentRunner, RunPolicy, ToolOutcome, ToolSet
from ..engine.openai import OpenAIEngine
# build_openai_messages + scaled_max_tokens stay for the in-process context bar + subagent engine.
from ..chat.client import build_openai_messages, scaled_max_tokens
from ..models import Attachment, Chat, ChatMessage, Project, Subagent
from ..models import ServerConfig
from ..widgets.code_block import CodeBlock
from ..widgets.path_input import resolve_path
from ..widgets.safe_content import plain, title_sub
from ..widgets.thinking import ThinkingIndicator
from ..widgets.toggle_chip import ToggleChip
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
    Dropping files onto it (a paste of file paths) attaches them instead of typing.
    When the "/" command menu is open it captures arrow/Enter/Tab/Esc to drive it."""

    class Submitted(Message):
        pass

    class SlashSelected(Message):  # Enter while the "/" menu is open → run the highlighted command
        pass

    class SlashComplete(Message):  # Tab while the "/" menu is open → fill in the highlighted command
        pass

    class FilesDropped(Message):
        def __init__(self, paths: list[str]) -> None:
            self.paths = paths
            super().__init__()

    def _slash_menu(self) -> Optional[OptionList]:
        """The sibling slash-command list, if it's currently shown (else None)."""
        try:
            menu = self.screen.query_one("#slash-suggest", OptionList)
        except Exception:  # noqa: BLE001 — not mounted / screen torn down
            return None
        return menu if menu.display else None

    async def _on_key(self, event: events.Key) -> None:
        menu = self._slash_menu()
        if menu is not None:  # the "/" command menu owns these keys while it's open
            if event.key in ("down", "ctrl+n"):
                menu.action_cursor_down()
                event.stop(); event.prevent_default(); return
            if event.key in ("up", "ctrl+p"):
                menu.action_cursor_up()
                event.stop(); event.prevent_default(); return
            if event.key == "escape":
                menu.display = False
                event.stop(); event.prevent_default(); return
            if event.key == "enter":
                self.post_message(self.SlashSelected())
                event.stop(); event.prevent_default(); return
            if event.key == "tab":
                self.post_message(self.SlashComplete())
                event.stop(); event.prevent_default(); return
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


class ConnectorsModal(ModalScreen[None]):
    """Pick which configured MCP connectors are enabled for tool use. Each server
    is a chip — lit = enabled. Toggling persists immediately; the tools loop reads
    `enabled` per turn (mcp_client.open_sessions), so changes take effect at once."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, data) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("[b]Connectors[/]  ·  MCP servers for tool use")
            if not self._data.mcp_servers:
                yield Label(plain("No connectors yet — add MCP servers with Ctrl+G."), classes="hint")
            with VerticalScroll(id="connector-list"):
                for srv in self._data.mcp_servers:
                    yield ToggleChip(plain(srv.name), f"mcp:{srv.id}", value=srv.enabled, classes="connector-chip")
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close", variant="primary")

    @on(ToggleChip.Changed)
    def _toggle(self, event: ToggleChip.Changed) -> None:
        if not event.key.startswith("mcp:"):
            return
        sid = event.key[len("mcp:"):]
        srv = next((s for s in self._data.mcp_servers if s.id == sid), None)
        if srv is not None:
            srv.enabled = event.value
            store.upsert_mcp(self._data, srv)
            store.save(self._data)

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class ProjectItem(ListItem):
    def __init__(self, project_id: Optional[str], label) -> None:
        super().__init__(Label(label))
        self.project_id = project_id


class ChatItem(ListItem):
    def __init__(self, chat: Chat, subtitle: str) -> None:
        super().__init__(Label(title_sub(chat.title, subtitle or "—")))
        self.chat_id = chat.id


# MLX model-server runtime failures (e.g. an attention/KV `[reshape]` mismatch, OOM, a
# shape/broadcast error) — these come from the model engine, NOT the `tools` param, so
# retrying the request in prompted-tools mode just fails again after another long wait.
_FATAL_GEN_MARKERS = ("reshape", "out of memory", "shape (", "broadcast")


def _is_fatal_generation_error(exc: Exception) -> bool:
    """True for a model-server generation failure that prompted-mode can't rescue."""
    s = str(exc).lower()
    return any(k in s for k in _FATAL_GEN_MARKERS)


# When a tool loop runs out of iterations still calling tools (the model never wrote a
# final answer — e.g. it kept re-searching), we make ONE more turn with no tools and
# this nudge, so the user gets a real answer instead of "(no answer)".
_WRAP_UP_PROMPT = ("Now answer my question using the information gathered above. "
                   "Do NOT call any more tools — give your best final answer.")

# A turn that finishes with finish_reason == "length" was cut off at the token limit, not done.
# We push the partial answer + this nudge so the model resumes, instead of the loop misreading a
# truncated turn as a finished one (the "reads a bit, then stops" symptom).
_CONTINUE_TRUNCATED_PROMPT = ("Your previous message was cut off at the token limit. Continue "
                              "exactly where you left off — do not repeat what you already wrote.")

# Slash commands typed into the prompt. Matched on the WHOLE (trimmed, lower-cased) message so a
# real message that merely starts with "/" (a path, "/plan the rollout") is sent normally.
# The "/" command menu: (command, one-line description). _SLASH_COMMANDS (matched as a whole
# message in _handle_slash_command) is derived from this, so the menu and the dispatcher never drift.
_SLASH_COMMAND_INFO = (
    ("/build", "make changes, ask before each file/command action"),
    ("/plan", "propose a plan, take no actions"),
    ("/auto", "make changes and run tools without asking"),
    ("/compact", "summarize the chat to free up context"),
    ("/help", "list the available commands"),
)
_SLASH_COMMANDS = {cmd for cmd, _ in _SLASH_COMMAND_INFO}
# A bare "/command" prefix being typed (no space/argument yet) → show the menu. Once a space is
# typed ("/plan the rollout"), this stops matching and the text is sent as a normal message.
_SLASH_TRIGGER_RE = re.compile(r"^/\w*$")

# Sent to the model to summarize the conversation when compacting context (manual /compact or the
# automatic >95% trigger). The summary REPLACES the prior turns, so it must stand on its own.
_COMPACT_INSTRUCTIONS = (
    "Summarize our conversation so far into a compact but complete brief, so we can keep going after "
    "the earlier turns are cleared from context. Preserve everything that matters: my goals and "
    "constraints, decisions made and why, key facts and code, file paths, and any unfinished tasks or "
    "next steps. Use tight bullet points under short headings. Do not ask questions, add pleasantries, "
    "or invent anything not in the conversation. Output ONLY the summary."
)

# The visible user turn that stands in for the cleared history (a valid user→assistant pair keeps
# templates that require alternating roles happy).
_COMPACT_USER_MARKER = "⟢ Earlier conversation compacted to free up context."


class SubagentsModal(ModalScreen[Optional[str]]):
    """The subagents dropdown: pick a specialist to open as a side chat, or manage
    the list (new / edit / delete). Dismisses with a subagent id to start a side
    chat with it, or None."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, data) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("[b]Subagents[/]  ·  open a specialist in a side chat")
            yield VerticalScroll(id="sa-modal-list")
            with Horizontal(id="modal-buttons"):
                yield Button("+ New", id="sa-modal-new", variant="primary")
                yield Button("Close", id="sa-modal-close")

    def on_mount(self) -> None:
        self._rebuild()

    def on_screen_resume(self) -> None:
        # back from the editor → reflect any new/edited subagent
        self._data.subagents = store.load().subagents
        self._rebuild()

    def _server_name(self, sid: Optional[str]) -> str:
        cfg = next((s for s in self.app.config.servers if s.id == sid), None) if sid else None
        return cfg.name if cfg else "no model"

    def _rebuild(self) -> None:
        lst = self.query_one("#sa-modal-list", VerticalScroll)
        lst.remove_children()
        if not self._data.subagents:
            lst.mount(Label(plain("No subagents yet — + New to create one."), classes="hint"))
            return
        for sub in self._data.subagents:
            name = Static(Content.assemble((sub.name, "bold"), "  ", (self._server_name(sub.server_id), "dim")),
                          classes="sa-modal-name")
            chat_btn = Button("Chat", variant="success", classes="sa-chat")
            edit_btn = Button("Edit", classes="sa-edit")
            del_btn = Button("✕", classes="sa-del")
            for b in (chat_btn, edit_btn, del_btn):
                b._sa_id = sub.id  # widget ids can't hold a uuid → carry it as an attribute
            lst.mount(Horizontal(name, chat_btn, edit_btn, del_btn, classes="sa-modal-row"))

    @on(Button.Pressed, ".sa-chat")
    def _chat(self, event: Button.Pressed) -> None:
        self.dismiss(getattr(event.button, "_sa_id", None))

    @on(Button.Pressed, ".sa-edit")
    def _edit(self, event: Button.Pressed) -> None:
        from .subagent_editor import SubagentEditorScreen
        sub = store.get_subagent(self._data, getattr(event.button, "_sa_id", ""))
        if sub is not None:
            self.app.push_screen(SubagentEditorScreen(sub))

    @on(Button.Pressed, ".sa-del")
    def _delete(self, event: Button.Pressed) -> None:
        sub = store.get_subagent(self._data, getattr(event.button, "_sa_id", ""))
        if sub is not None:
            self.run_worker(self._delete_flow(sub), exclusive=True)

    async def _delete_flow(self, sub: Subagent) -> None:
        ok = await self.app.push_screen_wait(
            ConfirmModal(f'Delete subagent "{sub.name}"?', confirm_label="Delete"))
        if not ok:
            return
        store.delete_subagent(self._data, sub.id)
        store.save(self._data)
        self._rebuild()
        self.notify("Subagent deleted")

    @on(Button.Pressed, "#sa-modal-new")
    def _new(self) -> None:
        from .subagent_editor import SubagentEditorScreen
        self.app.push_screen(SubagentEditorScreen())

    @on(Button.Pressed, "#sa-modal-close")
    def _close_btn(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


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
        Binding("ctrl+b", "subagents", "Subagents"),
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
        # per-pane generation state — the main model and a subagent run on separate
        # ports, so each pane streams independently; the Send/Stop button reflects the
        # FOCUSED pane only (you can talk to one while the other is still answering).
        self._gen = {"main": False, "side": False}      # is each pane generating
        self._cancel_flags = {"main": False, "side": False}  # Stop requested per pane
        self._auto_approve_fs = False  # "approve all" for file/command ops this session
        self._prompted_servers: set[str] = set()  # servers whose native tools failed → prompted mode
        self._slash_items: list[tuple[str, str]] = []  # commands currently shown in the "/" menu
        # side chat: a second 50/50 pane holding a live conversation with a subagent
        self._active_pane = "main"  # which pane the single input sends to ("main" | "side")
        self._side_open = False
        self._side_sub: Optional[Subagent] = None
        self._side_cfg: Optional[ServerConfig] = None  # the (possibly port-bumped) profile it runs on
        self._side_messages: list[ChatMessage] = []
        self._side_base_url = ""  # the subagent server's actual address (may be a bumped port)
        self._side_ready = False  # the subagent server finished loading
        self._recorder = None  # voice.Recorder while the mic is capturing, else None
        self._speaker = None   # voice.Speaker while reading a reply aloud, else None
        self._compacting = False  # a context-compaction summary is in flight (guards re-entry)
        self._pending_send: tuple = ("", [])  # (text, attachments) handed to the next main run
        self._active_main = None  # (client, run_id) of the in-flight main run, for Stop → cancel_run

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
                with Horizontal(id="panes"):
                    with Vertical(id="main-pane", classes="pane active"):
                        with Horizontal(id="chat-topbar"):
                            yield Static("", id="chat-title")
                            yield Select([], id="skill-select", prompt="skill", allow_blank=True)
                            yield Select([], id="server-select", prompt="server", allow_blank=True)
                        yield VerticalScroll(id="transcript")
                    with Vertical(id="side-pane", classes="pane hidden"):
                        with Horizontal(id="side-head"):
                            yield Static("", id="side-title")
                            yield Button("✕ Close", id="side-close", variant="error")
                        yield VerticalScroll(id="side-transcript")
                with Horizontal(id="chat-chips"):  # chips + the context counter, pushed right
                    yield Static("mode: build", id="chip-mode", classes="chip chip-action")
                    yield ToggleChip("reason", "reasoning", id="chip-reasoning")
                    yield Static("effort: auto", id="chip-effort", classes="chip")
                    yield ToggleChip("web", "web", id="chip-web")
                    yield ToggleChip("coding", "coding", id="chip-coding")
                    yield ToggleChip("tools", "tools", id="chip-tools")
                    yield Static("connectors ▾", id="chip-connectors", classes="chip chip-action")
                    yield Static("subagents ▾", id="chip-subagents", classes="chip chip-action")
                    yield Static(classes="actions-spacer")  # pushes the context counter to the right edge
                    yield Static("", id="context-bar", classes="ctx-bar")
                yield Static("", id="attachments", classes="hidden")
                yield Input(id="attach", placeholder="paste a file path, Enter to attach", classes="hidden")
                yield OptionList(id="slash-suggest")  # the "/" command menu (sits just above the prompt)
                with Horizontal(id="chat-inputrow"):
                    yield PromptArea(id="prompt", soft_wrap=True)
                    yield Button("+ Attach", id="attach-btn")
                    yield Button("Mic", id="mic-btn")
                    yield Button("Send", id="send", variant="primary")
                with Horizontal(id="chat-actions"):  # secondary actions, compact, below the input
                    yield Button("↻ Regenerate", id="regenerate")
                    yield Button("✎ Edit last", id="edit-last")
                    yield Button("Read aloud", id="read-aloud")
                    yield Button("⤓ Export", id="export")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", PromptArea).border_title = "Enter to send · Shift+Enter newline · type / for commands"
        # the "/" command menu is mouse-clickable but never takes keyboard focus — the prompt keeps
        # focus and routes arrow/Enter/Tab to it (see PromptArea._on_key).
        self.query_one("#slash-suggest", OptionList).can_focus = False
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

    # --- subagents (side chat) -------------------------------------------

    def action_subagents(self) -> None:
        self._open_subagents_modal()

    def _open_subagents_modal(self) -> None:
        def _opened(sub_id: Optional[str]) -> None:
            if not sub_id:
                return
            sub = store.get_subagent(self.data, sub_id)
            if sub is not None:
                self._open_side_chat(sub)
        self.app.push_screen(SubagentsModal(self.data), _opened)

    def _pane_of(self, widget) -> Optional[str]:
        """Which pane a clicked widget belongs to ('main' | 'side'), or None."""
        node = widget
        while node is not None:
            wid = getattr(node, "id", None)
            if wid == "side-pane":
                return "side"
            if wid == "main-pane":
                return "main"
            node = node.parent
        return None

    def _set_active_pane(self, which: str) -> None:
        """Route the single shared input to a pane, highlight it, and label the prompt."""
        if which == "side" and not self._side_open:
            which = "main"
        self._active_pane = which
        try:
            self.query_one("#main-pane").set_class(which == "main", "active")
            self.query_one("#side-pane").set_class(which == "side", "active")
        except Exception:  # noqa: BLE001
            pass
        try:
            prompt = self.query_one("#prompt", PromptArea)
            if which == "side" and self._side_sub is not None:
                prompt.border_title = f"↦ {self._side_sub.name} · Enter to send"
            else:
                prompt.border_title = "Enter to send · Shift+Enter for newline"
        except Exception:  # noqa: BLE001
            pass
        self._sync_send_button()  # the button shows THIS pane's Send/Stop state

    @work(exclusive=True, group="side-open", exit_on_error=False)
    async def _open_side_chat(self, sub: Subagent) -> None:
        """Open a 50/50 side chat with `sub`: start its model server on the BACKEND (on its own
        free port via ``bump``, so the main model stays loaded) and route the input to it."""
        cfg = self._server_by_id(sub.server_id)
        if cfg is None:
            self.notify("This subagent has no model — edit it to choose one", severity="warning")
            return
        if self._side_open:
            await self._close_side_chat(unload=True)  # one side chat at a time → replace
        client = await self.app.backend()

        self.query_one("#side-pane").remove_class("hidden")
        self._side_open = True
        self._side_sub = sub
        self._side_cfg = cfg
        self._side_base_url = cfg.base_url()
        self._side_ready = False
        self._side_messages = []
        self.query_one("#side-title", Static).update(
            Content.assemble(("↦ " + sub.name, "bold"), "  ", (cfg.model or "", "dim")))
        transcript = self.query_one("#side-transcript", VerticalScroll)
        transcript.remove_children()
        self._set_active_pane("side")
        self.query_one("#prompt", PromptArea).focus()
        await transcript.mount(Static(plain(f"Loading {cfg.name} … model load can take a while."), classes="hint"))
        try:
            snap = await client.start_server(cfg.id, bump=True)  # runs alongside the main model
        except Exception as exc:  # noqa: BLE001
            if self._side_open and self._side_sub is sub:
                await transcript.mount(Static(plain(f"✕ failed to load: {exc}"), classes="hint"))
            return
        self._side_base_url = snap.get("base_url") or cfg.base_url()
        ready = False
        for _ in range(900):  # poll readiness (~270s)
            if not (self._side_open and self._side_sub is sub):
                return  # closed/replaced while loading
            status = (await client.server_status(cfg.id)).get("status")
            if status == "ready":
                ready = True
                break
            if status in ("error", "stopped"):
                await transcript.mount(Static(plain("✕ failed to load — close and try again."), classes="hint"))
                return
            await asyncio.sleep(0.3)
        if not (self._side_open and self._side_sub is sub):
            return  # closed/replaced while loading
        if not ready:  # poll loop exhausted without the server reporting ready — don't fake it
            await transcript.mount(Static(
                plain("✕ timed out waiting for the model to load — close and try again."), classes="hint"))
            return
        self._side_ready = True
        transcript.remove_children()
        await transcript.mount(Static(plain(f"● {sub.name} ready · {cfg.model}"), classes="hint"))
        self.query_one("#prompt", PromptArea).focus()

    async def _close_side_chat(self, unload: bool = True) -> None:
        """Close the side pane and (by default) unload the subagent's server."""
        if not self._side_open:
            return
        cfg, sub = self._side_cfg, self._side_sub
        self._cancel_flags["side"] = True  # stop any in-flight side reply
        self._gen["side"] = False
        self._side_open = False
        self._side_sub = None
        self._side_cfg = None
        self._side_messages = []
        try:
            self.query_one("#side-pane").add_class("hidden")
            self.query_one("#side-transcript", VerticalScroll).remove_children()
        except Exception:  # noqa: BLE001
            pass
        self._set_active_pane("main")
        if unload and cfg is not None:
            self.notify(f"Unloading {sub.name if sub else cfg.name} …")
            try:
                client = await self.app.backend()
                await client.stop_server(cfg.id)
            except Exception:  # noqa: BLE001
                pass

    @on(Button.Pressed, "#side-close")
    def _side_close_btn(self) -> None:
        # _close_side_chat already cancels + stops the server
        self.run_worker(self._close_side_chat(unload=True))

    def on_unmount(self) -> None:
        # Leaving the chat for good (the screen is being popped): stop in-flight generation
        # and unload a side-chat subagent's server. That server lives on the app, not this
        # screen, so without this it stays resident (GBs of weights) until the app quits.
        # on_screen_suspend is deliberately NOT used — pushing a modal/manager on top must
        # not tear the side chat down.
        flags = getattr(self, "_cancel_flags", None)
        if isinstance(flags, dict):
            flags["main"] = flags["side"] = True
        # stop any in-flight voice recording / playback
        rec, self._recorder = getattr(self, "_recorder", None), None
        if rec is not None:
            try:
                rec.stop()
            except Exception:  # noqa: BLE001
                pass
        sp, self._speaker = getattr(self, "_speaker", None), None
        if sp is not None:
            try:
                sp.stop()
            except Exception:  # noqa: BLE001
                pass
        cfg = getattr(self, "_side_cfg", None)
        if getattr(self, "_side_open", False) and cfg is not None:
            async def _unload_side(server_id):
                try:
                    client = await self.app.backend()
                    await client.stop_server(server_id)
                except Exception:  # noqa: BLE001 — app may be tearing down too
                    pass
            try:
                self.app.run_worker(_unload_side(cfg.id), exclusive=False)
            except Exception:  # noqa: BLE001
                pass

    # --- opening / creating ---------------------------------------------

    def _default_server(self) -> Optional[ServerConfig]:
        servers = self.app.config.servers
        if not servers:
            return None
        last = self.app.config.settings.last_used_id
        return next((s for s in servers if s.id == last), servers[0])

    def _server_by_id(self, sid: Optional[str]) -> Optional[ServerConfig]:
        return next((s for s in self.app.config.servers if s.id == sid), None)

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
        # Re-read before writing: chats.json is one shared document and the backend appends turns to
        # it during runs, so saving our cached self.data could clobber those. Adopt the fresh
        # snapshot, add the new chat to it, and keep self.chat attached to it.
        self.data = store.load()
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
        self.query_one("#chip-web", ToggleChip).set_value(bool(self.chat and self.chat.web_search))
        self.query_one("#chip-tools", ToggleChip).set_value(bool(self.chat and self.chat.tools))
        self._sync_mode_chip()
        self.query_one("#chip-coding", ToggleChip).set_value(bool(self.chat and self.chat.coding))
        self._update_topbar()
        self._update_context_bar()
        self._render_transcript()
        self.query_one("#prompt", PromptArea).focus()

    def _sync_reasoning_switch(self) -> None:
        # Always togglable. The name heuristic only sets the DEFAULT (when the model/server
        # changes); the user can turn thinking on for ANY model — including reasoners we don't
        # recognize by name (e.g. Step) — so we never lock the control.
        chip = self.query_one("#chip-reasoning", ToggleChip)
        chip.set_enabled(True)
        chip.set_value(bool(self.chat and self.chat.reasoning))
        self._sync_effort_chip()

    _EFFORT_CYCLE = (None, "off", "low", "medium", "high")

    def _sync_effort_chip(self) -> None:
        """Reflect the reasoning-effort level. Always shown/clickable — an explicit effort is sent
        regardless of the name heuristic, so unrecognized reasoning models still respond."""
        try:
            chip = self.query_one("#chip-effort", Static)
        except Exception:  # noqa: BLE001 — not mounted yet
            return
        chip.set_class(False, "hidden")  # never gated off — the heuristic only chooses the default
        effort = self.chat.reasoning_effort if self.chat else None
        chip.update(f"effort: {effort or 'auto'}")
        chip.set_class(bool(effort), "-on")

    def _cycle_effort(self) -> None:
        """Click the effort chip: cycle auto → off → low → medium → high → auto. Maps to the
        right chat-template kwarg per model (gpt-oss reasoning_effort / Qwen3 enable_thinking)."""
        if not self.chat:
            return
        cur = self.chat.reasoning_effort if self.chat.reasoning_effort in self._EFFORT_CYCLE else None
        self.chat.reasoning_effort = self._EFFORT_CYCLE[
            (self._EFFORT_CYCLE.index(cur) + 1) % len(self._EFFORT_CYCLE)]
        self._sync_effort_chip()
        self._persist()
        self.notify(f"Reasoning effort: {self.chat.reasoning_effort or 'auto (model default)'}")

    _MODES = ("build", "plan", "auto")
    _MODE_NOTES = {
        "build": "Build mode — I make changes and ask before each file/command action.",
        "plan": "Plan mode — I'll propose a plan to approve, not make changes.",
        "auto": "Auto mode — I make changes and run tools WITHOUT asking. Use with care.",
    }

    def _sync_mode_chip(self) -> None:
        """Reflect build/plan/auto on the mode chip (lit when it's not the default 'build')."""
        try:
            chip = self.query_one("#chip-mode", Static)
        except Exception:  # noqa: BLE001 — not mounted yet
            return
        mode = self.chat.mode if self.chat else "build"
        chip.update(f"mode: {mode}")
        chip.set_class(mode != "build", "-on")

    def _cycle_mode(self) -> None:
        """Click the mode chip: build → plan → auto → build."""
        if not self.chat:
            return
        cur = self.chat.mode if self.chat.mode in self._MODES else "build"
        self._set_mode(self._MODES[(self._MODES.index(cur) + 1) % len(self._MODES)])

    def _set_mode(self, mode: str) -> None:
        if not self.chat or mode not in self._MODES:
            return
        self.chat.mode = mode
        self._sync_mode_chip()
        self._update_topbar()
        self.notify(self._MODE_NOTES[mode])
        self._persist()

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
        if self.chat.coding:
            parts += ["   ", ("● CODING", "bold #7fb069")]
        if self.chat.mode == "plan":
            parts += ["   ", ("● PLAN MODE", "bold #d19a66")]
        elif self.chat.mode == "auto":
            parts += ["   ", ("● AUTO", "bold #e06c75")]  # reddish: it acts without asking
        self.query_one("#chat-title", Static).update(Content.assemble(*parts))

    @staticmethod
    def _context_cap_of(cfg) -> Optional[int]:
        """The context size the user explicitly configured on a profile — `--max-kv-size` for
        mlx-vlm / vllm-mlx, `-c` for llama-cpp — or None when unset or the engine can't cap it.
        mlx-lm can't cap context, so a (stale) setting there is ignored."""
        engine = getattr(cfg, "engine", None) if cfg else None
        if engine in ("mlx-vlm", "vllm-mlx"):
            return cfg.max_kv_size or None
        if engine == "llama-cpp":
            return cfg.ctx or None
        return None

    @staticmethod
    def _sampling_of(cfg) -> dict:
        """A profile's sampling settings as OpenAI request params, sent on every request for the
        model — engine-independent (every OpenAI-compatible server reads them from the body),
        unlike the old per-engine launch flags. Only includes values the user actually set."""
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

    def _effective_context(self) -> Optional[int]:
        """The context budget to meter against: the profile's configured cap (bounded by the
        model's true max) when set, else the model's max alone."""
        if not self.chat:
            return None
        model_max = capabilities.context_window(self.chat.model)
        cfg = self._server_by_id(self.chat.server_id) if self.chat.server_id else None
        cap = self._context_cap_of(cfg)
        if cap and model_max:
            return min(cap, model_max)
        return cap or model_max

    def _context_usage(self) -> Optional[tuple[int, int]]:
        """(estimated tokens used, context window) for the current chat, or None when the
        window can't be determined. Shared by the context bar and the 95% auto-compact check."""
        if not self.chat:
            return None
        window = self._effective_context()
        if not window:
            return None
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        used = capabilities.estimate_prompt_tokens(messages)
        fs = self._fs_root()
        if fs:
            used += capabilities.approx_tokens(fs_tools.system_note(fs))
        return used, window

    def _update_context_bar(self) -> None:
        """Show how much of the configured context the conversation uses (the
        profile's --max-kv-size if set, else the model's window). Hidden when
        neither can be determined ('if available')."""
        bar = self.query_one("#context-bar", Static)
        usage = self._context_usage()
        if not usage:
            bar.update("")
            return
        bar.update(_context_bar_markup(*usage))

    def _maybe_autocompact(self) -> None:
        """Between runs (never mid-reply), auto-compact when the conversation passes 95% of the
        context window — so a long chat doesn't start silently dropping its own earliest turns."""
        if self._gen.get("main", False) or self._compacting or not self.chat:
            return
        usage = self._context_usage()
        if not usage:
            return
        used, window = usage
        if window < 2000 or used < int(window * 0.95):
            return  # tiny windows can't fit a useful summary; below threshold → nothing to do
        real = [m for m in self.chat.messages if m.role in ("user", "assistant")]
        if len(real) < 3:
            return  # a single oversized turn won't shrink by summarizing — don't thrash
        self.notify("Context over 95% — compacting automatically to free up space.", timeout=6)
        self._start_compaction(auto=True)

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

    async def _mount_thinking(self, transcript, label: Optional[str] = None) -> Horizontal:
        """Mount an animated 'thinking' bubble at the bottom of `transcript` and
        return its row so the caller can remove it once a reply/tool result lands."""
        row = self._assemble_row(
            label or self._server_label_for(self.chat), "msg-assistant",
            [ThinkingIndicator(classes="msg-body thinking-indicator")],
        )
        await transcript.mount(row)
        self._scroll_widget(transcript)
        return row

    def _assistant_body_widgets(self, text: str) -> list:
        """Prose runs as Markdown + each fenced code block as a copyable CodeBlock."""
        widgets: list = []
        for block in split_blocks(text):
            if block[0] == "code":
                widgets.append(CodeBlock(block[2], block[1]))
            elif block[1].strip():
                # A Markdown WIDGET (not a Static + Rich-Markdown renderable) so the text is
                # selectable — drag to highlight any part of a reply, then Ctrl/Cmd+C. Links are
                # clickable too (Markdown.LinkClicked); linkify_urls turns bare URLs into links.
                widgets.append(Markdown(linkify_urls(block[1]), classes="msg-body"))
        return widgets or [Static("[dim](no content)[/]", classes="msg-body")]

    @staticmethod
    def _stats_text(tps: float, tokens: int, elapsed: float) -> str:
        return f"↯ {tps:.1f} tok/s · {tokens} tok · {elapsed:.1f}s"

    def _copy_control(self, text: str) -> Static:
        """A clickable '⧉ Copy' affordance carrying the message text (read in on_click)."""
        c = Static("⧉ Copy", classes="msg-copy")
        c._copy_text = text or ""
        return c

    def _tool_call_widgets(self, calls: list[dict]) -> list:
        """Compact lines describing the tool calls an assistant turn made — for reloaded
        transcripts (live turns show richer per-call bubbles via _exec_tool)."""
        out: list = []
        for c in calls or []:
            phrase = _tool_phrase(c.get("name") or "", c.get("arguments") or {})
            out.append(Static(Content.assemble(("▸ ", "dim"), (phrase, "")), classes="msg-stats"))
        return out

    def _message_widget(self, m: ChatMessage) -> Horizontal:
        if m.role == "tool":  # a persisted tool result — same compact bubble as the live one
            preview = m.text if len(m.text) <= 500 else m.text[:500] + " …"
            row, _ = self._bubble("▸", "msg-tool",
                                  Content.assemble((_tool_phrase(m.tool_name or "", {}), "bold"), "\n", (preview, "dim")))
            return row
        if m.role == "assistant":
            has_text = bool((m.text or "").strip())
            widgets = self._assistant_body_widgets(m.text) if has_text else []
            if m.tool_calls:  # an agentic tool-call turn — list what it called
                widgets += self._tool_call_widgets(m.tool_calls)
            if not widgets:
                widgets = [Static("[dim](no content)[/]", classes="msg-body")]
            if m.tps:
                widgets.append(Static(self._stats_text(m.tps, m.n_tokens or 0, m.elapsed or 0.0), classes="msg-stats"))
            if has_text:
                widgets.append(self._copy_control(m.text))
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

    def _scroll_widget(self, w) -> None:
        try:
            w.scroll_end(animate=False)
        except Exception:  # noqa: BLE001
            pass

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
        if not chat or (self.chat and chat.id == self.chat.id):
            return
        if self._gen.get("main", False):
            # the in-flight worker appends/persists into self.chat at the end — switching now
            # would land the reply on the wrong chat (and wedge the new chat's Send button).
            self.notify("Stop the current response before switching chats", severity="warning")
            self._reselect_current_chat()
            return
        self._open_chat(chat)

    def _reselect_current_chat(self) -> None:
        """Re-highlight the open chat after refusing a list selection. Setting `.index` posts a
        Highlighted message (not Selected), so it doesn't re-enter _chat_selected."""
        if not self.chat:
            return
        lv = self.query_one("#chats", ListView)
        for i, item in enumerate(lv.children):
            if getattr(item, "chat_id", None) == self.chat.id:
                lv.index = i
                return

    @on(Select.Changed, "#server-select")
    def _server_changed(self, event: Select.Changed) -> None:
        # event.value == current → programmatic sync, re-select, or a revert: ignore.
        if not self.chat or event.value is Select.NULL or event.value == self.chat.server_id:
            return
        cfg = self._server_by_id(event.value)
        if cfg is None:
            return
        if self._gen.get("main", False):
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

    async def _confirm_and_switch(self, cfg: ServerConfig) -> None:
        """Switch the chat to `cfg`'s model: confirm, free its port (over the wire), repoint the
        chat, and load the model — all via the backend."""
        client = await self.app.backend()
        names = {s.id: s.name for s in self.app.config.servers}
        statuses = await client.all_server_status()
        blockers = {s["id"] for s in statuses if s.get("is_running")
                    and (s.get("host"), s.get("port")) == (cfg.host, cfg.port) and s.get("id") != cfg.id}
        old = self.chat.server_id
        if old and old != cfg.id and any(s.get("id") == old and s.get("is_running") for s in statuses):
            blockers.add(old)
        unload = ("This unloads " + ", ".join(names.get(b, b) for b in blockers)) if blockers \
            else "This starts the selected model"
        prompt = (f"Switch this chat to '{cfg.name}'?\n\n{unload} and loads '{cfg.name}' on the server — "
                  "the current model's loaded context (weights + KV cache) will be lost.")
        ok = await self.app.push_screen_wait(ConfirmModal(prompt, confirm_label="Switch & reload"))
        if not ok:
            self._revert_server_select()
            return
        for bid in blockers:
            await client.stop_server(bid)
        self.chat.server_id = cfg.id
        self.chat.base_url = cfg.base_url()
        self.chat.model = cfg.model
        self.chat.reasoning = self.chat.reasoning and capabilities.supports_reasoning(cfg.model)
        await client.patch_session(self.chat.id, {"server_id": cfg.id, "reasoning": self.chat.reasoning})
        await client.patch_settings({"last_used_id": cfg.id})
        await self.app.refresh_config()
        self._sync_reasoning_switch()
        self._update_topbar()
        self._refresh_chats()
        await self._load_server(cfg)

    async def _load_server(self, cfg: ServerConfig) -> None:
        """Start `cfg`'s model server on the backend and surface readiness (polls status)."""
        client = await self.app.backend()
        self.notify(f"Loading {cfg.name} — model load can take a while …")
        try:
            await client.start_server(cfg.id)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "not found" in msg.lower():
                self.notify(f"{cfg.engine} server binary not found — press p on the dashboard to install",
                            severity="error")
            else:
                self.notify(f"Failed to start: {msg}", severity="error")
            return
        for _ in range(900):  # ~270s budget for a slow model load
            status = (await client.server_status(cfg.id)).get("status")
            if status == "ready":
                self.notify(f"{cfg.name} ready")
                return
            if status in ("error", "stopped"):
                st = await client.server_status(cfg.id)
                self.notify(st.get("message") or "model failed to load", severity="error")
                return
            await asyncio.sleep(0.3)
        self.notify(f"{cfg.name}: still loading — open it from the dashboard to watch", severity="warning")

    @on(ToggleChip.Changed)
    def _chip_changed(self, event: ToggleChip.Changed) -> None:
        if not self.chat:
            return
        key, val = event.key, event.value
        if key == "reasoning":
            self.chat.reasoning = val
        elif key == "web":
            self.chat.web_search = val
        elif key == "tools":
            self.chat.tools = val
        elif key == "coding":
            self.chat.coding = val
            self.notify("Coding mode on — senior-engineer prompt; the model validates its work"
                        if val else "Coding mode off")
        self._update_topbar()
        self._persist()

    @on(Markdown.LinkClicked)
    def _markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        # a link clicked inside a rendered reply → open it in the browser
        if event.href:
            self.app.open_url(event.href)

    def on_click(self, event: events.Click) -> None:
        # a click on a rendered link (markdown link or a linkified bare URL) → open it
        style = getattr(event, "style", None)
        link = getattr(style, "link", None) if style else None
        if link:
            # links inside a Markdown widget are already opened by @on(Markdown.LinkClicked);
            # skip them here so the same click isn't handled (and opened) twice.
            w = event.widget
            if w is None or not any(isinstance(a, Markdown) for a in w.ancestors_with_self):
                self.app.open_url(link)
            return
        wid = getattr(event.widget, "id", None)
        if wid == "chip-connectors":  # a plain Static, not a toggle → open the popup
            self.app.push_screen(ConnectorsModal(self.data))
            return
        if wid == "chip-subagents":
            self._open_subagents_modal()
            return
        if wid == "chip-effort":
            self._cycle_effort()
            return
        if wid == "chip-mode":
            self._cycle_mode()
            return
        if event.widget is not None and event.widget.has_class("msg-copy"):
            text = getattr(event.widget, "_copy_text", "")
            if text:
                self.app.copy_text(text)
                self.notify("Copied to clipboard")
        # clicking inside a pane focuses it → the single input sends there
        if self._side_open and event.widget is not None:
            pane = self._pane_of(event.widget)
            if pane is not None:
                self._set_active_pane(pane)

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

    # --- voice: mic (speech-to-text) -------------------------------------

    @on(Button.Pressed, "#mic-btn")
    def _mic_btn(self) -> None:
        # toggle push-to-talk: first click records, second click transcribes
        if self._recorder is not None:
            self._stop_recording_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        avail = voice.availability()
        if not avail.can_transcribe:
            self.notify(f"Voice input needs the audio deps. Install:  {voice.install_command()}",
                        severity="warning", timeout=10)
            return
        try:
            rec = voice.Recorder()
            rec.start()
        except Exception as exc:  # noqa: BLE001 — no mic / busy device / unsupported rate
            self._recorder = None
            self.notify(str(exc), severity="error", timeout=8)
            return
        self._recorder = rec
        self._set_mic_button(recording=True)
        self.notify("Listening… click the mic again to transcribe.", timeout=4)

    def _stop_recording_and_transcribe(self) -> None:
        rec, self._recorder = self._recorder, None
        self._set_mic_button(recording=False)
        if rec is None:
            return
        try:
            audio = rec.stop()
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Mic error: {exc}", severity="error")
            return
        if audio is None or len(audio) == 0:
            self.notify("No audio captured.", severity="warning")
            return
        self.notify("Transcribing…", timeout=3)
        self.run_worker(self._transcribe_worker(audio), group="voice-stt", exclusive=True)

    async def _transcribe_worker(self, audio) -> None:
        model = self.app.config.settings.voice_stt_model or voice.DEFAULT_STT_MODEL
        try:
            text = await asyncio.to_thread(voice.transcribe, audio, model)
        except Exception as exc:  # noqa: BLE001 — backend/model error
            self.notify(f"Transcription failed: {exc}", severity="error", timeout=10)
            return
        text = (text or "").strip()
        if not text:
            self.notify("Didn't catch that — try again.", severity="warning")
            return
        prompt = self.query_one("#prompt", PromptArea)
        existing = prompt.text
        joiner = " " if existing and not existing.endswith((" ", "\n")) else ""
        prompt.load_text(existing + joiner + text)
        prompt.focus()
        if self.app.config.settings.voice_autosend:
            self.action_send()

    def _set_mic_button(self, *, recording: bool) -> None:
        try:
            btn = self.query_one("#mic-btn", Button)
        except Exception:  # noqa: BLE001 — not mounted
            return
        btn.label = "Stop" if recording else "Mic"
        btn.set_class(recording, "-recording")

    # --- voice: read aloud (text-to-speech) ------------------------------

    @on(Button.Pressed, "#read-aloud")
    def _read_btn(self) -> None:
        if self._speaker is not None:
            self._stop_reading()
        else:
            self._start_reading()

    def _last_assistant_text(self) -> str:
        if not self.chat or not self.chat.messages:
            return ""
        for msg in reversed(self.chat.messages):
            if msg.role == "assistant" and msg.text.strip():
                return msg.text
        return ""

    def _start_reading(self) -> None:
        text = self._last_assistant_text()
        if not text:
            self.notify("No reply to read yet.", severity="warning")
            return
        if not voice.availability().can_speak:
            self.notify(f"No text-to-speech available. Install:  {voice.install_command()}",
                        severity="warning", timeout=10)
            return
        if voice.availability().tts_kokoro and not voice.kokoro_models_ready():
            self.notify("Fetching the Kokoro voice model (~325 MB) on first use…", timeout=6)
        self._speak(text)

    def _speak(self, text: str) -> None:
        sp = voice.Speaker(text, self.app.config.settings.voice_tts_voice or voice.DEFAULT_TTS_VOICE)
        self._speaker = sp
        self._set_read_button(reading=True)
        self.run_worker(self._speak_worker(sp), group="voice-tts", exclusive=True)

    async def _speak_worker(self, sp) -> None:
        try:
            await asyncio.to_thread(sp.run)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Couldn't read it aloud: {exc}", severity="error", timeout=8)
        finally:
            if self._speaker is sp:
                self._speaker = None
            self._set_read_button(reading=False)

    def _stop_reading(self) -> None:
        sp, self._speaker = self._speaker, None
        if sp is not None:
            try:
                sp.stop()
            except Exception:  # noqa: BLE001
                pass
        self._set_read_button(reading=False)

    def _set_read_button(self, *, reading: bool) -> None:
        try:
            btn = self.query_one("#read-aloud", Button)
        except Exception:  # noqa: BLE001 — not mounted
            return
        btn.label = "Stop" if reading else "Read aloud"
        btn.set_class(reading, "-reading")

    def _maybe_autoread(self) -> None:
        """After a main-pane reply lands, read it aloud if the user enabled auto-read."""
        if not getattr(self.app.config.settings, "voice_autoread", False):
            return
        if self._speaker is not None:  # already reading something
            return
        if not voice.availability().can_speak:
            return
        text = self._last_assistant_text()
        if text:
            self._speak(text)

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
        if self._gen.get(self._active_pane, False):
            self.action_stop()
        else:
            self.action_send()

    @on(PromptArea.Submitted)
    def _prompt_submitted(self) -> None:
        self.action_send()

    # --- "/" command menu ------------------------------------------------

    @on(TextArea.Changed, "#prompt")
    def _prompt_changed(self) -> None:
        self._update_slash_menu()

    def _update_slash_menu(self) -> None:
        """Show/filter the slash-command menu as the user types a bare '/command' prefix. Only for
        the main chat — slash commands don't apply to a subagent side chat."""
        sug = self.query_one("#slash-suggest", OptionList)
        text = self.query_one("#prompt", PromptArea).text
        sending_to_side = self._active_pane == "side" and self._side_open
        if sending_to_side or not _SLASH_TRIGGER_RE.match(text):
            self._slash_items = []
            sug.display = False
            return
        prefix = text.lower()
        self._slash_items = [(c, d) for c, d in _SLASH_COMMAND_INFO if c.startswith(prefix)]
        sug.clear_options()
        if not self._slash_items:
            sug.display = False
            return
        sug.add_options([self._slash_option(c, d) for c, d in self._slash_items])
        sug.highlighted = 0
        sug.display = True

    @staticmethod
    def _slash_option(cmd: str, desc: str) -> Option:
        return Option(Content.assemble((cmd, "bold"), (f"   {desc}", "dim")))

    @on(PromptArea.SlashSelected)
    def _slash_enter(self) -> None:  # Enter while the menu is open → run the highlighted command
        sug = self.query_one("#slash-suggest", OptionList)
        if sug.highlighted is not None and 0 <= sug.highlighted < len(self._slash_items):
            self._run_slash(self._slash_items[sug.highlighted][0])

    @on(OptionList.OptionSelected, "#slash-suggest")
    def _slash_clicked(self, event: OptionList.OptionSelected) -> None:  # click a row
        if 0 <= event.option_index < len(self._slash_items):
            self._run_slash(self._slash_items[event.option_index][0])
        event.stop()

    def _run_slash(self, cmd: str) -> None:
        self.query_one("#slash-suggest", OptionList).display = False
        self._handle_slash_command(cmd)  # clears the prompt + executes (mode change / compact / help)
        self.query_one("#prompt", PromptArea).focus()

    @on(PromptArea.SlashComplete)
    def _slash_complete(self) -> None:  # Tab fills in the highlighted command without running it
        sug = self.query_one("#slash-suggest", OptionList)
        if sug.highlighted is None or not (0 <= sug.highlighted < len(self._slash_items)):
            return
        prompt = self.query_one("#prompt", PromptArea)
        prompt.load_text(self._slash_items[sug.highlighted][0])
        prompt.move_cursor(prompt.document.end)  # caret to end so the next keystroke appends

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

    def _set_generating(self, pane: str, on: bool) -> None:
        self._gen[pane] = on
        if on:
            self._cancel_flags[pane] = False
        if pane == self._active_pane:
            self._sync_send_button()

    def _sync_send_button(self) -> None:
        """The single Send/Stop button reflects the FOCUSED pane's state, so the other
        pane can keep generating without locking it."""
        on = self._gen.get(self._active_pane, False)
        try:
            send = self.query_one("#send", Button)
            send.label = "■ Stop" if on else "Send"
            send.variant = "error" if on else "primary"
        except Exception:  # noqa: BLE001
            pass

    def _cancel_cb(self, pane: str):
        return lambda: self._cancel_flags.get(pane, False)

    # --- actions ---------------------------------------------------------

    def _request_cancel(self, pane: str) -> None:
        """Stop the focused pane's reply. The side pane polls a flag (in-process loop); the main
        pane's run lives in the backend, so we POST a cancel for it."""
        self._cancel_flags[pane] = True
        if pane == "main" and self._active_main is not None:
            client, run_id = self._active_main
            self.run_worker(client.cancel_run(self.chat.id, run_id), exclusive=False, exit_on_error=False)

    def action_back(self) -> None:
        pane = self._active_pane
        if self._gen.get(pane, False):  # Esc stops the focused pane's reply, not the screen
            self._request_cancel(pane)
            self.notify("Stopped")
            return
        self.app.pop_screen()

    def action_stop(self) -> None:
        pane = self._active_pane
        if self._gen.get(pane, False):
            self._request_cancel(pane)

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
        # returning from a manager/editor: pick up new skills, subagents, project edits.
        # Reload only the lists edited elsewhere — replacing self.data would detach
        # self.chat from self.data.chats.
        fresh = store.load()
        self.data.subagents = fresh.subagents
        self.data.mcp_servers = fresh.mcp_servers
        # Reload projects too (the project editor writes them backend-side) — without this a
        # newly-created project never shows up in the sidebar. Safe to replace the list: nothing
        # holds a reference into it the way self.chat references self.data.chats.
        self.data.projects = fresh.projects
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
        if kind == "chat" and self._gen.get("main", False):
            # deleting mid-generation re-points self.chat, so the worker's reply lands elsewhere
            self.notify("Stop the current response before deleting a chat", severity="warning")
            return
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
        if getattr(self, "_active_pane", "main") == "side":
            self.notify("Regenerate applies to the main chat, not the subagent side chat.", severity="warning")
            return
        if self._gen.get("main", False):
            self.notify("Still generating — Esc to stop", severity="warning")
            return
        if not self.chat or not self.chat.messages:
            return
        if self.chat.messages[-1].role != "assistant":
            self.notify("Nothing to regenerate", severity="warning")
            return
        # Drop the last answer AND any tool steps that produced it, back to the last user turn,
        # so regenerating redoes the whole turn (tool calls included), not just the final wording.
        last_user = next((i for i in range(len(self.chat.messages) - 1, -1, -1)
                          if self.chat.messages[i].role == "user"), None)
        if last_user is None:
            self.chat.messages.pop()
        else:
            del self.chat.messages[last_user + 1:]
        self._persist()
        self._render_transcript()
        self._generate()

    def action_edit_last(self) -> None:
        if getattr(self, "_active_pane", "main") == "side":
            self.notify("Edit-last applies to the main chat, not the subagent side chat.", severity="warning")
            return
        if self._gen.get("main", False) or not self.chat:
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
            if m.role == "tool":  # an agentic tool result — show it as a fenced block, not as prose
                lines += [f"## tool · {m.tool_name or 'tool'}", "", "```", m.text, "```", ""]
                continue
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
            for c in (m.tool_calls or []):  # tool-call turns: name the calls it made
                lines.append(f"- ▸ called `{c.get('name', '?')}`")
            if m.text:
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
        pane = "side" if (self._active_pane == "side" and self._side_open) else "main"
        if self._gen.get(pane, False):  # only THIS pane being busy blocks a send
            self.notify("Still generating in this pane — Esc to stop", severity="warning")
            return
        if pane == "side":
            self._send_side()
        else:
            self._send_main()

    def _send_side(self) -> None:
        if not self._side_sub or not self._side_cfg:
            self.notify("Side chat isn't ready yet", severity="warning")
            return
        prompt = self.query_one("#prompt", PromptArea)
        text = prompt.text.strip()
        if not text:
            return
        if not self._side_ready:
            self.notify("Subagent server isn't ready yet — give it a moment", severity="warning")
            return
        self._side_messages.append(ChatMessage(role="user", text=text))
        prompt.load_text("")
        transcript = self.query_one("#side-transcript", VerticalScroll)
        row, _ = self._bubble("You", "msg-user", plain(text), right=True)
        transcript.mount(row)
        self.call_after_refresh(lambda: self._scroll_widget(transcript))
        self._generate_side()

    def _send_main(self) -> None:
        if not self.chat:
            return
        text = self.query_one("#prompt", PromptArea).text.strip()
        if self._handle_slash_command(text):
            return  # a command (/plan, /compact, …) — consumed, not sent as a message
        if not text and not self._pending:
            return
        if not self.chat.base_url or not self.chat.model:
            self.notify("Pick a server first (or create one in the dashboard).", severity="warning")
            return

        atts = list(self._pending)
        # Render the user turn optimistically; the BACKEND appends + persists it (and the assistant
        # reply) when the run starts, so we don't write the store here (avoids a double-append).
        msg = ChatMessage(role="user", text=text, attachments=atts)
        if self.chat.title == "New chat" and text:
            self.chat.title = text[:40]
        self.query_one("#prompt", PromptArea).load_text("")
        self._pending = []
        self._refresh_attachments()

        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(self._message_widget(msg))
        self.call_after_refresh(self._scroll_end)
        self._pending_send = (text, atts)
        self._generate()

    # --- slash commands --------------------------------------------------

    def _handle_slash_command(self, text: str) -> bool:
        """Run a `/command` typed into the prompt. Returns True if it was a command (so the
        caller doesn't also send it as a chat message)."""
        cmd = text.strip().lower()
        if cmd not in _SLASH_COMMANDS:
            return False
        self.query_one("#prompt", PromptArea).load_text("")  # clear the typed command
        if cmd == "/help":
            self.notify("Commands:  /build · /plan · /auto (modes)   ·   /compact — summarize the chat to free context",
                        timeout=8)
        elif cmd in ("/build", "/plan", "/auto"):
            self._set_mode(cmd[1:])
        elif cmd == "/compact":
            self._start_compaction(auto=False)
        return True

    # --- context compaction ----------------------------------------------

    def _start_compaction(self, *, auto: bool) -> None:
        """Kick off a compaction summary, unless one (or a reply) is already running."""
        if self._gen.get("main", False) or self._compacting:
            if not auto:
                self.notify("Busy — wait for the current reply to finish.", severity="warning")
            return
        self._compaction_worker(auto)

    @work(exit_on_error=False, group="compact")
    async def _compaction_worker(self, auto: bool) -> None:
        """Run a backend compaction over the wire: it summarizes the conversation and replaces the
        history with a user→assistant pair (server-side). Manual via /compact, or automatic >95%."""
        chat = self.chat
        if not chat or not chat.base_url or not chat.model:
            if not auto:
                self.notify("Pick a server with a model first.", severity="warning")
            return
        real = [m for m in chat.messages if m.role in ("user", "assistant") and (m.text or m.attachments)]
        if len(real) < 2:
            if not auto:
                self.notify("Not enough conversation to compact yet.", severity="warning")
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        self._compacting = True
        self._set_generating("main", True)  # shows Stop; Esc cancels the compact run
        thinking = await self._mount_thinking(transcript, "Compacting context…")
        try:
            client = await self.app.backend()
            run_id = await client.start_run(chat.id, "", kind="compact")
            self._active_main = (client, run_id)
            async for _etype, _data in client.stream_run(chat.id, run_id):
                pass  # the summary replaces the history server-side; we just await completion
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Compaction failed: {exc}", severity="error", timeout=8)
        finally:
            try:
                await thinking.remove()
            except Exception:  # noqa: BLE001
                pass
            self._active_main = None
            self._compacting = False
            self._set_generating("main", False)
        # the backend replaced the history on disk — reload so self.chat reflects it and stays
        # attached to self.data.chats (same anti-clobber reason as after a normal run)
        self._resync_active_chat()
        self._render_transcript()
        self._scroll_end()
        self.notify("Context compacted — earlier turns summarized to free up space.")

    def _fs_root(self) -> Optional[str]:
        """The project's working directory if it exists on disk, else None."""
        proj = self._current_project()
        if proj and proj.working_dir:
            path = os.path.expanduser(proj.working_dir)
            if os.path.isdir(path):
                return path
        return None

    def _resync_active_chat(self) -> None:
        """Reload the store and re-point ``self.chat`` into the FRESH snapshot. The backend persists
        a run's turns to disk, so the in-memory cache is stale afterward. Reloading the whole
        ``self.data`` (and keeping ``self.chat`` attached to ``self.data.chats``, not a detached
        copy) is what makes the later ``store.save(self.data)`` writes — creating a chat, editing a
        title — safe: otherwise they'd write back a stale chat and clobber the turns the backend
        just appended."""
        if not self.chat:
            return
        self.data = store.load()
        got = store.get_chat(self.data, self.chat.id)
        if got is not None:
            self.chat = got

    @work(exit_on_error=False)
    async def _generate(self) -> None:
        """Run the user's turn on the BACKEND over HTTP+SSE: start a run, then render its event
        stream. The agent loop, tool execution, and persistence all live in the backend now — this
        method only kicks the run and paints the events."""
        text, atts = self._pending_send
        self._pending_send = ("", [])
        transcript = self.query_one("#transcript", VerticalScroll)
        self._set_generating("main", True)
        try:
            client = await self.app.backend()
            run_id = await client.start_run(self.chat.id, text,
                                            attachments=[a.model_dump() for a in atts])
            self._active_main = (client, run_id)
            await self._render_run(transcript, client.stream_run(self.chat.id, run_id),
                                   self._server_label_for(self.chat), show_reasoning=self.chat.reasoning,
                                   client=client, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 — a worker error must not wedge the UI
            self.notify(f"Generation failed: {exc}", severity="error", timeout=8)
        finally:
            self._active_main = None
            self._set_generating("main", False)
            # the backend persisted the turn — reload so the chat record + sidebar are current AND
            # self.chat stays attached to self.data.chats (so a later save can't clobber the turn)
            self._resync_active_chat()
            try:
                self._refresh_chats()
            except Exception:  # noqa: BLE001
                pass
            self._maybe_autocompact()  # between runs only — auto-summarize if context is nearly full
            self._maybe_autoread()

    async def _render_run(self, transcript, events, label: str, *, show_reasoning: bool,
                          client=None, run_id: Optional[str] = None) -> None:
        """Paint a backend run's SSE event stream into ``transcript``, reusing the same widgets the
        old in-process loop built — only the event SOURCE changed (now the wire). Handles streamed
        content/reasoning, per-call tool bubbles, the permission prompt, open-url, and the final
        answer. Shared shape so the renderer stays UI-only and the logic stays in the backend."""
        assistant_body: Optional[Static] = None  # the live streaming bubble (no-tools path)
        think_body: Optional[Static] = None
        thinking_row = None
        tool_bodies: dict = {}  # tool_id -> (body Static, phrase)
        content_acc: list[str] = []
        reason_acc: list[str] = []
        last_render = 0
        n_tools = 0
        final_text = ""
        finish_reason = "stop"

        async def _drop_spinner():
            nonlocal thinking_row
            if thinking_row is not None:
                try:
                    await thinking_row.remove()
                except Exception:  # noqa: BLE001
                    pass
                thinking_row = None

        async for etype, data in events:
            if etype == "run_start":
                thinking_row = await self._mount_thinking(transcript, label)
            elif etype == "reason":
                reason_acc.append(data.get("text", ""))
                if show_reasoning:
                    if think_body is None:
                        box, think_body = self._bubble("◌ thinking", "msg-think", "")
                        await transcript.mount(box, before=thinking_row) if thinking_row else await transcript.mount(box)
                    think_body.update(plain("".join(reason_acc)))
            elif etype == "content":
                if assistant_body is None:
                    await _drop_spinner()
                    assistant_body = Static("", classes="msg-body")
                    await transcript.mount(self._assemble_row(label, "msg-assistant", [assistant_body]))
                content_acc.append(data.get("text", ""))
                joined = "".join(content_acc)
                if last_render == 0 or "\n" in data.get("text", "") or len(joined) - last_render >= 24:
                    assistant_body.update(RichMarkdown(joined))
                    last_render = len(joined)
                self._scroll_widget(transcript)
            elif etype == "tool_start":
                await _drop_spinner()
                row, body = self._bubble("▸", "msg-tool", Content.assemble((data.get("phrase", ""), "bold")))
                await transcript.mount(row)
                tool_bodies[data.get("tool_id")] = (body, data.get("phrase", ""))
                n_tools += 1
                thinking_row = await self._mount_thinking(transcript, label)  # spinner for the next step
            elif etype == "tool_end":
                tb = tool_bodies.get(data.get("tool_id"))
                if tb is not None:
                    body, phrase = tb
                    style = "#e06c75" if data.get("status") == "denied" else "dim"
                    body.update(Content.assemble((phrase, "bold"), "\n", (data.get("preview", ""), style)))
            elif etype == "permission_request":
                await _drop_spinner()
                decision = await self.app.push_screen_wait(
                    PermissionModal(data.get("summary", ""), data.get("detail", "")))
                dec = decision if decision in ("once", "all") else "deny"
                if client is not None and run_id is not None:
                    await client.answer_permission(self.chat.id, run_id, data.get("id", ""), dec)
                thinking_row = await self._mount_thinking(transcript, label)
            elif etype == "open_url":
                try:
                    self.app.open_url(data.get("url", ""))
                except Exception:  # noqa: BLE001
                    pass
            elif etype == "notice":
                lvl = data.get("level", "info")
                sev = lvl if lvl in ("warning", "error") else "information"
                self.notify(data.get("text", ""), severity=sev, timeout=8)
            elif etype == "finish":
                finish_reason = data.get("reason", "stop")
                final_text = data.get("text", final_text)
            elif etype == "error":
                final_text = f"▲ {data.get('message', '')}"

        await _drop_spinner()
        answer = ""
        if assistant_body is not None:  # streamed path → re-render into prose + copyable code blocks
            answer = "".join(content_acc) or final_text
            try:
                bubble = assistant_body.parent
                await assistant_body.remove()
                for widget in self._assistant_body_widgets(answer):
                    await bubble.mount(widget)
                await bubble.mount(self._copy_control(answer))
            except Exception:  # noqa: BLE001
                pass
        elif final_text:  # tool path / non-streamed → mount the final answer row
            answer = final_text
            widgets = self._assistant_body_widgets(final_text)
            if n_tools:
                widgets.append(Static(f"▸ {n_tools} tool call{'' if n_tools == 1 else 's'}", classes="msg-stats"))
            widgets.append(self._copy_control(final_text))
            await transcript.mount(self._assemble_row(label, "msg-assistant", widgets))
        else:
            placeholder = "(stopped)" if finish_reason == "cancelled" else "(no answer)"
            await transcript.mount(self._assemble_row(
                label, "msg-assistant", [Static(f"[dim]{placeholder}[/]", classes="msg-body")]))
        self._scroll_end()
        return answer

    # --- subagents (side chat) -------------------------------------------

    def _subagent_system(self, sub: Subagent) -> str:
        parts = [p for p in (skills.instructions_for(sid) for sid in sub.skill_ids) if p]
        if sub.system_prompt.strip():
            parts.append(sub.system_prompt.strip())
        kb = knowledge.load_knowledge(list(getattr(sub, "knowledge_paths", []) or []))
        if kb:  # uploaded docs — always in the subagent's context
            parts.append(kb)
        if sub.web_search:  # the same use-the-tool insistence the old loop injected
            parts.append(
                "IMPORTANT: when the user asks for current information, facts, statistics, news, "
                "prices, or sources, you MUST call web_search first and base your answer on the "
                "results. NEVER fabricate sources, URLs, citations, or data.")
        return "\n\n---\n\n".join(parts)

    async def _subagent_toolset(self, sub: Subagent, stack) -> ToolSet:
        """web_search + the subagent's selected MCP connections (forced-enabled). No filesystem
        tools — a subagent never touches the project working dir."""
        specs: list = []
        sessions: dict = {}
        router: dict = {}
        if sub.web_search:
            specs.append(chat_tools.web_search_spec())
        if sub.tools and sub.mcp_server_ids:
            wanted = [s.model_copy(update={"enabled": True})
                      for s in store.load().mcp_servers if s.id in sub.mcp_server_ids]
            sessions, mcp_specs, router = await mcp_client.open_sessions(
                stack, wanted, on_error=lambda n, e: self.notify(f"MCP {n}: {e}", severity="warning"))
            specs += mcp_specs

        async def execute(name: str, args: dict) -> ToolOutcome:
            if name == "web_search":
                return ToolOutcome(await chat_tools.run_web_search(args.get("query", ""), args.get("max_results", 6)))
            if name in router:
                return ToolOutcome(await mcp_client.call_mcp(sessions, router, name, args))
            return ToolOutcome(f"Unknown tool: {name}", ok=False)

        return ToolSet(specs=specs, execute=execute)  # no mutating tools → no permission prompts

    @staticmethod
    async def _run_events(run_gen):
        """Adapt in-process AgentRunner events to the (type, data) tuples _render_run consumes —
        so the subagent pane renders through the same path as the main chat's SSE stream."""
        async for event in run_gen:
            yield (event.type, core_events.to_dict(event))

    @work(exit_on_error=False)
    async def _generate_side(self) -> None:
        """Answer the latest side-chat turn with the subagent's own model via the SHARED
        core.AgentRunner (same loop as the main chat + ACP) — no duplicated loop here."""
        sub, cfg = self._side_sub, self._side_cfg
        if sub is None or cfg is None:
            return
        transcript = self.query_one("#side-transcript", VerticalScroll)
        self._set_generating("side", True)
        try:
            engine = OpenAIEngine(
                self._side_base_url or cfg.base_url(), cfg.model,
                max_tokens=(sub.max_tokens or cfg.max_tokens
                            or scaled_max_tokens(cfg.model, self._context_cap_of(cfg))),
                sampling=self._sampling_of(cfg) or None,
            )
            messages = [{"role": m.role, "content": m.text} for m in self._side_messages]
            async with AsyncExitStack() as stack:
                tools = await self._subagent_toolset(sub, stack)
                runner = AgentRunner(
                    engine, tools=tools,
                    policy=RunPolicy(max_iters=8, max_tool_calls=8, native_tools=True),
                    system_note=self._subagent_system(sub) or None,
                    cancel=self._cancel_cb("side"),
                )
                answer = await self._render_run(transcript, self._run_events(runner.run(messages)),
                                                sub.name, show_reasoning=False)
            self._side_messages.append(ChatMessage(role="assistant", text=answer))
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Subagent failed: {exc}", severity="error", timeout=8)
        finally:
            self._set_generating("side", False)
            self._scroll_widget(transcript)

    def _current_project(self) -> Optional[Project]:
        if self.chat and self.chat.project_id:
            return store.get_project(self.data, self.chat.project_id)
        return None

    def _skill_instructions(self) -> Optional[str]:
        return skills.instructions_for(self.chat.skill_id) if self.chat else None
