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

from ..chat import capabilities, fs_tools, knowledge, mcp_client, prompted_tools, skills, store, voice
from ..chat import tools as chat_tools
from ..chat.blocks import linkify_urls, split_blocks
from ..chat.client import (
    ChatClient,
    build_openai_messages,
    parse_harmony,
    prepend_system,
    recover_stripped_harmony,
    scaled_max_tokens,
)
from ..chat.tool_calls import extract_tool_calls
from ..chat.models import Attachment, Chat, ChatMessage, Project, Subagent
from ..config.models import ServerConfig
from ..server import discovery
from ..server.manager import BinaryNotFound, PortInUse, ServerStatus
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
    if name == "open_in_browser":
        return f"Open in browser  {args.get('path') or args.get('url', '?')}", ""
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
        self._recorder = None  # voice.Recorder while the mic is capturing, else None
        self._speaker = None   # voice.Speaker while reading a reply aloud, else None
        self._compacting = False  # a context-compaction summary is in flight (guards re-entry)

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

    def _side_cfg_for(self, base_cfg: ServerConfig) -> ServerConfig:
        """A copy of the subagent's profile on a port that won't collide with the main
        model — so both can be loaded at once. Returns base_cfg unchanged when its own
        port is already free and distinct; otherwise bumps to the next free port. All
        profiles default to :8080, so a bump is the common case."""
        reserved = {m.cfg.port for m in self.app.running_managers() if m.cfg.host == base_cfg.host}
        main_cfg = self._server_by_id(self.chat.server_id) if (self.chat and self.chat.server_id) else None
        if main_cfg is not None and main_cfg.host == base_cfg.host:
            reserved.add(main_cfg.port)
        port = base_cfg.port
        if port in reserved or not discovery.is_port_free(base_cfg.host, port):
            port = next((p for p in range(base_cfg.port + 1, base_cfg.port + 64)
                         if p not in reserved and discovery.is_port_free(base_cfg.host, p)), base_cfg.port)
        return base_cfg if port == base_cfg.port else base_cfg.model_copy(update={"port": port})

    @work(exclusive=True, group="side-open", exit_on_error=False)
    async def _open_side_chat(self, sub: Subagent) -> None:
        """Open a 50/50 side chat with `sub`: load its server on its own port (so the
        main model stays loaded) and route the input to it."""
        base_cfg = self._server_by_id(sub.server_id)
        if base_cfg is None:
            self.notify("This subagent has no model — edit it to choose one", severity="warning")
            return
        if self._side_open:
            await self._close_side_chat(unload=True)  # one side chat at a time → replace
        existing = self.app.get_manager(base_cfg.id)
        reuse = existing is not None and existing.is_running
        cfg = existing.cfg if reuse else self._side_cfg_for(base_cfg)

        self.query_one("#side-pane").remove_class("hidden")
        self._side_open = True
        self._side_sub = sub
        self._side_cfg = cfg
        self._side_messages = []
        self.query_one("#side-title", Static).update(
            Content.assemble(("↦ " + sub.name, "bold"), "  ", (cfg.model or "", "dim")))
        transcript = self.query_one("#side-transcript", VerticalScroll)
        transcript.remove_children()
        self._set_active_pane("side")
        self.query_one("#prompt", PromptArea).focus()

        if reuse:
            await transcript.mount(Static(plain(f"● {sub.name} ready · {cfg.name} (already running)"), classes="hint"))
            return
        await transcript.mount(Static(
            plain(f"Loading {cfg.name} on :{cfg.port} … model load can take a while."), classes="hint"))
        for mgr in self._port_occupants(cfg):  # only same host:port — never the main on a distinct port
            await mgr.stop()
        await self._load_server(cfg)
        mgr = self.app.get_manager(cfg.id)
        if mgr is None or not mgr.is_running or mgr.status is not ServerStatus.READY:
            if self._side_open and self._side_cfg is cfg:
                await transcript.mount(Static(plain("✕ failed to load — close and try again."), classes="hint"))
            return
        if self._side_open and self._side_cfg is cfg:  # still the active side chat (not closed/replaced)
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
            mgr = self.app.get_manager(cfg.id)
            if mgr is not None and mgr.is_running:
                self.notify(f"Unloading {sub.name if sub else cfg.name} …")
                try:
                    await mgr.stop()
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
            mgr = self.app.get_manager(cfg.id)
            if mgr is not None and mgr.is_running:
                try:
                    self.app.run_worker(mgr.stop(), exclusive=False)
                except Exception:  # noqa: BLE001 — app may be tearing down too
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

    def _client(self) -> ChatClient:
        """A chat client whose token budget respects the profile's --max-tokens (if the user set
        one) and otherwise scales with the context window (see scaled_max_tokens) — never the
        server's truncating 512-token default."""
        cfg = self._server_by_id(self.chat.server_id) if (self.chat and self.chat.server_id) else None
        max_tokens = (cfg.max_tokens if cfg and cfg.max_tokens
                      else scaled_max_tokens(self.chat.model, self._context_cap_of(cfg)))
        ctk = capabilities.reasoning_template_kwargs(self.chat.model, self.chat.reasoning_effort)
        return ChatClient(self.chat.base_url, self.chat.model, max_tokens=max_tokens,
                          chat_template_kwargs=ctk or None)

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
        self.query_one("#chip-web", ToggleChip).set_value(bool(self.chat and self.chat.web_search))
        self.query_one("#chip-tools", ToggleChip).set_value(bool(self.chat and self.chat.tools))
        self._sync_mode_chip()
        self.query_one("#chip-coding", ToggleChip).set_value(bool(self.chat and self.chat.coding))
        self._update_topbar()
        self._update_context_bar()
        self._render_transcript()
        self.query_one("#prompt", PromptArea).focus()

    def _sync_reasoning_switch(self) -> None:
        chip = self.query_one("#chip-reasoning", ToggleChip)
        supported = bool(self.chat and capabilities.supports_reasoning(self.chat.model))
        chip.set_enabled(supported)
        chip.set_value(bool(self.chat and self.chat.reasoning and supported))
        self._sync_effort_chip()

    _EFFORT_CYCLE = (None, "off", "low", "medium", "high")

    def _sync_effort_chip(self) -> None:
        """Reflect the reasoning-effort level; the chip shows only for reasoning models."""
        try:
            chip = self.query_one("#chip-effort", Static)
        except Exception:  # noqa: BLE001 — not mounted yet
            return
        supported = bool(self.chat and capabilities.supports_reasoning(self.chat.model))
        chip.set_class(not supported, "hidden")
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
        """Compact dim lines naming the tool calls an assistant turn made — for reloaded
        transcripts (live turns show richer per-call bubbles via _exec_tool)."""
        out: list = []
        for c in calls or []:
            args = json.dumps(c.get("arguments") or {})
            if len(args) > 80:
                args = args[:80] + " …"
            out.append(Static(Content.assemble(("▸ " + (c.get("name") or "tool"), "bold"), "  ", (args, "dim")),
                              classes="msg-stats"))
        return out

    def _message_widget(self, m: ChatMessage) -> Horizontal:
        if m.role == "tool":  # a persisted tool result — same compact bubble as the live one
            preview = m.text if len(m.text) <= 500 else m.text[:500] + " …"
            row, _ = self._bubble("▸ tool", "msg-tool",
                                  Content.assemble((m.tool_name or "tool", "bold"), "\n", (preview, "dim")))
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

    def _port_occupants(self, cfg: ServerConfig) -> list:
        """Running managers ACTUALLY bound to cfg's host:port (excluding cfg). Unlike
        _port_blockers it does NOT also stop the chat's main server — so a subagent on
        its own port loads without unloading the main model."""
        return [m for m in self.app.running_managers()
                if m.cfg.id != cfg.id and m.cfg.host == cfg.host and m.cfg.port == cfg.port]

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

    def action_back(self) -> None:
        pane = self._active_pane
        if self._gen.get(pane, False):  # Esc stops the focused pane's reply, not the screen
            self._cancel_flags[pane] = True
            self.notify("Stopped")
            return
        self.app.pop_screen()

    def action_stop(self) -> None:
        pane = self._active_pane
        if self._gen.get(pane, False):
            self._cancel_flags[pane] = True

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
        mgr = self.app.get_manager(self._side_cfg.id)
        if mgr is None or not mgr.is_running:
            self.notify("Subagent server isn't running — reopen its chat", severity="warning")
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

    def _compaction_messages(self) -> list[dict]:
        """The request that asks the model to summarize the conversation. Plan/coding framing is
        dropped so it produces a summary, not a plan or a code review (skill/project context is
        harmless to keep), then the compact instruction is appended as the final user turn."""
        summary_chat = self.chat.model_copy(update={"mode": "build", "coding": False})
        messages = build_openai_messages(summary_chat, self._current_project(), self._skill_instructions())
        messages.append({"role": "user", "content": _COMPACT_INSTRUCTIONS})
        return messages

    def _start_compaction(self, *, auto: bool) -> None:
        """Kick off a compaction summary, unless one (or a reply) is already running."""
        if self._gen.get("main", False) or self._compacting:
            if not auto:
                self.notify("Busy — wait for the current reply to finish.", severity="warning")
            return
        self._compaction_worker(auto)

    @work(exit_on_error=False, group="compact")
    async def _compaction_worker(self, auto: bool) -> None:
        """Ask the model to summarize the conversation, then replace the history with that summary —
        a user→assistant pair so role-alternating templates stay valid. Manual via /compact, or
        automatic when context passes 95%."""
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
        self._set_generating("main", True)  # shows Stop; Esc cancels via the main cancel flag
        cancel = self._cancel_cb("main")
        thinking = await self._mount_thinking(transcript, "Compacting context…")
        summary, error = "", None
        try:
            client = self._client()
            async for kind, chunk in client.stream(self._compaction_messages(), cancel=cancel):
                if kind == "content":
                    summary += chunk
        except Exception as exc:  # noqa: BLE001 — surface, but always clean up below
            error = exc
        finally:
            try:
                await thinking.remove()
            except Exception:  # noqa: BLE001
                pass
            self._compacting = False
            self._set_generating("main", False)

        if error is not None:
            self.notify(f"Compaction failed: {error}", severity="error", timeout=8)
            return
        if cancel():
            self.notify("Compaction stopped.")
            return
        summary = summary.strip()
        if not summary:
            self.notify("Compaction produced no summary — leaving the history as-is.", severity="warning")
            return
        chat.messages = [
            ChatMessage(role="user", text=_COMPACT_USER_MARKER),
            ChatMessage(role="assistant", text=summary),
        ]
        chat.updated = chat.messages[-1].ts
        self._render_transcript()
        self._persist()
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

    @work(exit_on_error=False)
    async def _generate(self) -> None:
        try:
            if self.chat and (self.chat.web_search or self.chat.tools or self._fs_root()):
                await self._generate_tools()
            else:
                await self._generate_stream()
        except Exception as exc:  # noqa: BLE001 — a worker error must not wedge the UI
            self.notify(f"Generation failed: {exc}", severity="error", timeout=8)
        finally:
            # always clear the per-pane flag, so the Send button can never get stuck on "Stop"
            self._set_generating("main", False)
            self._maybe_autocompact()  # between runs only — auto-summarize if context is nearly full

    # --- subagents (side chat) -------------------------------------------

    def _subagent_system(self, sub: Subagent) -> str:
        parts = [p for p in (skills.instructions_for(sid) for sid in sub.skill_ids) if p]
        if sub.system_prompt.strip():
            parts.append(sub.system_prompt.strip())
        kb = knowledge.load_knowledge(list(getattr(sub, "knowledge_paths", []) or []))
        if kb:  # uploaded docs — always in the subagent's context
            parts.append(kb)
        return "\n\n---\n\n".join(parts)

    def _side_openai_messages(self, sub: Subagent) -> list[dict]:
        """The side conversation as OpenAI messages, seeded by the subagent's system
        prompt (its skills + instructions). Text-only — attachments stay on the main."""
        msgs: list[dict] = []
        system = self._subagent_system(sub)
        if system:
            msgs.append({"role": "system", "content": system})
        for m in self._side_messages:
            msgs.append({"role": m.role, "content": m.text})
        return msgs

    @work(exit_on_error=False)
    async def _generate_side(self) -> None:
        """Answer the latest side-chat turn with the subagent's model: stream when it
        has no tools, else run a compact tool loop (web_search + its MCP connections)."""
        sub, cfg = self._side_sub, self._side_cfg
        if sub is None or cfg is None:
            return
        transcript = self.query_one("#side-transcript", VerticalScroll)
        client = ChatClient(cfg.base_url(), cfg.model,
                            max_tokens=(sub.max_tokens or cfg.max_tokens
                                        or scaled_max_tokens(cfg.model, self._context_cap_of(cfg))))
        cancel = self._cancel_cb("side")
        self._set_generating("side", True)
        try:
            if sub.web_search or (sub.tools and sub.mcp_server_ids):
                await self._side_tool_turn(sub, client, transcript, cancel=cancel)
            else:
                msg = await self._stream_into(transcript, client, self._side_openai_messages(sub),
                                              sub.name, show_reasoning=False, cancel=cancel)
                self._side_messages.append(msg)
        finally:
            self._set_generating("side", False)
            self._scroll_widget(transcript)

    async def _side_tool_turn(self, sub: Subagent, client: ChatClient, transcript, *, cancel) -> None:
        """One tool-using turn for the side chat: a single 'thinking' spinner while the
        subagent searches / calls its MCP tools, then render + persist the answer."""
        messages = self._side_openai_messages(sub)
        thinking = await self._mount_thinking(transcript, sub.name)
        t0 = time.monotonic()
        try:
            text = await self._subagent_tool_loop(sub, client, messages, cancel=cancel)
        except Exception as exc:  # noqa: BLE001
            text = f"▲ {exc}"
            if _is_fatal_generation_error(exc):
                text += ("\n\n*The model server failed to generate (this comes from MLX, not the "
                         "launcher). It usually means this model + engine combo can't run — try "
                         "switching the subagent's profile to the **mlx-lm** engine, picking a "
                         "different model, or lowering its max-tokens / KV-cache settings.*")
        try:
            await thinking.remove()
        except Exception:  # noqa: BLE001
            pass
        if text is None:  # Stop pressed
            text = "(stopped)"
        elapsed = time.monotonic() - t0
        widgets = self._assistant_body_widgets(text)
        widgets.append(Static(f"▸ {elapsed:.1f}s", classes="msg-stats"))
        if text and not text.startswith(("(stopped)", "▲ ")):
            widgets.append(self._copy_control(text))
        await transcript.mount(self._assemble_row(sub.name, "msg-assistant", widgets))
        self._side_messages.append(ChatMessage(role="assistant", text=text, elapsed=round(elapsed, 1)))
        self._scroll_widget(transcript)

    async def _exec_subagent_tool(self, name: str, args: dict, sessions: dict, router: dict) -> str:
        try:
            if name == "web_search":
                return await chat_tools.run_web_search(args.get("query", ""), args.get("max_results", 6))
            if name in router:
                return await mcp_client.call_mcp(sessions, router, name, args)
            return f"Unknown tool: {name}"
        except Exception as exc:  # noqa: BLE001
            return f"tool error: {exc}"

    @staticmethod
    def _subagent_tool_directive(sub: Subagent, specs: list) -> str:
        """A system note that DESCRIBES the tools in the prompt and insists on using
        them. Native `tools` are also sent, but many MLX servers (notably mlx-vlm)
        silently ignore that param — then the model never sees the tool and answers
        from memory, fabricating sources. Putting the tools (and a use-them rule) in
        the prompt works regardless of native support."""
        parts = [prompted_tools.tool_instructions(specs)]
        if sub.web_search:
            parts.append(
                "IMPORTANT: when the user asks for current information, facts, statistics, "
                "news, prices, or sources, you MUST call web_search first and base your answer "
                "on the results. NEVER fabricate sources, URLs, citations, or data — if you "
                "need a source, search for it and cite the URLs the tool returns."
            )
        return "\n\n".join(parts)

    async def _subagent_tool_loop(self, sub: Subagent, client: ChatClient, messages: list[dict],
                                  *, cancel=None) -> Optional[str]:
        """Compact agentic loop for the side chat: web_search + the subagent's selected
        MCP tools. The tools are described in the prompt AND offered natively, and we
        accept native / harmony / text-protocol calls alike — so it works whether or not
        the server supports the native `tools` param. Returns the answer, or None on Stop.
        `messages` is the running conversation (mutated in place)."""
        stopped = cancel or (lambda: False)
        async with AsyncExitStack() as stack:
            specs: list = []
            if sub.web_search:
                specs.append(chat_tools.web_search_spec())
            sessions, router = {}, {}
            if sub.tools and sub.mcp_server_ids:
                # force-enable the subagent's chosen MCP servers (independent of the global
                # Connectors toggle); reload from disk so edits in the MCP manager apply.
                wanted = [s.model_copy(update={"enabled": True})
                          for s in store.load().mcp_servers if s.id in sub.mcp_server_ids]
                sessions, mcp_specs, router = await mcp_client.open_sessions(
                    stack, wanted, on_error=lambda n, e: self.notify(f"MCP {n}: {e}", severity="warning"))
                specs += mcp_specs
            if specs:  # make the tools visible in the prompt + insist on using them
                prepend_system(messages, self._subagent_tool_directive(sub, specs))
            tool_names = [(s.get("function") or {}).get("name") for s in specs]
            send_native = bool(specs)  # also offer them natively; drop on template rejection
            data: dict = {}
            n_calls = 0  # cap total tool calls so a runaway searcher can't loop forever
            for _ in range(8):
                if stopped():
                    return None
                if n_calls >= 8:
                    break  # → wrap-up below forces a final answer
                try:
                    data = await self._bridge_chat(
                        client, messages, specs if send_native else None, cancel=cancel)
                except Exception as exc:  # template rejects the `tools` param → text protocol only
                    if send_native and not _is_fatal_generation_error(exc):
                        send_native = False
                        continue
                    raise
                if data is None:
                    return None
                # same format-agnostic extraction as the main loop (native / Harmony / MiniMax / …)
                ext = extract_tool_calls((data.get("choices") or [{}])[0].get("message") or {}, None, tool_names)
                if not ext.calls:
                    return prompted_tools.strip_tool_calls(ext.content) or ext.content
                n_calls += len(ext.calls)
                if ext.is_native:  # echo as a tool-call turn, results as the `tool` role
                    messages.append({"role": "assistant", "content": ext.content or None, "tool_calls": ext.native})
                    for raw_call, call in zip(ext.native, ext.calls):
                        result = await self._exec_subagent_tool(call["name"], call["arguments"], sessions, router)
                        messages.append({"role": "tool", "tool_call_id": raw_call.get("id", ""), "content": result[:8000]})
                else:  # text protocol — echo the model's own markup, results as user turns
                    messages.append({"role": "assistant",
                                     "content": self._tool_call_echo(ext.content, ext.reason, ext.calls)})
                    for call in ext.calls:
                        result = await self._exec_subagent_tool(call["name"], call["arguments"], sessions, router)
                        messages.append({"role": "user", "content": prompted_tools.tool_response(call["name"], result[:8000])})
            # ran out of iterations still calling tools → one no-tools turn for an answer
            if stopped():
                return None
            messages.append({"role": "user", "content": _WRAP_UP_PROMPT})
            data = await self._bridge_chat(client, messages, None, cancel=cancel)
            if data is None:
                return None
            content, _ = parse_harmony((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
            return prompted_tools.strip_tool_calls(content) or content

    async def _bridge_chat(self, client: ChatClient, messages: list, specs, *, cancel=None) -> Optional[dict]:
        """A non-streaming completion that aborts within ~0.1s when the user hits Stop
        (the raw call blocks for the whole response, so we poll `cancel` and cancel the
        request). Returns the response, or None if cancelled; bridge errors propagate to
        the caller. `cancel` is a per-pane predicate (defaults to never-cancel)."""
        stopped = cancel or (lambda: False)
        task = asyncio.ensure_future(client.bridge.chat(messages, tools=specs or None))
        while not task.done():
            if stopped():
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
        client = self._client()
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        self._set_generating("main", True)
        msg = await self._stream_into(transcript, client, messages,
                                      self._server_label_for(self.chat), show_reasoning=self.chat.reasoning,
                                      cancel=self._cancel_cb("main"))
        self.chat.messages.append(msg)
        self.chat.updated = msg.ts
        self._persist()
        self._set_generating("main", False)
        try:
            self._refresh_chats()
        except Exception:  # noqa: BLE001
            pass
        self._scroll_end()
        self._maybe_autoread()

    async def _stream_into(self, transcript, client: ChatClient, messages: list, label: str,
                           *, show_reasoning: bool, cancel=None) -> ChatMessage:
        """Stream a completion into `transcript`, returning the assistant ChatMessage
        (the caller persists it). Shared by the main chat and the subagent side chat;
        honors the per-pane `cancel` predicate and recovers token-stripped Harmony leaks."""
        stopped = cancel or (lambda: False)
        # The answer bubble animates ("Thinking…") until the first content token,
        # then we stop the spinner and stream the reply into the same widget.
        assistant_body = ThinkingIndicator(classes="msg-body thinking-indicator")
        assistant_box = self._assemble_row(label, "msg-assistant", [assistant_body])
        await transcript.mount(assistant_box)
        think_body: Optional[Static] = None
        reason_acc: list[str] = []
        content_acc: list[str] = []
        last_render = 0
        tokens = 0
        t_first: Optional[float] = None
        errored = False
        try:
            async for kind, chunk in client.stream(messages, cancel=stopped):
                if kind in ("reason", "content"):
                    if t_first is None:
                        t_first = time.monotonic()
                    tokens += 1
                if kind == "reason":
                    reason_acc.append(chunk)
                    if show_reasoning:
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
                self._scroll_widget(transcript)
        except Exception as exc:  # noqa: BLE001
            errored = True
            try:
                assistant_body.stop()
                assistant_body.update(f"[#e06c75]▲ {escape(str(exc))}[/]")
            except Exception:  # noqa: BLE001
                pass

        final = "".join(content_acc)
        # Some gpt-oss servers strip the Harmony <|...|> tokens but leak the channel
        # names ('analysis…assistantfinal…') into the stream — recover the clean answer
        # and move the reasoning to the thinking panel.
        if not errored:
            recovered = recover_stripped_harmony(final)
            if recovered is not None:
                final, leaked_reason = recovered
                if leaked_reason:
                    reason_acc.append(leaked_reason)
                    if show_reasoning and think_body is None:
                        think_box, think_body = self._bubble("◌ thinking", "msg-think", plain(leaked_reason))
                        await transcript.mount(think_box, before=assistant_box)
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
                await bubble.mount(self._copy_control(final))
            except Exception:  # noqa: BLE001
                pass
        self._scroll_widget(transcript)
        return ChatMessage(
            role="assistant",
            text=final,
            reasoning="".join(reason_acc),
            tps=round(tps, 1) if tps > 0 else None,
            n_tokens=tokens or None,
            elapsed=round(elapsed, 1) if elapsed > 0 else None,
        )

    def _continue_if_truncated(self, messages: list, clean: str, finish: Optional[str]) -> bool:
        """If the model's turn was cut off at the token limit (finish_reason == 'length') with
        partial answer text and no tool call, push the partial + a 'continue' nudge so it resumes,
        instead of the loop misreading a truncated turn as finished. Returns True when it did so."""
        if finish != "length" or not (clean or "").strip():
            return False
        messages.append({"role": "assistant", "content": clean})
        messages.append({"role": "user", "content": _CONTINUE_TRUNCATED_PROMPT})
        return True

    @staticmethod
    def _tool_call_echo(content: str, reason: str, calls: list[dict]) -> str:
        """The assistant turn to put back in the in-flight history for a recovered tool call.

        MiniMax emits its own ``<minimax:tool_call>`` XML; if we strip it and re-render the turn as
        Hermes ``<tool_call>`` JSON, MiniMax sees its own actions in a foreign dialect and drifts
        into a half-remembered format on later calls. So we echo MiniMax's native XML VERBATIM to
        keep it in that dialect. Everything else is rebuilt as a clean prose + ``<tool_call>`` turn:
        that nudges a drifted/loose form back toward the instructed protocol and avoids gpt-oss
        Harmony's raw ``<|...|>`` control tokens (which nest channels and confuse the template)."""
        c = content or ""
        if "<minimax:tool_call>" in c or "<invoke name=" in c:
            return content  # MiniMax's native XML — keep it as-is
        prose = (prompted_tools.strip_tool_calls(content) or reason or "").strip()
        tags = "\n".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)
        return f"{prose}\n{tags}".strip() if prose else tags

    def _persist_tool_turn(self, prose: str, calls: list[dict]) -> None:
        """Record an agentic tool-call turn in the chat history so a follow-up ('continue') keeps
        the context. `calls` are normalized [{name, arguments}] dicts."""
        if self.chat is not None:
            self.chat.messages.append(ChatMessage(
                role="assistant", text=(prose or "").strip(),
                tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in calls]))

    def _persist_tool_result(self, name: str, result: str) -> None:
        """Record a tool result in the chat history (mirrors what's fed back to the model)."""
        if self.chat is not None:
            self.chat.messages.append(ChatMessage(role="tool", tool_name=name, text=result[:8000]))

    async def _run_tool_calls(self, ext, messages: list, sessions: dict, router: dict,
                              transcript, fs_root: Optional[str]) -> int:
        """Echo the model's tool-call turn, execute each call, feed the results back, and persist —
        for both protocols. Structured `tool_calls` → echo native + results as the `tool` role;
        text-recovered calls → echo the model's own markup (MiniMax XML kept verbatim) + results as
        user <tool_response>. Mutates `messages`; returns the number of calls executed."""
        if ext.is_native:
            messages.append({"role": "assistant", "content": ext.content or None, "tool_calls": ext.native})
            self._persist_tool_turn(ext.content, ext.calls)
            for raw_call, call in zip(ext.native, ext.calls):
                result = await self._exec_tool(call["name"], call["arguments"], sessions, router, transcript, fs_root)
                messages.append({"role": "tool", "tool_call_id": raw_call.get("id", ""), "content": result[:8000]})
                self._persist_tool_result(call["name"], result)
        else:
            messages.append({"role": "assistant", "content": self._tool_call_echo(ext.content, ext.reason, ext.calls)})
            self._persist_tool_turn(prompted_tools.strip_tool_calls(ext.content) or ext.reason, ext.calls)
            for call in ext.calls:
                result = await self._exec_tool(call["name"], call["arguments"], sessions, router, transcript, fs_root)
                messages.append({"role": "user", "content": prompted_tools.tool_response(call["name"], result[:8000])})
                self._persist_tool_result(call["name"], result)
        return len(ext.calls)

    async def _generate_tools(self) -> None:
        """Function-calling loop: offer web_search + connected MCP tools, execute the
        model's tool calls, then render the final answer. The whole tool exchange is persisted
        into the chat history (as assistant tool-call turns + `tool` results) so a follow-up turn
        keeps the work context instead of starting over."""
        assert self.chat is not None
        transcript = self.query_one("#transcript", VerticalScroll)
        client = self._client()
        messages = build_openai_messages(self.chat, self._current_project(), self._skill_instructions())
        fs_root = self._fs_root()
        if fs_root:
            # ONE leading system message only — see prepend_system (templates 500 on two)
            prepend_system(messages, fs_tools.system_note(fs_root))
        cancel = self._cancel_cb("main")
        self._set_generating("main", True)
        t0 = time.monotonic()
        n_calls = 0
        final_text = ""
        truncating = False  # the last turn hit the token limit → accumulate, don't treat as final
        thinking = None  # the live "thinking…" bubble between turns (cleaned up below)
        max_iters = 24 if fs_root else 8
        # cap web/MCP search loops so a model that keeps searching (some batch several
        # web_search calls per turn) can't run away — then the wrap-up forces an answer.
        # File tools (coding) legitimately make many calls, so they're uncapped.
        max_tool_calls = None if fs_root else 8
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
                tool_names = [(s.get("function") or {}).get("name") for s in specs]
                # Describe the tools in the prompt (so the model SEES them even when the
                # server silently ignores the native `tools` param — mlx-vlm, and mlx_lm
                # with models it has no tool parser for) AND offer them natively. We accept
                # native / harmony / <tool_call> / stripped calls alike. `prompted` only
                # decides whether we keep SENDING the native param (dropped if it 500s).
                if specs:
                    prepend_system(messages, prompted_tools.tool_instructions(specs))
                # `prompted` only decides whether we keep SENDING the native tools param; the tools
                # are described in the prompt either way, and extract_tool_calls accepts native /
                # Harmony / MiniMax / <tool_call> / loose calls alike — so the loop is format-agnostic.
                prompted = (self.chat.server_id or "") in self._prompted_servers
                for _ in range(max_iters):
                    if cancel():
                        break
                    if max_tool_calls is not None and n_calls >= max_tool_calls:
                        self.notify(f"Reached the {max_tool_calls}-search limit — answering with what I found",
                                    severity="warning")
                        break  # → wrap-up below forces a final answer
                    thinking = await self._mount_thinking(transcript)
                    try:
                        data = await self._bridge_chat(client, messages, None if prompted else specs, cancel=cancel)
                    except Exception as exc:  # native tools rejected → switch this server to prompted
                        await thinking.remove()
                        thinking = None
                        # a fatal generation error (reshape/OOM) isn't a tools rejection —
                        # don't waste another long retry in prompted mode; surface it now.
                        if specs and not prompted and not _is_fatal_generation_error(exc):
                            prompted = True  # stop sending the native param (tools already described up front)
                            self._prompted_servers.add(self.chat.server_id or "")
                            self.notify("Native tool-calling failed — using prompted tools for this model.",
                                        severity="warning", timeout=8)
                            continue
                        raise
                    await thinking.remove()
                    thinking = None
                    if data is None:  # stopped
                        break
                    choice = (data.get("choices") or [{}])[0]
                    ext = extract_tool_calls(choice.get("message") or {}, choice.get("finish_reason"), tool_names)
                    if not ext.calls:  # a final answer — or a truncated turn we should continue
                        clean = prompted_tools.strip_tool_calls(ext.content) or ext.content
                        if self._continue_if_truncated(messages, clean, ext.finish):
                            final_text += clean
                            truncating = True
                            continue
                        final_text = (final_text + clean) if truncating else clean
                        truncating = False
                        messages.append({"role": "assistant", "content": ext.content})
                        break
                    truncating = False
                    final_text = ""  # the real answer comes after the tool work
                    n_calls += await self._run_tool_calls(ext, messages, sessions, router, transcript, fs_root)
                    self._scroll_end()
            # ran out of iterations still calling tools but never answered → one more
            # turn with NO tools so the user gets an answer, not "(no answer)".
            if not final_text and not cancel() and n_calls > 0:
                messages.append({"role": "user", "content": _WRAP_UP_PROMPT})
                thinking = await self._mount_thinking(transcript)
                try:
                    data = await self._bridge_chat(client, messages, None, cancel=cancel)
                finally:
                    if thinking is not None:
                        try:
                            await thinking.remove()
                        except Exception:  # noqa: BLE001
                            pass
                        thinking = None
                if data:
                    content, _ = parse_harmony((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
                    final_text = prompted_tools.strip_tool_calls(content) or content
        except Exception as exc:  # noqa: BLE001
            final_text = f"▲ {exc}"
        if thinking is not None:  # never leave a spinner spinning
            try:
                await thinking.remove()
            except Exception:  # noqa: BLE001
                pass

        elapsed = time.monotonic() - t0
        widgets = self._assistant_body_widgets(final_text or ("(stopped)" if cancel() else "(no answer)"))
        plural = "" if n_calls == 1 else "s"
        widgets.append(Static(f"▸ {n_calls} tool call{plural} · {elapsed:.1f}s", classes="msg-stats"))
        if final_text:
            widgets.append(self._copy_control(final_text))
        await transcript.mount(self._assemble_row(self._server_label_for(self.chat), "msg-assistant", widgets))
        self.chat.messages.append(
            ChatMessage(role="assistant", text=final_text, n_tokens=n_calls or None, elapsed=round(elapsed, 1))
        )
        self.chat.updated = self.chat.messages[-1].ts
        self._persist()
        self._set_generating("main", False)
        try:
            self._refresh_chats()
        except Exception:  # noqa: BLE001
            pass
        self._scroll_end()
        self._maybe_autoread()

    async def _exec_tool(self, name: str, args: dict, sessions: dict, router: dict, transcript, fs_root: Optional[str] = None) -> str:
        row, body = self._bubble("▸ tool", "msg-tool",
                                 Content.assemble((name, "bold"), "  ", (json.dumps(args)[:80], "dim")))
        await transcript.mount(row)
        self._scroll_end()
        try:
            if name == "web_search":
                result = await chat_tools.run_web_search(args.get("query", ""), args.get("max_results", 6))
            elif fs_root and name in fs_tools.FS_TOOL_NAMES:
                auto = self._auto_approve_fs or (self.chat is not None and self.chat.mode == "auto")
                if name in fs_tools.MUTATING_TOOLS and not auto:
                    summary, detail = _perm_prompt(name, args)
                    decision = await self.app.push_screen_wait(PermissionModal(summary, detail))
                    if decision == "all":
                        self._auto_approve_fs = True
                    elif decision != "once":
                        body.update(Content.assemble((name, "bold"), "\n", ("✕ denied by the user", "#e06c75")))
                        return "The user DENIED this action. Do not retry it; ask how to proceed."
                if name == "open_in_browser":
                    # must run on the UI thread (App.open_url), unlike the threaded fs ops
                    result = self._open_in_browser(fs_root, args)
                else:
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

    def _open_in_browser(self, fs_root: str, args: dict) -> str:
        """Resolve the model's target (a file in the working dir, or an http(s) URL) and open it
        in the OS browser via the app. Confined to the working directory for local files."""
        target = args.get("path") or args.get("url") or ""
        try:
            url = fs_tools.resolve_browser_target(fs_root, target)
        except ValueError as exc:
            return f"error: {exc}"
        try:
            self.app.open_url(url)
        except Exception as exc:  # noqa: BLE001
            return f"error: couldn't open the browser: {exc}"
        return f"Opened {url} in the browser."

    def _current_project(self) -> Optional[Project]:
        if self.chat and self.chat.project_id:
            return store.get_project(self.data, self.chat.project_id)
        return None

    def _skill_instructions(self) -> Optional[str]:
        return skills.instructions_for(self.chat.skill_id) if self.chat else None
