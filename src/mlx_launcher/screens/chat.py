"""A Claude-app-like chat front-end for talking to a running MLX server.

Sidebar of projects + chats, a streaming transcript that renders Markdown
(headings, lists, and syntax-highlighted code / JSON) live as it arrives, a
collapsible "thinking" block for reasoning models, a server/model picker, a
reasoning toggle, file attachments, a multiline prompt (Enter sends,
Shift+Enter / Ctrl+J for a newline), plus regenerate / edit-last / export."""

from __future__ import annotations

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

from ..chat import capabilities, mcp_client, store
from ..chat import tools as chat_tools
from ..chat.blocks import split_blocks
from ..chat.client import ChatClient, build_openai_messages
from ..chat.models import Attachment, Chat, ChatMessage, Project
from ..config.models import ServerConfig
from ..widgets.code_block import CodeBlock
from ..widgets.path_input import resolve_path


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

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self._prompt)
            with Horizontal(id="modal-buttons"):
                yield Button("Delete", id="yes", variant="error")
                yield Button("Cancel", id="no", variant="primary")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_no(self) -> None:
        self.dismiss(False)


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
    def __init__(self, project_id: Optional[str], label: str) -> None:
        super().__init__(Label(label))
        self.project_id = project_id


class ChatItem(ListItem):
    def __init__(self, chat: Chat, subtitle: str) -> None:
        super().__init__(Label(f"[b]{escape(chat.title)}[/]\n[dim]{escape(subtitle or '—')}[/]"))
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
                    yield Select([], id="server-select", prompt="server", allow_blank=True)
                yield VerticalScroll(id="transcript")
                with Horizontal(id="chat-toggles"):
                    yield Static(classes="actions-spacer")
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
                    yield Button("⊕", id="attach-btn")
                    yield Button("Send", id="send", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", PromptArea).border_title = "Enter to send · Shift+Enter for newline"
        self._refresh_servers()
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
        options = [(s.name, s.id) for s in self.app.config.servers]
        self.query_one("#server-select", Select).set_options(options)

    def _refresh_projects(self) -> None:
        lv = self.query_one("#projects", ListView)
        lv.clear()
        lv.append(ProjectItem(None, "All chats"))
        for p in self.data.projects:
            lv.append(ProjectItem(p.id, f"▪ {escape(p.name)}"))

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
        select.value = chat.server_id if chat.server_id in ids else Select.BLANK
        self._sync_reasoning_switch()
        self.query_one("#web", Switch).value = bool(self.chat and self.chat.web_search)
        self.query_one("#tools", Switch).value = bool(self.chat and self.chat.tools)
        self._update_topbar()
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
        vision = " · ◉ vision" if capabilities.supports_vision(self.chat.model) else ""
        if not self.chat.server_id and not self.chat.model:
            name = "[no server]"
        else:
            name = self._server_label_for(self.chat)
        self.query_one("#chat-title", Static).update(
            f"[b]{escape(self.chat.title)}[/]   [dim]{escape(name)}{vision}[/]"
        )

    # --- transcript / message widgets -----------------------------------

    def _assemble_row(self, label: str, role_class: str, body_widgets: list, *, right: bool = False) -> Horizontal:
        bubble = Vertical(Label(label, classes="msg-role"), *body_widgets, classes=f"msg {role_class}")
        spacer = Static(classes="msg-spacer")
        children = (spacer, bubble) if right else (bubble, spacer)
        return Horizontal(*children, classes="msg-row")

    def _bubble(self, role_label: str, role_class: str, body, *, right: bool = False) -> tuple[Horizontal, Static]:
        body_widget = Static(body, classes="msg-body")
        return self._assemble_row(role_label, role_class, [body_widget], right=right), body_widget

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
        text = escape(m.text)
        if m.attachments:
            chips = "  ".join(f"⊕ {escape(a.name or a.path)}" for a in m.attachments)
            text = f"{text}\n[dim]{chips}[/]" if text else f"[dim]{chips}[/]"
        row, _ = self._bubble("You", "msg-user", text or "[dim](empty)[/]", right=True)
        return row

    def _think_widget(self, text: str) -> Horizontal:
        row, _ = self._bubble("◌ thinking", "msg-think", escape(text))
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
        if not self.chat or event.value is Select.BLANK:
            return
        cfg = self._server_by_id(event.value)
        if cfg is None:
            return
        self.chat.server_id = cfg.id
        self.chat.base_url = cfg.base_url()
        self.chat.model = cfg.model
        self.chat.reasoning = self.chat.reasoning and capabilities.supports_reasoning(cfg.model)
        self._sync_reasoning_switch()
        self._update_topbar()
        self._persist()

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
            chips = "   ".join(f"⊕ {escape(a.name)} [dim]({a.kind})[/]" for a in self._pending)
            bar.update(f"{chips}   [dim]· ctrl+l clears[/]")
            bar.remove_class("hidden")
        else:
            bar.update("")
            bar.add_class("hidden")

    def _persist(self) -> None:
        if self.chat:
            store.upsert_chat(self.data, self.chat)
            store.save(self.data)

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
        ok = await self.app.push_screen_wait(ConfirmModal(f'Delete this {kind}?\n"{label}"'))
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

    @work
    async def _new_project(self) -> None:
        name = await self.app.push_screen_wait(TextPromptModal("New project name"))
        if not name:
            return
        project = Project(name=name)
        store.upsert_project(self.data, project)
        store.save(self.data)
        self.project_filter = project.id
        self._refresh_projects()
        self._refresh_chats()

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

    @work
    async def _generate(self) -> None:
        if self.chat and (self.chat.web_search or self.chat.tools):
            await self._generate_tools()
        else:
            await self._generate_stream()

    async def _generate_stream(self) -> None:
        assert self.chat is not None
        transcript = self.query_one("#transcript", VerticalScroll)
        assistant_box, assistant_body = self._bubble(self._server_label_for(self.chat), "msg-assistant", "[dim]…[/]")
        await transcript.mount(assistant_box)
        think_body: Optional[Static] = None

        client = ChatClient(self.chat.base_url, self.chat.model)
        messages = build_openai_messages(self.chat, self._current_project())
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
                        think_body.update(escape("".join(reason_acc)))
                elif kind == "content":
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
        client = ChatClient(self.chat.base_url, self.chat.model)
        messages = build_openai_messages(self.chat, self._current_project())
        self._cancel = False
        self._set_generating(True)
        t0 = time.monotonic()
        n_calls = 0
        final_text = ""
        try:
            async with AsyncExitStack() as stack:
                specs = []
                if self.chat.web_search:
                    specs.append(chat_tools.web_search_spec())
                sessions, router = {}, {}
                if self.chat.tools:
                    servers = store.load().mcp_servers  # pick up edits from the MCP manager
                    sessions, mcp_specs, router = await mcp_client.open_sessions(
                        stack,
                        servers,
                        on_error=lambda name, err: self.notify(f"MCP {name}: {err}", severity="warning"),
                    )
                    specs += mcp_specs
                for _ in range(8):
                    if self._cancel:
                        break
                    data = await client.bridge.chat(messages, tools=specs)
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    tool_calls = msg.get("tool_calls") or []
                    content = msg.get("content") or ""
                    if not tool_calls:
                        messages.append({"role": "assistant", "content": content})
                        final_text = content
                        break
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                    for call in tool_calls:
                        n_calls += 1
                        result = await self._exec_tool(call, sessions, router, transcript)
                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result[:8000]})
                    self._scroll_end()
        except Exception as exc:  # noqa: BLE001
            final_text = f"▲ {exc}"

        elapsed = time.monotonic() - t0
        widgets = self._assistant_body_widgets(final_text or "(no answer)")
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

    async def _exec_tool(self, call: dict, sessions: dict, router: dict, transcript) -> str:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        row, body = self._bubble("▸ tool", "msg-tool", f"[b]{escape(name)}[/]  [dim]{escape(json.dumps(args))[:80]}[/]")
        await transcript.mount(row)
        self._scroll_end()
        try:
            if name == "web_search":
                result = await chat_tools.run_web_search(args.get("query", ""), args.get("max_results", 6))
            elif name in router:
                result = await mcp_client.call_mcp(sessions, router, name, args)
            else:
                result = f"Unknown tool: {name}"
        except Exception as exc:  # noqa: BLE001
            result = f"tool error: {exc}"
        preview = result if len(result) <= 500 else result[:500] + " …"
        body.update(f"[b]{escape(name)}[/]\n[dim]{escape(preview)}[/]")
        return result

    def _current_project(self) -> Optional[Project]:
        if self.chat and self.chat.project_id:
            return store.get_project(self.data, self.chat.project_id)
        return None
