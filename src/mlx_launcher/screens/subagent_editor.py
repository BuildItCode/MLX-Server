"""Create or edit a subagent: a named specialist with its own model (server
profile), system prompt, and capabilities (web search, MCP connections, skills).
Opened from the subagents menu as a side chat."""

from __future__ import annotations

import os
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, TextArea

from ..chat import skills, store
from ..chat.models import Subagent
from ..widgets.path_input import DropPathInput
from ..widgets.safe_content import plain
from ..widgets.toggle_chip import ToggleChip


class SubagentEditorScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, sub: Optional[Subagent] = None) -> None:
        super().__init__()
        self.sub = sub
        self.data = store.load()
        self._knowledge: list[str] = list(sub.knowledge_paths) if sub else []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="subagent-form"):
            yield Label("Edit subagent" if self.sub else "New subagent", classes="section")
            yield Label("Name — shown in the subagents menu")
            yield Input(id="sa-name", placeholder="Researcher")
            yield Label("Model — the server profile this subagent runs on")
            yield Select(
                [(plain(s.name), s.id) for s in self.app.config.servers],
                id="sa-server", prompt="server profile", allow_blank=True,
            )
            yield Label("System prompt — the specialist's instructions")
            yield TextArea(id="sa-prompt")
            yield Label("Capabilities")
            with Horizontal(classes="chip-row"):
                yield ToggleChip("web search", "web", id="sa-web")
                yield ToggleChip("MCP tools", "tools", id="sa-tools")
            yield Label("MCP connections it may use")
            with Vertical(id="sa-mcp"):
                if not self.data.mcp_servers:
                    yield Label(plain("none — add MCP servers with Ctrl+G in chat"), classes="hint")
                for m in self.data.mcp_servers:
                    yield ToggleChip(plain(m.name), f"mcp:{m.id}", classes="connector-chip")
            yield Label("Skills injected into its prompt")
            with Vertical(id="sa-skills"):
                for s in skills.all_skills():
                    yield ToggleChip(plain(s.name), f"skill:{s.id}", classes="connector-chip")
            yield Label("Knowledge base — docs always in its context (drag a file/folder onto the field, or paste a path)")
            with Horizontal(classes="row"):
                yield DropPathInput(id="sa-kb-path", placeholder="~/docs/handbook.md  or  a folder")
                yield Button("Add", id="sa-kb-add")
            yield Vertical(id="sa-kb-list")
            yield Label("Max tokens (optional)")
            yield Input(id="sa-maxtok", placeholder="e.g. 4096")
        with Horizontal(id="subagent-buttons"):
            yield Button("Save", id="sa-save", variant="primary")
            yield Button("Cancel", id="sa-cancel")
        yield Footer()

    def on_mount(self) -> None:
        if self.sub:
            self.query_one("#sa-name", Input).value = self.sub.name
            if self.sub.server_id:
                ids = {s.id for s in self.app.config.servers}
                self.query_one("#sa-server", Select).value = (
                    self.sub.server_id if self.sub.server_id in ids else Select.NULL)
            self.query_one("#sa-prompt", TextArea).text = self.sub.system_prompt
            self.query_one("#sa-web", ToggleChip).set_value(self.sub.web_search)
            self.query_one("#sa-tools", ToggleChip).set_value(self.sub.tools)
            self.query_one("#sa-maxtok", Input).value = "" if self.sub.max_tokens is None else str(self.sub.max_tokens)
            for chip in self.query(".connector-chip").results(ToggleChip):
                kind, _, ident = chip.key.partition(":")
                if (kind == "mcp" and ident in self.sub.mcp_server_ids) or \
                   (kind == "skill" and ident in self.sub.skill_ids):
                    chip.set_value(True)
        self._refresh_kb()
        self.query_one("#sa-name", Input).focus()

    def _selected(self, kind: str) -> list[str]:
        out = []
        for chip in self.query(".connector-chip").results(ToggleChip):
            k, _, ident = chip.key.partition(":")
            if k == kind and chip.value:
                out.append(ident)
        return out

    # --- knowledge base --------------------------------------------------

    def _refresh_kb(self) -> None:
        box = self.query_one("#sa-kb-list", Vertical)
        box.remove_children()
        if not self._knowledge:
            box.mount(Label(plain("none yet — PDFs, text, markdown, code; folders OK"),
                            classes="hint"))
            return
        for path in self._knowledge:
            btn = Button("✕", classes="kb-del")
            btn._kb_path = path  # an id can't hold a path → carry it on the widget
            box.mount(Horizontal(Static(plain(path), classes="kb-path"), btn, classes="kb-row"))

    def _add_knowledge(self, raw: str) -> None:
        path = os.path.expanduser((raw or "").strip())
        if not path:
            return
        if not os.path.exists(path):
            self.notify(f"Not found: {path}", severity="error")
            return
        if path not in self._knowledge:
            self._knowledge.append(path)
        self.query_one("#sa-kb-path", DropPathInput).value = ""
        self._refresh_kb()

    @on(Button.Pressed, "#sa-kb-add")
    def _kb_add_btn(self) -> None:
        self._add_knowledge(self.query_one("#sa-kb-path", DropPathInput).value)

    @on(Input.Submitted, "#sa-kb-path")
    def _kb_submit(self) -> None:
        self._add_knowledge(self.query_one("#sa-kb-path", DropPathInput).value)

    @on(DropPathInput.PathDropped)
    def _kb_dropped(self, event: DropPathInput.PathDropped) -> None:
        self._add_knowledge(event.path)

    @on(Button.Pressed, ".kb-del")
    def _kb_remove(self, event: Button.Pressed) -> None:
        path = getattr(event.button, "_kb_path", None)
        if path in self._knowledge:
            self._knowledge.remove(path)
            self._refresh_kb()

    def _save(self) -> bool:
        name = self.query_one("#sa-name", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return False
        server_val = self.query_one("#sa-server", Select).value
        server_id = None if server_val is Select.NULL else server_val
        if not server_id:
            self.notify("Pick a server profile (the subagent's model)", severity="error")
            return False
        maxtok_raw = self.query_one("#sa-maxtok", Input).value.strip()
        max_tokens = None
        if maxtok_raw:
            try:
                max_tokens = int(maxtok_raw)
            except ValueError:
                self.notify("Max tokens must be a whole number", severity="error")
                return False
        sub = self.sub or Subagent()
        sub.name = name
        sub.server_id = server_id
        sub.system_prompt = self.query_one("#sa-prompt", TextArea).text.strip()
        sub.web_search = self.query_one("#sa-web", ToggleChip).value
        sub.tools = self.query_one("#sa-tools", ToggleChip).value
        sub.mcp_server_ids = self._selected("mcp")
        sub.skill_ids = self._selected("skill")
        sub.knowledge_paths = list(self._knowledge)
        sub.max_tokens = max_tokens
        # Re-read before writing: chats.json is one document (chats/projects/subagents),
        # so saving a stale snapshot here would clobber edits made elsewhere meanwhile.
        fresh = store.load()
        store.upsert_subagent(fresh, sub)
        store.save(fresh)
        self.notify(f"Saved {name}")
        return True

    @on(Button.Pressed, "#sa-save")
    def _save_btn(self) -> None:
        if self._save():
            self.app.pop_screen()

    @on(Button.Pressed, "#sa-cancel")
    def _cancel_btn(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        if self._save():
            self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()
