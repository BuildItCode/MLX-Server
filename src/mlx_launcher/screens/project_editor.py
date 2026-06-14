"""Create or edit a project: name, working directory, and instructions.

The working directory is what turns a project into a coding workspace — when set,
chats in the project get file tools (read/write/edit/delete/run) scoped to it."""

from __future__ import annotations

import os
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, TextArea

from ..chat import store
from ..chat.models import ChatStoreFile, Project
from ..widgets.path_input import DropPathInput, resolve_path


class ProjectEditorScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, data: ChatStoreFile, project: Optional[Project] = None) -> None:
        super().__init__()
        self.data = data  # shared with the chat screen so edits are picked up
        self.project = project

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="project-form"):
            yield Label("Edit project" if self.project else "New project", classes="section")
            yield Label("Name")
            yield Input(id="p-name", placeholder="My app")
            yield Label("Working directory — drag a folder onto the terminal or type a path. "
                        "The model gets file tools (read/create/edit/delete/run) scoped here.")
            yield DropPathInput(id="p-cwd", placeholder="~/Code/my-app  (leave blank for no file access)")
            yield Label("", id="p-cwd-hint", classes="hint")
            yield Label("Instructions (optional) — added to the system prompt for every chat in this project")
            yield TextArea(id="p-instructions")
        with Horizontal(id="project-buttons"):
            yield Button("Save", id="p-save", variant="primary")
            yield Button("Cancel", id="p-cancel")
        yield Footer()

    def on_mount(self) -> None:
        if self.project:
            self.query_one("#p-name", Input).value = self.project.name
            self.query_one("#p-cwd", DropPathInput).value = self.project.working_dir or ""
            self.query_one("#p-instructions", TextArea).text = self.project.instructions or ""
        self._cwd_hint()
        self.query_one("#p-name", Input).focus()

    # --- working-directory input ----------------------------------------

    @on(DropPathInput.PathDropped)
    def _dropped(self, event: DropPathInput.PathDropped) -> None:
        self._set_cwd(event.path)

    def on_paste(self, event: events.Paste) -> None:
        # a folder dropped while focus isn't on the cwd field bubbles up here
        text = event.text.splitlines()[0] if event.text else ""
        resolved = resolve_path(text)
        if resolved and os.path.isdir(resolved):
            event.stop()
            self._set_cwd(resolved)

    def _set_cwd(self, path: str) -> None:
        self.query_one("#p-cwd", DropPathInput).value = path
        self._cwd_hint()

    @on(Input.Changed, "#p-cwd")
    def _cwd_changed(self) -> None:
        self._cwd_hint()

    def _cwd_hint(self) -> None:
        raw = self.query_one("#p-cwd", DropPathInput).value.strip()
        hint = self.query_one("#p-cwd-hint", Label)
        if not raw:
            hint.update("[dim]No working directory — file tools stay off for this project.[/]")
            return
        path = os.path.expanduser(raw)
        if os.path.isdir(path):
            hint.update("[#7fb069]✓ folder exists[/] — file tools will be enabled")
        else:
            hint.update("[#d19a66]folder will be created on save[/]")

    # --- save / cancel ---------------------------------------------------

    def _save(self) -> bool:
        name = self.query_one("#p-name", Input).value.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return False
        raw = self.query_one("#p-cwd", DropPathInput).value.strip()
        cwd: Optional[str] = None
        if raw:
            cwd = os.path.expanduser(raw)
            try:
                os.makedirs(cwd, exist_ok=True)
            except OSError as exc:
                self.notify(f"Can't create working directory: {exc}", severity="error")
                return False
        instructions = self.query_one("#p-instructions", TextArea).text.strip()
        proj = self.project or Project(name=name)
        proj.name = name
        proj.working_dir = cwd
        proj.instructions = instructions
        store.upsert_project(self.data, proj)
        store.save(self.data)
        return True

    @on(Button.Pressed, "#p-save")
    def _save_btn(self) -> None:
        if self._save():
            self.app.pop_screen()

    @on(Button.Pressed, "#p-cancel")
    def _cancel_btn(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        if self._save():
            self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()
