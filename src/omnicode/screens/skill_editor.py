"""Create or edit a custom skill (name, description, Markdown instruction body)."""

from __future__ import annotations

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, TextArea

from ..chat import skills


class SkillEditorScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, skill: Optional[skills.Skill] = None) -> None:
        super().__init__()
        # only custom skills are editable; anything else opens as a fresh "New"
        self.skill = skill if (skill and skill.is_custom) else None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="skill-form"):
            yield Label("Edit custom skill" if self.skill else "New custom skill", classes="section")
            yield Label("Name")
            yield Input(id="s-name", placeholder="my-style-guide")
            yield Label("Description — a one-liner on when to use it")
            yield Input(id="s-desc", placeholder="House code style for our Python services")
            yield Label("Instructions (Markdown) — injected as the system prompt when this skill is active")
            yield TextArea(id="s-body")
        with Horizontal(id="skill-buttons"):
            yield Button("Save", id="s-save", variant="primary")
            yield Button("Cancel", id="s-cancel")
        yield Footer()

    def on_mount(self) -> None:
        if self.skill:
            self.query_one("#s-name", Input).value = self.skill.name
            self.query_one("#s-desc", Input).value = self.skill.description
            self.query_one("#s-body", TextArea).text = self.skill.body()
        self.query_one("#s-name", Input).focus()

    def _save(self) -> bool:
        name = self.query_one("#s-name", Input).value.strip()
        desc = self.query_one("#s-desc", Input).value.strip()
        body = self.query_one("#s-body", TextArea).text.strip()
        if not name:
            self.notify("Name is required", severity="error")
            return False
        if not body:
            self.notify("Instructions body is required", severity="error")
            return False
        if self.skill:
            skills.update_custom_skill(self.skill, name, desc, body)
            self.notify(f"Updated “{name}”")
        else:
            skills.create_custom_skill(name, desc, body)
            self.notify(f"Created “{name}”")
        return True

    @on(Button.Pressed, "#s-save")
    def _save_btn(self) -> None:
        if self._save():
            self.app.pop_screen()

    @on(Button.Pressed, "#s-cancel")
    def _cancel_btn(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        if self._save():
            self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()
