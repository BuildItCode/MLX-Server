"""Browse, create, edit, delete, and install the skills the chat can use.

Skills carry an origin badge so the user can tell custom (user-created) skills
apart from bundled and BMAD ones."""

from __future__ import annotations

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, RichLog

from ..chat import skills

_BADGE = {
    skills.ORIGIN_CUSTOM: "[#7fb069]★ custom[/]",
    skills.ORIGIN_BMAD: "[#d19a66]◆ bmad[/]",
    skills.ORIGIN_BUNDLED: "[dim]• bundled[/]",
}


class SkillItem(ListItem):
    def __init__(self, skill: skills.Skill) -> None:
        badge = _BADGE.get(skill.origin, "")
        desc = (skill.description or "—").strip()
        super().__init__(Label(f"{badge}  [b]{escape(skill.name)}[/]\n[dim]{escape(desc[:120])}[/]"))
        self.skill = skill


class SkillsManagerScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("n", "new", "New"),
        Binding("e", "edit", "Edit"),
        Binding("d", "delete", "Delete"),
        Binding("b", "install_bmad", "Install BMAD"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            "Skills — pick one per chat from the chat's skill menu  ·  ★ custom · ◆ bmad · • bundled",
            classes="section",
        )
        yield ListView(id="skills-list")
        with Horizontal(id="skills-buttons"):
            yield Button("New custom", id="sk-new", variant="primary")
            yield Button("Edit", id="sk-edit")
            yield Button("Delete", id="sk-delete", variant="error")
            yield Button("Install BMAD", id="sk-bmad", variant="success")
            yield Button("Back", id="sk-back")
        yield RichLog(id="skills-log", markup=False, wrap=True, max_lines=2000, classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()

    def on_screen_resume(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lv = self.query_one("#skills-list", ListView)
        idx = lv.index
        lv.clear()
        items = skills.all_skills()
        if not items:
            lv.append(ListItem(Label("[dim]No skills found — create one with n, or install BMAD[/]")))
            return
        for s in items:
            lv.append(SkillItem(s))
        if idx is not None:
            lv.index = min(idx, len(items) - 1)

    def _selected(self):
        item = self.query_one("#skills-list", ListView).highlighted_child
        return getattr(item, "skill", None)

    # --- actions ---------------------------------------------------------

    @on(Button.Pressed, "#sk-new")
    def _new_btn(self) -> None:
        self.action_new()

    def action_new(self) -> None:
        from .skill_editor import SkillEditorScreen

        self.app.push_screen(SkillEditorScreen())

    @on(Button.Pressed, "#sk-edit")
    def _edit_btn(self) -> None:
        self.action_edit()

    def action_edit(self) -> None:
        s = self._selected()
        if s is None:
            return
        if not s.is_custom:
            self.notify("Only custom skills can be edited", severity="warning")
            return
        from .skill_editor import SkillEditorScreen

        self.app.push_screen(SkillEditorScreen(s))

    @on(Button.Pressed, "#sk-delete")
    def _delete_btn(self) -> None:
        self.action_delete()

    def action_delete(self) -> None:
        s = self._selected()
        if s is None:
            return
        if not s.is_custom:
            self.notify("Only custom skills can be deleted", severity="warning")
            return
        skills.delete_custom_skill(s)
        self.notify(f"Deleted “{s.name}”")
        self._refresh()

    @on(Button.Pressed, "#sk-bmad")
    def _bmad_btn(self) -> None:
        self.action_install_bmad()

    def action_install_bmad(self) -> None:
        log = self.query_one("#skills-log", RichLog)
        log.remove_class("hidden")
        log.clear()
        self.run_worker(self._install_bmad(log), exclusive=True)

    async def _install_bmad(self, log: RichLog) -> None:
        log.write("Downloading BMAD skills …")
        try:
            n = await skills.install_bmad(lambda line: log.write(line))
        except Exception as exc:  # noqa: BLE001
            self.notify(f"BMAD install failed: {exc}", severity="error")
            return
        self._refresh()
        if n:
            self.notify(f"Installed {n} BMAD skills — pick them from a chat's skill menu")
        else:
            self.notify("No BMAD skills installed — check your connection", severity="error")

    @on(Button.Pressed, "#sk-back")
    def _back_btn(self) -> None:
        self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()
