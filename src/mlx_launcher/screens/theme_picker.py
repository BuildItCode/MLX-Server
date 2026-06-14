"""Color-scheme picker that previews each theme live as you move through the list."""

from __future__ import annotations

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView


class ThemeItem(ListItem):
    def __init__(self, name: str) -> None:
        super().__init__(Label(name))
        self.theme_name = name


class ThemeScreen(Screen):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._original: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Color scheme — ↑/↓ previews live · Enter keeps it · Esc cancels", classes="section")
        yield ListView(id="themes")
        yield Footer()

    def on_mount(self) -> None:
        self._original = self.app.theme
        names = sorted(self.app.available_themes.keys())
        lv = self.query_one("#themes", ListView)
        for name in names:
            lv.append(ThemeItem(name))
        if self._original in names:
            lv.index = names.index(self._original)
        lv.focus()

    @on(ListView.Highlighted, "#themes")
    def _preview(self, event: ListView.Highlighted) -> None:
        name = getattr(event.item, "theme_name", None)
        if name:
            self.app.theme = name  # live preview

    @on(ListView.Selected, "#themes")
    def _commit(self, event: ListView.Selected) -> None:
        name = getattr(event.item, "theme_name", None) or self.app.theme
        self.app.theme = name
        self.app.config.settings.theme = name
        self.app.save_config()
        self.app.pop_screen()
        self.notify(f"Theme: {name}")

    def action_cancel(self) -> None:
        if self._original:
            self.app.theme = self._original  # revert the preview
        self.app.pop_screen()
