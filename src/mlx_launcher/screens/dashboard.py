"""The main dashboard: saved server profiles + quick launch."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView

from ..config import store
from ..config.models import ServerConfig
from ..widgets.banner import Banner
from ..widgets.safe_content import title_sub


class ServerItem(ListItem):
    def __init__(self, server: ServerConfig) -> None:
        self.server = server
        sub = f"{server.model or '(no model set)'}  ·  {server.host}:{server.port}  ·  {server.engine}"
        super().__init__(Label(title_sub(server.name, sub)))


class DashboardScreen(Screen):
    BINDINGS = [
        Binding("n", "new", "New"),
        Binding("l", "launch", "Launch"),
        Binding("c", "chat", "Chat"),
        Binding("e", "edit", "Edit"),
        Binding("d", "delete", "Delete"),
        Binding("x", "xcode", "Xcode"),
        Binding("t", "theme", "Theme"),
        Binding("m", "mcp", "MCP servers"),
        Binding("k", "skills", "Skills"),
        Binding("p", "deps", "Dependencies"),
        Binding("g", "global_install", "Install globally"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Banner(id="banner")
        yield Label("Saved servers — Enter or l to launch · n new · e edit · d delete", classes="section")
        yield ListView(id="servers")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_list()

    def on_screen_resume(self) -> None:
        self.refresh_list()

    def refresh_list(self) -> None:
        lv = self.query_one("#servers", ListView)
        index = lv.index
        lv.clear()
        servers = self.app.config.servers
        if not servers:
            lv.append(ListItem(Label("[dim]No saved servers yet — press n to create one[/]")))
            return
        for s in servers:
            lv.append(ServerItem(s))
        if index is not None:
            lv.index = min(index, len(servers) - 1)
        lv.focus()

    def _selected(self) -> ServerConfig | None:
        item = self.query_one("#servers", ListView).highlighted_child
        return getattr(item, "server", None)

    # --- actions ---------------------------------------------------------

    def action_new(self) -> None:
        from .editor import EditorScreen

        self.app.push_screen(EditorScreen())

    def action_edit(self) -> None:
        s = self._selected()
        if s is None:
            self.notify("Select a server first", severity="warning")
            return
        from .editor import EditorScreen

        self.app.push_screen(EditorScreen(server=s))

    def action_delete(self) -> None:
        s = self._selected()
        if s is None:
            self.notify("Select a server first", severity="warning")
            return
        store.delete_server(self.app.config, s.id)
        self.app.save_config()
        self.refresh_list()
        self.notify(f"Deleted {s.name}")

    def action_launch(self) -> None:
        s = self._selected()
        if s is None:
            self.notify("Select a server first", severity="warning")
            return
        from .running import RunningScreen

        self.app.push_screen(RunningScreen(s))

    def action_xcode(self) -> None:
        s = self._selected()
        if s is None:
            self.notify("Select a server first", severity="warning")
            return
        from .xcode_help import XcodeHelpScreen

        self.app.push_screen(XcodeHelpScreen(s))

    def action_chat(self) -> None:
        from .chat import ChatScreen

        self.app.push_screen(ChatScreen())

    def action_theme(self) -> None:
        from .theme_picker import ThemeScreen

        self.app.push_screen(ThemeScreen())

    def action_mcp(self) -> None:
        from .mcp_manager import McpManagerScreen

        self.app.push_screen(McpManagerScreen())

    def action_skills(self) -> None:
        from .skills_manager import SkillsManagerScreen

        self.app.push_screen(SkillsManagerScreen())

    def action_deps(self) -> None:
        from .setup import SetupScreen

        self.app.push_screen(SetupScreen())

    def action_global_install(self) -> None:
        from .setup import SetupScreen

        self.app.push_screen(SetupScreen())

    @on(ListView.Selected, "#servers")
    def _on_selected(self, event: ListView.Selected) -> None:
        s = getattr(event.item, "server", None)
        if s is None:
            return
        from .running import RunningScreen

        self.app.push_screen(RunningScreen(s))
