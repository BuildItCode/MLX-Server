"""Manage MCP servers the chat models can call tools on (stdio or SSE)."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Select

from ..models import McpServer


class McpItem(ListItem):
    def __init__(self, server: McpServer) -> None:
        mark = ("●", "#7fb069") if server.enabled else ("○", "dim")
        detail = server.command if server.transport == "stdio" else server.url
        super().__init__(Label(Content.assemble(
            mark, " ", (server.name, "bold"), "  ", (f"{server.transport}: {detail or '—'}", "dim"))))
        self.server_id = server.id


class McpManagerScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("space", "toggle", "Enable/disable"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._servers: list[McpServer] = []  # cache fetched from the backend

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("MCP servers — tools the chat can call · Enter toggles · d deletes", classes="section")
        yield ListView(id="mcp-list")
        with Vertical(id="mcp-form"):
            yield Label("Add a server", classes="section")
            with Horizontal(classes="row"):
                with Vertical(classes="col"):
                    yield Label("Name")
                    yield Input(id="m-name", placeholder="my-tools")
                with Vertical(classes="col"):
                    yield Label("Transport")
                    yield Select([("stdio", "stdio"), ("sse", "sse")], value="stdio", allow_blank=False, id="m-transport")
            yield Label("Command (stdio)")
            yield Input(id="m-command", placeholder="uvx")
            yield Label("Args (stdio)")
            yield Input(id="m-args", placeholder="mcp-server-fetch")
            yield Label("Env (stdio, KEY=VALUE …)")
            yield Input(id="m-env")
            yield Label("URL (sse)")
            yield Input(id="m-url", placeholder="http://127.0.0.1:8000/sse")
            with Horizontal(id="mcp-buttons"):
                yield Button("Add server", id="m-add", variant="primary")
                yield Button("Back", id="m-back")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._reload())

    async def _reload(self) -> None:
        client = await self.app.backend()
        self._servers = [McpServer.model_validate(s) for s in await client.list_resource("mcp-servers")]
        self._refresh()

    def _refresh(self) -> None:
        lv = self.query_one("#mcp-list", ListView)
        lv.clear()
        if not self._servers:
            lv.append(ListItem(Label("[dim]No MCP servers yet — add one below[/]")))
            return
        for s in self._servers:
            lv.append(McpItem(s))

    def _by_id(self, sid):
        return next((s for s in self._servers if s.id == sid), None)

    @on(ListView.Selected, "#mcp-list")
    async def _toggle_selected(self, event: ListView.Selected) -> None:
        await self._toggle(getattr(event.item, "server_id", None))

    async def action_toggle(self) -> None:
        item = self.query_one("#mcp-list", ListView).highlighted_child
        await self._toggle(getattr(item, "server_id", None))

    async def _toggle(self, sid) -> None:
        s = self._by_id(sid)
        if not s:
            return
        s.enabled = not s.enabled
        client = await self.app.backend()
        await client.upsert_resource("mcp-servers", s.model_dump())
        await self._reload()

    async def action_delete(self) -> None:
        item = self.query_one("#mcp-list", ListView).highlighted_child
        sid = getattr(item, "server_id", None)
        if sid:
            client = await self.app.backend()
            await client.delete_resource("mcp-servers", sid)
            await self._reload()
            self.notify("Server removed")

    @on(Button.Pressed, "#m-add")
    async def _add(self) -> None:
        name = self.query_one("#m-name", Input).value.strip()
        transport = self.query_one("#m-transport", Select).value
        if not name:
            self.notify("Name is required", severity="error")
            return
        srv = McpServer(
            name=name,
            transport=transport,
            command=self.query_one("#m-command", Input).value.strip(),
            args=self.query_one("#m-args", Input).value.strip(),
            env=self.query_one("#m-env", Input).value.strip(),
            url=self.query_one("#m-url", Input).value.strip(),
        )
        if transport == "stdio" and not srv.command:
            self.notify("Command is required for stdio", severity="error")
            return
        if transport == "sse" and not srv.url:
            self.notify("URL is required for sse", severity="error")
            return
        client = await self.app.backend()
        await client.upsert_resource("mcp-servers", srv.model_dump())
        for fid in ("m-name", "m-command", "m-args", "m-env", "m-url"):
            self.query_one(f"#{fid}", Input).value = ""
        await self._reload()
        self.notify(f"Added MCP server “{name}”")

    @on(Button.Pressed, "#m-back")
    def _back_btn(self) -> None:
        self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()
