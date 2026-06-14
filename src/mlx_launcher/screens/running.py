"""The running-server view: address panel + live logs + stop/restart/Xcode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog

from ..config.models import ServerConfig
from ..server.manager import BinaryNotFound, PortInUse, ServerStatus
from ..widgets.address_panel import AddressPanel


class RunningScreen(Screen):
    BINDINGS = [
        Binding("s", "stop", "Stop"),
        Binding("r", "restart", "Restart"),
        Binding("c", "chat", "Chat"),
        Binding("x", "xcode", "Xcode"),
        Binding("y", "copy", "Copy URL"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, cfg: ServerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.manager = None
        self._token = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield AddressPanel(id="address")
        yield RichLog(id="log", markup=False, wrap=True, max_lines=4000)
        yield Footer()

    def on_mount(self) -> None:
        panel = self.query_one("#address", AddressPanel)
        existing = self.app.get_manager(self.cfg.id)
        if existing is not None and existing.is_running:
            self.manager = existing
            panel.show(existing.status, self.cfg, existing.status_message)
            self._subscribe(replay=True)
        else:
            self.manager = self.app.create_manager(self.cfg)
            panel.show(ServerStatus.IDLE, self.cfg, "")
            self._subscribe(replay=False)
            self.run_worker(self._start(), exclusive=True)

    def _subscribe(self, replay: bool) -> None:
        if replay:
            for stream, line in list(self.manager.log_buffer):
                self._append_log(stream, line)

        def on_log(stream: str, line: str) -> None:
            self._append_log(stream, line)

        def on_status(status: ServerStatus, message: str) -> None:
            self.query_one("#address", AddressPanel).show(status, self.cfg, message)
            if status is ServerStatus.READY:
                self.notify(f"{self.cfg.name} ready at {self.cfg.base_url()}")
            elif status is ServerStatus.ERROR and message:
                self.notify(message, severity="error")

        self._token = self.manager.subscribe(on_log, on_status)

    def _append_log(self, stream: str, line: str) -> None:
        prefix = "» " if stream == "meta" else ""
        self.query_one("#log", RichLog).write(prefix + line)

    async def _start(self) -> None:
        try:
            await self.manager.start()
        except BinaryNotFound:
            from ..server import discovery
            from .setup import SetupScreen

            binary = discovery.binary_name(self.cfg.engine)
            self.notify(f"{binary} not found — opening setup", severity="error")
            self.app.push_screen(SetupScreen())
        except PortInUse as exc:
            self.notify(str(exc), severity="error")
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Failed to start: {exc}", severity="error")

    def on_unmount(self) -> None:
        if self.manager is not None and self._token is not None:
            self.manager.unsubscribe(self._token)
            self._token = None

    # --- actions ---------------------------------------------------------

    def action_stop(self) -> None:
        if self.manager is not None:
            self.run_worker(self.manager.stop())

    def action_restart(self) -> None:
        self.run_worker(self._restart(), exclusive=True)

    async def _restart(self) -> None:
        if self.manager is not None:
            if self._token is not None:
                self.manager.unsubscribe(self._token)
                self._token = None
            await self.manager.stop()
        self.query_one("#log", RichLog).clear()
        self.manager = self.app.create_manager(self.cfg)
        self._subscribe(replay=False)
        await self._start()

    def action_chat(self) -> None:
        from .chat import ChatScreen

        self.app.push_screen(ChatScreen(server_id=self.cfg.id))

    def action_xcode(self) -> None:
        from .xcode_help import XcodeHelpScreen

        self.app.push_screen(XcodeHelpScreen(self.cfg))

    def action_copy(self) -> None:
        self.app.copy_text(self.cfg.base_url())
        self.notify(f"Copied {self.cfg.base_url()}")

    def action_back(self) -> None:
        if self.manager is not None and self.manager.is_running:
            self.notify(f"{self.cfg.name} still running in the background")
        self.app.pop_screen()
