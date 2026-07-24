"""The running-server view: address panel + live logs + stop/restart/Xcode.

The model server runs in the BACKEND now; this screen drives it over the wire — `start`/`stop`/
`restart` are client calls and the live logs + status arrive on the backend's log SSE stream."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog

from ..models import ServerConfig
from ..widgets.address_panel import AddressPanel

# status string (from the backend's SSE) → (display label shown by the address panel)
_IDLE, _STARTING, _READY, _ERROR, _STOPPED = "idle", "starting", "ready", "error", "stopped"


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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield AddressPanel(id="address")
        yield RichLog(id="log", markup=False, wrap=True, max_lines=4000)
        yield Footer()

    def on_mount(self) -> None:
        self._show(_STARTING, "starting…")
        self.run_worker(self._launch(), exclusive=True)

    def _show(self, status: str, message: str) -> None:
        # AddressPanel renders a status string + the address; mapping is purely presentational.
        self.query_one("#address", AddressPanel).show(status, self.cfg, message)

    async def _launch(self) -> None:
        client = await self.app.backend()
        try:
            await client.start_server(self.cfg.id)
        except Exception as exc:  # noqa: BLE001
            self._show(_ERROR, str(exc))
            self.notify(f"Failed to start: {exc}", severity="error")
            if "not found" in str(exc).lower():  # binary missing → point at setup
                from .setup import SetupScreen
                self.app.push_screen(SetupScreen())
            return
        self._stream_logs()

    @work(exit_on_error=False, exclusive=True)
    async def _stream_logs(self) -> None:
        # exclusive: a restart re-subscribes, so cancel the prior stream (and its SSE connection)
        # first instead of stacking a second subscriber on every restart.
        client = await self.app.backend()
        log = self.query_one("#log", RichLog)
        try:
            async for etype, data in client.stream_server_logs(self.cfg.id):
                if etype == "log":
                    prefix = "» " if data.get("stream") == "meta" else ""
                    log.write(prefix + data.get("line", ""))
                elif etype == "status":
                    status, message = data.get("status", _IDLE), data.get("message", "")
                    self._show(status, message)
                    if status == _READY:
                        self.notify(f"{self.cfg.name} ready at {self.cfg.base_url()}")
                    elif status == _ERROR and message:
                        self.notify(message, severity="error")
        except Exception:  # noqa: BLE001 — stream closed / backend gone
            pass

    # --- actions ---------------------------------------------------------

    def action_stop(self) -> None:
        self.run_worker(self._stop())

    async def _stop(self) -> None:
        client = await self.app.backend()
        await client.stop_server(self.cfg.id)

    def action_restart(self) -> None:
        self.query_one("#log", RichLog).clear()
        self._show(_STARTING, "restarting…")
        self.run_worker(self._restart(), exclusive=True)

    async def _restart(self) -> None:
        client = await self.app.backend()
        try:
            await client.restart_server(self.cfg.id)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Restart failed: {exc}", severity="error")
            return
        self._stream_logs()

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
        # the server keeps running in the backend; just leave the view
        self.app.pop_screen()
