"""Dependency check / install screen, and the global-install flow."""

from __future__ import annotations

import os

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from .. import bootstrap


class SetupScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(id="status")
            with Horizontal(id="setup-buttons"):
                yield Button("Install mlx-lm", id="install_mlx", variant="primary")
                yield Button("Install mlx-vlm", id="install_vlm", variant="primary")
                yield Button("Locate binary", id="locate")
                yield Button("Install globally", id="install_global", variant="success")
                yield Button("Back", id="back")
            yield Input(id="locate_path", placeholder="/path/to/mlx_lm.server", classes="hidden")
            yield RichLog(id="setup-log", markup=False, wrap=True, max_lines=4000)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        located = self.app.config.settings.mlx_server_path
        lines = []
        for engine, binary in (("mlx-lm", "mlx_lm.server"), ("mlx-vlm", "mlx_vlm.server")):
            path = bootstrap.find_mlx_server(engine)
            if path:
                lines.append(f"[#7fb069]✓ {binary} found[/]  [dim]{escape(path)}[/]")
            elif engine == "mlx-lm" and located and os.path.exists(located):
                lines.append(f"[#7fb069]✓ {binary} (located)[/]  [dim]{escape(located)}[/]")
            else:
                lines.append(f"[#e06c75]✗ {binary} not found[/] — install it, or locate an existing binary.")
        gi = bootstrap.global_install_argv()
        if gi is not None:
            lines.append("[dim]‘Install globally’ will run: " + escape(" ".join(gi)) + "[/]")
        elif bootstrap.pipx_available():
            lines.append("[dim]Global install: run from a source checkout to use pipx.[/]")
        else:
            lines.append("[dim]Global install: pipx not found — run ./install.sh (it bootstraps pipx or symlinks ~/.local/bin).[/]")
        self.query_one("#status", Static).update("\n".join(lines))

    # --- actions ---------------------------------------------------------

    @on(Button.Pressed, "#install_mlx")
    def _install_mlx(self) -> None:
        self.run_worker(self._run(bootstrap.pip_install_argv("mlx-lm")), exclusive=True)

    @on(Button.Pressed, "#install_vlm")
    def _install_vlm(self) -> None:
        self.run_worker(self._run(bootstrap.pip_install_argv("mlx-vlm")), exclusive=True)

    @on(Button.Pressed, "#install_global")
    def _install_global(self) -> None:
        argv = bootstrap.global_install_argv()
        if argv is None:
            self.notify("Run ./install.sh for global install (pipx not available here).", severity="warning")
            return
        self.run_worker(self._run(argv), exclusive=True)

    @on(Button.Pressed, "#locate")
    def _show_locate(self) -> None:
        field = self.query_one("#locate_path", Input)
        field.remove_class("hidden")
        field.focus()

    @on(Input.Submitted, "#locate_path")
    def _locate_submit(self, event: Input.Submitted) -> None:
        path = os.path.expanduser(event.value.strip())
        if not path:
            return
        if not os.path.isfile(path):
            self.notify("Not a file: " + path, severity="error")
            return
        self.app.config.settings.mlx_server_path = path
        self.app.save_config()
        self.notify(f"Using {path}")
        self.query_one("#locate_path", Input).add_class("hidden")
        self._refresh_status()

    @on(Button.Pressed, "#back")
    def _back_button(self) -> None:
        self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()

    async def _run(self, argv: list[str]) -> None:
        log = self.query_one("#setup-log", RichLog)
        log.write("$ " + " ".join(argv))
        code = await bootstrap.run_streamed(argv, lambda line: log.write(line))
        log.write(f"[exit code {code}]")
        if code == 0:
            self.notify("Done")
        else:
            self.notify(f"Command exited with code {code}", severity="error")
        self._refresh_status()
