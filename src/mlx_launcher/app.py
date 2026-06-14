"""The Textual application: theme, global styles, screen wiring, and the
registry of running server managers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Optional

from textual.app import App
from textual.binding import Binding

from .config import store
from .config.models import ConfigFile, ServerConfig
from .server.manager import ServerManager
from .theme import MLX_THEME


class MlxLauncherApp(App):
    TITLE = "MLXS"
    SUB_TITLE = ""

    CSS = """
    .section { color: $accent; text-style: bold; padding: 1 1 0 1; }

    #servers { height: 1fr; border: round $panel; }
    #themes { height: 1fr; border: round $panel; }
    #log { height: 1fr; border: round $panel; padding: 0 1; }
    #setup-log { height: 1fr; border: round $panel; }

    .row { height: auto; }
    .col { width: 1fr; height: auto; }

    .hint { color: $text-muted; padding: 0 1 1 1; }
    .preview { color: $text-muted; padding: 1; }

    #buttons { height: auto; padding: 1 0; }
    #buttons Button { margin: 0 1 0 0; }
    #setup-buttons { height: auto; padding: 1 0; }
    #setup-buttons Button { margin: 0 1 0 0; }

    #form { height: 1fr; }
    .switch-row { height: auto; }
    .switch-row Label { padding: 1 0 0 1; }

    #provider, #acp, #test-result, #status { padding: 1 2; }
    #acp { border-top: solid $panel; }

    /* chat */
    #chat-body { height: 1fr; }
    #chat-sidebar { width: 34; border-right: solid $panel; }
    #chat-sidebar .section { color: $accent; text-style: bold; padding: 1 1 0 1; }
    #projects { height: 7; border: round $panel; }
    #chats { height: 1fr; border: round $panel; }
    #sidebar-buttons { height: auto; padding: 1 0 0 0; }
    #sidebar-buttons Button { width: 1fr; margin: 0 0 1 0; }
    #chat-main { width: 1fr; }
    #chat-topbar { height: auto; padding: 0 1; }
    #chat-title { width: 1fr; padding: 1 0 0 0; }
    #server-select { width: 34; }
    #skill-select { width: 40; }
    #reason-label { padding: 1 1 0 1; }
    #transcript { height: 1fr; padding: 0 1; }
    #attachments { height: auto; padding: 0 1; }
    #attach { margin: 0 1; }
    #chat-actions { height: auto; padding: 0 1; }
    #chat-actions Button { margin: 0 1 0 0; }
    #chat-inputrow { height: auto; padding: 0 1 1 1; }
    #prompt { width: 1fr; height: 6; border: round $panel; border-title-color: $accent; }
    #chat-inputrow Button { margin: 0 0 0 1; }
    #chat-toggles { height: auto; padding: 0 1; }
    .actions-spacer { width: 1fr; height: 1; }
    .toggle-label { padding: 1 1 0 1; color: $text-muted; }
    .ctx-bar { width: auto; padding: 1 1 0 0; }
    .msg-row { height: auto; width: 1fr; }
    .msg-spacer { width: 1fr; height: auto; }
    .msg { height: auto; width: 80%; max-width: 80%; margin: 1 0 0 0; padding: 0 0 0 1; border-left: solid $panel; }
    .msg-role { text-style: bold; width: auto; }
    .msg-body { height: auto; }
    /* model + thinking: fill up to 80% so markdown/code has room */
    .msg-assistant { border-left: solid $primary; }
    .msg-assistant .msg-role { color: $primary; }
    .msg-assistant .msg-body { width: 1fr; }
    .thinking-indicator { color: $accent; }
    .msg-think { border-left: solid $secondary; }
    .msg-think .msg-role { color: $secondary; }
    .msg-think .msg-body { width: 1fr; color: $text-muted; }
    /* user: hug content on the right, capped at 80% */
    .msg-user { width: auto; border-left: none; border-right: solid $accent; padding: 0 1 0 0; }
    .msg-user .msg-role { color: $accent; text-align: right; }
    .msg-user .msg-body { width: auto; }
    .msg-stats { width: 1fr; color: $text-muted; padding: 1 0 0 0; }
    /* copyable code blocks */
    .code-block { height: auto; width: 1fr; margin: 1 0; border: round $panel; }
    .code-head { height: 1; width: 1fr; background: $panel; }
    .code-lang { width: 1fr; color: $text-muted; padding: 0 1; }
    .code-copy { width: auto; color: $accent; padding: 0 1; }
    .code-copy:hover { background: $accent; color: $background; text-style: bold; }
    .code-body { width: 1fr; height: auto; padding: 0 1; }
    #tools-label { padding: 1 1 0 1; }
    .msg-tool { width: auto; border-left: solid $warning; }
    .msg-tool .msg-role { color: $warning; }
    .msg-tool .msg-body { width: auto; }
    #mcp-list { height: 1fr; border: round $panel; }
    #mcp-form { height: auto; padding: 0 1; }
    #mcp-buttons { height: auto; padding: 1 0; }
    #mcp-buttons Button { margin: 0 1 0 0; }
    #skills-list { height: 1fr; border: round $panel; }
    #skills-buttons { height: auto; padding: 1 0; }
    #skills-buttons Button { margin: 0 1 0 0; }
    #skills-log { height: 12; border: round $panel; }
    #skill-form { height: 1fr; padding: 0 1; }
    #s-body { height: 16; border: round $panel; }
    #skill-buttons { height: auto; padding: 1 0; }
    #skill-buttons Button { margin: 0 1 0 0; }
    #project-form { height: 1fr; padding: 0 1; }
    #p-instructions { height: 10; border: round $panel; }
    #project-buttons { height: auto; padding: 1 0; }
    #project-buttons Button { margin: 0 1 0 0; }
    TextPromptModal, ConfirmModal, PermissionModal { align: center middle; background: $background 60%; }
    #modal-box { width: 64; height: auto; padding: 1 2; border: round $primary; background: $surface; }
    #modal-buttons { height: auto; padding: 1 0 0 0; }
    #modal-buttons Button { margin: 0 1 0 0; }
    .perm-summary { padding: 1 0 0 0; color: $accent; }
    .perm-detail { max-height: 12; padding: 1 1 0 1; color: $text-muted; }

    .hidden { display: none; }
    """

    BINDINGS = [Binding("ctrl+q", "quit", "Quit", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.config: ConfigFile = store.load()
        self._managers: dict[str, ServerManager] = {}

    def on_mount(self) -> None:
        try:
            self.register_theme(MLX_THEME)
            saved = self.config.settings.theme
            self.theme = saved if saved in self.available_themes else "mlx-dark"
        except Exception:
            pass  # fall back to a built-in theme rather than crashing

        from .screens.dashboard import DashboardScreen

        self.push_screen(DashboardScreen())

        from .bootstrap import mlx_server_available

        if not mlx_server_available() and not (self.config.settings.mlx_server_path):
            from .screens.setup import SetupScreen

            self.push_screen(SetupScreen())

    # --- config ----------------------------------------------------------

    def notify(self, message: str, **kwargs) -> None:
        # toast text is arbitrary (names, paths, errors) and may contain markup
        # chars like `[w=600&h=400]`; never parse it as markup (it would crash).
        kwargs.setdefault("markup", False)
        super().notify(message, **kwargs)

    def save_config(self) -> None:
        store.save(self.config)

    # --- clipboard -------------------------------------------------------

    def copy_text(self, text: str) -> bool:
        """Copy text to the system clipboard. Returns True if it definitely landed.

        Textual's `copy_to_clipboard` emits OSC 52 escape sequences, which many
        terminals — notably macOS Terminal.app — silently ignore, so the paste
        never happens even though we showed "Copied". On macOS we also pipe to
        `pbcopy`, which always works for a local session; OSC 52 still covers
        `textual serve` / remote terminals that do support it.
        """
        self.copy_to_clipboard(text)
        clip = shutil.which("pbcopy") if sys.platform == "darwin" else None
        if clip:
            try:
                subprocess.run([clip], input=text.encode(), check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                pass
        return False

    # --- server manager registry ----------------------------------------

    def create_manager(self, cfg: ServerConfig) -> ServerManager:
        manager = ServerManager(
            cfg, mlx_override=self.config.settings.mlx_server_path or None
        )
        self._managers[cfg.id] = manager
        return manager

    def get_manager(self, cfg_id: str) -> Optional[ServerManager]:
        return self._managers.get(cfg_id)

    def running_managers(self) -> list[ServerManager]:
        """Every server manager with a live process. Only one server can bind a
        given host:port, so callers use this to free a port before launching."""
        return [m for m in self._managers.values() if m.is_running]

    def on_unmount(self) -> None:
        # Best-effort: don't orphan server processes on quit.
        for manager in self._managers.values():
            manager.terminate()


def run() -> None:
    MlxLauncherApp().run()


if __name__ == "__main__":
    run()
