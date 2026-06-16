"""The Textual application: theme, global styles, screen wiring, and the
registry of running server managers."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Optional

from textual.app import App
from textual.binding import Binding

from .config import store
from .config.models import ConfigFile, ServerConfig
from .server.manager import ServerManager
from .theme import MLX_THEME


def _clipboard_cmd() -> Optional[list[str]]:
    """Argv for a local CLI that reliably sets the system clipboard (so copy works even
    where OSC 52 is ignored — notably macOS Terminal.app), or None if none is available."""
    if sys.platform == "darwin":
        p = shutil.which("pbcopy")
        return [p] if p else None
    if sys.platform == "win32":
        p = shutil.which("clip")
        return [p] if p else None
    for cmd, args in (("wl-copy", []), ("xclip", ["-selection", "clipboard"]),
                      ("xsel", ["--clipboard", "--input"])):
        p = shutil.which(cmd)
        if p:
            return [p, *args]
    return None


class MlxLauncherApp(App):
    TITLE = "LIS"
    SUB_TITLE = "Local Inference Server"

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
    /* two side-by-side conversation panes (main + optional subagent side chat) */
    #panes { height: 1fr; }
    .pane { width: 1fr; border-top: solid $panel; }
    .pane.active { border-top: solid $accent; }
    #side-pane { border-left: solid $panel; }
    #side-head { height: auto; padding: 0 1; }
    #side-title { width: 1fr; padding: 1 0 0 0; color: $secondary; text-style: bold; }
    #side-head Button { width: auto; min-width: 10; margin: 0 0 0 1; }
    #side-transcript { height: 1fr; padding: 0 1; }
    #chat-topbar { height: auto; padding: 0 1; }
    #chat-title { width: 1fr; padding: 1 0 0 0; }
    #server-select { width: 34; }
    #skill-select { width: 40; }
    #transcript { height: 1fr; padding: 0 1; }
    #attachments { height: auto; padding: 0 1; }
    #attach { margin: 0 1; }
    #chat-actions { height: auto; padding: 0 1 1 1; }
    /* compact, flat secondary actions (sit below the input row) */
    #chat-actions Button { height: 1; min-width: 0; border: none; padding: 0 1; margin: 0 1 0 0; color: $text-muted; background: $panel; }
    #chat-actions Button:hover { color: $text; }
    /* read-aloud while speaking → flat accent text (label also flips to "Stop") */
    #chat-actions Button.-reading { color: $accent; text-style: bold; }
    #chat-inputrow { height: auto; padding: 0 1 1 1; }
    #prompt { width: 1fr; height: 6; border: round $panel; border-title-color: $accent; }
    #chat-inputrow Button { margin: 0 0 0 1; }
    /* mic while recording → flat red text (label also flips to "Stop") */
    #mic-btn.-recording { color: $error; text-style: bold; }
    #chat-chips { height: auto; padding: 1 1; align-vertical: middle; }
    .actions-spacer { width: 1fr; height: 1; }
    .ctx-bar { width: auto; padding: 0; }
    /* click-to-toggle chips (plan/reason/web/coding/tools) + connectors button */
    .chip { width: auto; height: 1; padding: 0 1; margin: 0 1 0 0; color: $text-muted; background: $panel; }
    .chip:hover { color: $text; }
    .chip.-on { background: $accent; color: $background; text-style: bold; }
    .chip.-disabled { color: #555; }
    .chip-action { color: $accent; }
    .connector-chip { width: 1fr; height: 1; margin: 0 0 1 0; }
    #connector-list { height: auto; max-height: 16; }
    .msg-row { height: auto; width: 1fr; }
    .msg-spacer { width: 1fr; height: auto; }
    .msg { height: auto; width: 80%; max-width: 80%; margin: 1 0 0 0; padding: 0 0 0 1; border-left: solid $panel; }
    .msg-role { text-style: bold; width: auto; }
    .msg-body { height: auto; padding: 0; }
    /* assistant prose is a Markdown widget (so it's selectable) — strip its default chrome */
    Markdown.msg-body { background: transparent; margin: 0; padding: 0; }
    Markdown.msg-body MarkdownBlock { margin: 0; padding: 0; }
    Markdown.msg-body MarkdownParagraph { margin: 0 0 1 0; }
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
    .msg-copy { width: auto; color: $text-muted; padding: 0; margin: 0 0 0 0; }
    .msg-copy:hover { color: $accent; text-style: bold; }
    /* copyable code blocks */
    .code-block { height: auto; width: 1fr; margin: 1 0; border: round $panel; }
    .code-head { height: 1; width: 1fr; background: $panel; }
    .code-lang { width: 1fr; color: $text-muted; padding: 0 1; }
    .code-copy { width: auto; color: $accent; padding: 0 1; }
    .code-copy:hover { background: $accent; color: $background; text-style: bold; }
    .code-body { width: 1fr; height: auto; padding: 0 1; }
    .msg-tool { width: auto; border-left: solid $warning; }
    .msg-tool .msg-role { color: $warning; }
    .msg-tool .msg-body { width: auto; }
    /* subagents dropdown (modal): one row per subagent with compact Chat/Edit/× actions */
    #sa-modal-list { height: auto; max-height: 16; padding: 1 0; }
    .sa-modal-row { height: 1; align-vertical: middle; margin: 0 0 1 0; }
    .sa-modal-name { width: 1fr; height: 1; padding: 0 1 0 0; }
    .sa-modal-row Button { height: 1; min-width: 8; margin: 0 0 0 1; border: none; }
    .sa-modal-row .sa-del { min-width: 5; }
    .sa-modal-row .sa-chat { background: $success; color: $background; text-style: bold; }
    .sa-modal-row .sa-edit { background: $panel; color: $text; }
    .sa-modal-row .sa-del { background: $panel; color: $text-muted; }
    .sa-modal-row .sa-chat:hover { background: $success-lighten-1; }
    .sa-modal-row .sa-edit:hover, .sa-modal-row .sa-del:hover { background: $primary; color: $background; }
    /* subagent editor */
    #subagent-form { height: 1fr; padding: 0 1; }
    #sa-prompt { height: 8; border: round $panel; }
    #sa-mcp, #sa-skills { height: auto; }
    .chip-row { height: 1; }
    #sa-kb-add { min-width: 8; margin: 0 0 0 1; }
    #sa-kb-list { height: auto; padding: 0 0 1 0; }
    .kb-row { height: 1; align-vertical: middle; }
    .kb-path { width: 1fr; color: $text-muted; }
    .kb-del { min-width: 5; height: 1; border: none; background: $panel; color: $text-muted; }
    .kb-del:hover { background: $primary; color: $background; }
    #subagent-buttons { height: auto; padding: 1 0; }
    #subagent-buttons Button { margin: 0 1 0 0; }
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
    /* HuggingFace model browser (search + download), opened from the server editor */
    #hf-body { height: 1fr; padding: 0 1; }
    #hf-controls { height: auto; padding: 1 0; align-vertical: middle; }
    #hf-controls #hf-query { width: 1fr; }
    #hf-controls #hf-go { margin: 0 0 0 1; }
    #hf-controls .chip { margin: 0 0 0 1; }
    #hf-budget { height: auto; color: $text-muted; padding: 0 0 1 0; }
    #hf-status { height: auto; color: $text-muted; }
    #hf-results { height: 1fr; border: round $panel; padding: 0 1; }
    .hf-row { height: auto; align-vertical: middle; padding: 0 0 1 0; }
    .hf-row-name { width: 1fr; height: auto; }
    .hf-row-fit { width: 22; height: auto; }
    .hf-row-meta { width: 12; height: auto; color: $text-muted; }
    .hf-row Button { width: 14; min-width: 10; height: 1; margin: 0 0 0 1; border: none; }
    .hf-row .hf-pick { background: $success; color: $background; text-style: bold; }
    .hf-row .hf-choose { background: $panel; color: $accent; }
    #hf-log { height: 12; border: round $panel; margin: 1 0 0 0; }
    #hf-buttons { height: auto; padding: 1 0; }
    #hf-buttons Button { margin: 0 1 0 0; }
    /* editor: model field + Search HF button on one row */
    #model-row { height: auto; }
    #model-row #model { width: 1fr; }
    #model-row #hf-search { width: auto; min-width: 12; margin: 0 0 0 1; }
    TextPromptModal, ConfirmModal, PermissionModal, ConnectorsModal, SubagentsModal { align: center middle; background: $background 60%; }
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

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to the system clipboard. Used by BOTH our ⧉ Copy controls AND
        Textual's built-in **text selection** — drag to highlight any part of a reply,
        then press Ctrl+C / Cmd+C (`screen.copy_text` → `action_copy_text` calls this).

        Textual's base only emits OSC 52, which macOS Terminal.app (and some others)
        silently ignore — so we ALSO pipe to a native clipboard CLI (pbcopy / clip /
        wl-copy / xclip) for a local session. OSC 52 still covers `textual serve` and
        remote terminals that support it."""
        super().copy_to_clipboard(text)  # OSC 52
        cmd = _clipboard_cmd()
        if cmd:
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass

    def copy_text(self, text: str) -> bool:
        """Back-compat helper for our copy affordances — delegates to copy_to_clipboard
        (which now also drives the native clipboard)."""
        self.copy_to_clipboard(text)
        return True

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
        # Stop any voice playback/recording so its worker thread returns — otherwise asyncio's
        # shutdown_default_executor() blocks joining it and the process appears to hang on quit.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:  # noqa: BLE001 — sounddevice not installed / no active stream
            pass
        # Don't orphan model-server subprocesses on quit: SIGTERM them all, give them a
        # moment to exit, then SIGKILL any survivor. A server still loading weights can be
        # slow to honor SIGTERM and would otherwise outlive us holding memory and the port
        # (children run in their own session, so they aren't killed along with us).
        running = [m for m in self._managers.values() if m.is_running]
        for manager in running:
            manager.terminate()
        deadline = time.monotonic() + 2.0
        while running and time.monotonic() < deadline:
            time.sleep(0.05)
            running = [m for m in running if m.is_alive()]
        for manager in running:
            manager.kill_now()


def run() -> None:
    # Do this on the MAIN thread before the app starts: tqdm's default multiprocessing lock
    # (used by mlx_whisper + our download bar) spawns the resource_tracker via fork_exec, which
    # crashes with "bad value(s) in fds_to_keep" when first triggered from a worker thread.
    from .chat import voice
    voice.ensure_threading_tqdm_lock()
    MlxLauncherApp().run()


if __name__ == "__main__":
    run()
