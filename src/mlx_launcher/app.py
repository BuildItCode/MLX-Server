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

from .models import AppSettings, ConfigFile, ServerConfig
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
    /* the "/" command menu, shown just above the prompt while typing a slash command */
    #slash-suggest { display: none; height: auto; max-height: 8; margin: 0 1; border: round $accent; background: $surface; }
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
    #hf-progress { width: 1fr; height: 1; margin: 1 0 0 0; }
    #hf-progress Bar { width: 1fr; }
    #hf-log { height: 12; border: round $panel; margin: 1 0 0 0; }
    #hf-buttons { height: auto; padding: 1 0; }
    #hf-buttons Button { margin: 0 1 0 0; }
    /* editor: model field + Search HF button on one row */
    #model-row { height: auto; }
    #model-row #model { width: 1fr; }
    #model-row #hf-search { width: auto; min-width: 12; margin: 0 0 0 1; }
    /* editor: dropdown of already-downloaded HF models under the model field */
    #model-suggest { display: none; height: auto; max-height: 10; margin: 0; border: round $panel; background: $surface; }
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
        self.config: ConfigFile = ConfigFile()  # in-memory cache; populated from the backend on mount
        self._backend = None  # connected/spawned BackendClient (see backend())
        self._backend_lock = None

    async def backend(self):
        """The wire client for the local backend service, connecting to a running one or spawning
        ``lis-backend`` on first use. The frontend reaches the backend ONLY through this client."""
        import asyncio as _asyncio

        from .client import connect

        if self._backend_lock is None:
            self._backend_lock = _asyncio.Lock()
        async with self._backend_lock:
            if self._backend is None:
                self._backend = await connect()
        return self._backend

    def on_mount(self) -> None:
        # Paint the first screen IMMEDIATELY — do NOT block first paint on the backend.
        # Spawning/discovering lis-backend can take a second or two; loading config behind
        # it would leave the terminal blank until then. Register a theme synchronously
        # (no backend needed), push the dashboard, then finish startup in a worker.
        try:
            self.register_theme(MLX_THEME)
            self.theme = "mlx-dark"
        except Exception:
            pass  # fall back to a built-in theme rather than crashing

        from .screens.dashboard import DashboardScreen

        self.push_screen(DashboardScreen())
        self.run_worker(self._startup(), exclusive=False)

    async def _startup(self) -> None:
        """Connect to the backend, load config, then apply the saved theme and (if no model
        server is installed) surface the setup screen. Runs as a worker so first paint isn't
        blocked on the backend coming up."""
        await self.refresh_config()  # load profiles + settings from the backend
        try:
            saved = self.config.settings.theme
            if saved in self.available_themes:
                self.theme = saved
        except Exception:
            pass

        from .bootstrap import mlx_server_available

        if not mlx_server_available() and not (self.config.settings.mlx_server_path):
            from .screens.setup import SetupScreen

            self.push_screen(SetupScreen())

    # --- config (cached from the backend) --------------------------------

    async def refresh_config(self) -> None:
        """Reload the cached server profiles + settings from the backend over the wire."""
        client = await self.backend()
        try:
            servers = [ServerConfig.model_validate(s) for s in await client.list_servers()]
            settings = AppSettings.model_validate(await client.get_settings())
            self.config = ConfigFile(servers=servers, settings=settings)
        except Exception:  # noqa: BLE001 — keep the last good cache rather than crash the UI
            pass

    def notify(self, message: str, **kwargs) -> None:
        # toast text is arbitrary (names, paths, errors) and may contain markup
        # chars like `[w=600&h=400]`; never parse it as markup (it would crash).
        kwargs.setdefault("markup", False)
        super().notify(message, **kwargs)

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

    def on_unmount(self) -> None:
        # Stop any voice playback/recording so its worker thread returns — otherwise asyncio's
        # shutdown_default_executor() blocks joining it and the process appears to hang on quit.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:  # noqa: BLE001 — sounddevice not installed / no active stream
            pass
        # Model-server subprocesses are owned by the backend now; it stops them on its own
        # shutdown (core.service lifespan), so the TUI doesn't manage their lifecycle here.


def run() -> None:
    # Do this on the MAIN thread before the app starts: tqdm's default multiprocessing lock
    # (used by mlx_whisper + our download bar) spawns the resource_tracker via fork_exec, which
    # crashes with "bad value(s) in fds_to_keep" when first triggered from a worker thread.
    from .chat import voice
    voice.ensure_threading_tqdm_lock()
    MlxLauncherApp().run()


if __name__ == "__main__":
    run()
