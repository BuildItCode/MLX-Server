"""Two copy-paste blocks for connecting the running server to Xcode 27."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..config.models import ServerConfig
from ..xcode import helpers


class XcodeHelpScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("t", "test", "Test /v1/models"),
        Binding("c", "copy", "Copy ACP JSON"),
    ]

    def __init__(self, cfg: ServerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # --config-id makes the registration stable across host/port/model edits:
        # the agent resolves them from the saved profile each launch.
        self.reg = helpers.acp_registration(cfg, by_config_id=True)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(self._provider_block(), id="provider")
            yield Static(self._acp_block(), id="acp")
            yield Static("[dim]Press t to test the provider endpoint.[/]", id="test-result")
        yield Footer()

    def on_mount(self) -> None:
        # Persist the profile so `mlx-acp-agent --config-id` can resolve it at launch.
        from ..config import store

        store.upsert_server(self.app.config, self.cfg)
        self.app.save_config()

    def _provider_block(self) -> Content:
        p = helpers.openai_provider(self.cfg)
        return Content.assemble(
            ("1 · OpenAI-compatible provider", "bold"), ("  (works today)", "dim"), "\n",
            "Xcode → Settings → Intelligence → add a model provider → ", ("Locally Hosted", "bold"), ":\n\n",
            "   Base URL   ", (p.base_url, "bold"), "\n",
            "   API Key    ", p.api_key, ("   (mlx ignores it; field may be required)", "dim"), "\n",
            "   Model      ", (p.model, "bold"), "\n",
            "   Port       ", str(p.port),
        )

    def _acp_block(self) -> Content:
        return Content.assemble(
            ("2 · ACP agent", "bold"), ("  (Xcode 27 beta — verify in Settings → Intelligence)", "dim"), "\n",
            "Register an external agent with this command, or paste the JSON:\n\n",
            "   Command   ", (self.reg.command, "bold"), "\n",
            "   Args      ", " ".join(self.reg.args), "\n\n",
            self.reg.json_block, "\n\n",
            ("Uses --config-id, so it resolves this profile's URL + model at launch "
             "(survives port/model edits). Does agentic file edits when Xcode grants "
             "filesystem access; otherwise streams chat. The MLX server must be running.", "dim"), "\n",
            ("Press c to copy the JSON.", "dim"), "\n",
            ("Docs: " + helpers.XCODE_DOCS_URL, "dim"),
        )

    # --- actions ---------------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_copy(self) -> None:
        self.app.copy_text(self.reg.json_block)
        self.notify("Copied ACP registration JSON")

    def action_test(self) -> None:
        self.run_worker(self._test(), exclusive=True)

    async def _test(self) -> None:
        from ..acp.bridge import fetch_models

        result = self.query_one("#test-result", Static)
        result.update("[dim]testing GET /v1/models …[/]")
        try:
            models = await fetch_models(self.cfg.base_url())
            listed = ", ".join(models) if models else "(no models reported)"
            result.update(Content.assemble(("✓ /v1/models → ", "#7fb069"), listed))
        except Exception as exc:  # noqa: BLE001
            result.update(Content.assemble(
                ("✗ could not reach " + self.cfg.base_url(), "#e06c75"), "\n", (str(exc), "dim")))
