"""Two copy-paste blocks for connecting the running server to Xcode 27."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
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

    def _provider_block(self) -> str:
        p = helpers.openai_provider(self.cfg)
        return (
            "[b]1 · OpenAI-compatible provider[/]  [dim](works today)[/]\n"
            "Xcode → Settings → Intelligence → add a model provider → [b]Locally Hosted[/]:\n\n"
            f"   Base URL   [b]{escape(p.base_url)}[/]\n"
            f"   API Key    {escape(p.api_key)}   [dim](mlx ignores it; field may be required)[/]\n"
            f"   Model      [b]{escape(p.model)}[/]\n"
            f"   Port       {p.port}"
        )

    def _acp_block(self) -> str:
        return (
            "[b]2 · ACP agent[/]  [dim](Xcode 27 beta — verify in Settings → Intelligence)[/]\n"
            "Register an external agent with this command, or paste the JSON:\n\n"
            f"   Command   [b]{escape(self.reg.command)}[/]\n"
            f"   Args      {escape(' '.join(self.reg.args))}\n\n"
            f"{escape(self.reg.json_block)}\n\n"
            "[dim]Uses --config-id, so it resolves this profile's URL + model at launch "
            "(survives port/model edits). Does agentic file edits when Xcode grants "
            "filesystem access; otherwise streams chat. The MLX server must be running.[/]\n"
            "[dim]Press c to copy the JSON.[/]\n"
            f"[dim]Docs: {escape(helpers.XCODE_DOCS_URL)}[/]"
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
            listed = escape(", ".join(models)) if models else "(no models reported)"
            result.update(f"[#7fb069]✓ /v1/models →[/] {listed}")
        except Exception as exc:  # noqa: BLE001
            result.update(
                f"[#e06c75]✗ could not reach {escape(self.cfg.base_url())}[/]\n[dim]{escape(str(exc))}[/]"
            )
