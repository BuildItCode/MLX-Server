"""Panel showing the running server's status and addresses."""

from __future__ import annotations

from rich.markup import escape
from textual.content import Content
from textual.widgets import Static

from ..models import ServerConfig
from ..theme import STATUS_BADGE


class AddressPanel(Static):
    DEFAULT_CSS = """
    AddressPanel {
        height: auto;
        padding: 1 2;
        border: round $primary;
        border-title-color: $primary;
    }
    """

    def show(self, status: str, cfg: ServerConfig, message: str = "") -> None:
        glyph, color = STATUS_BADGE.get(status, ("○", "#8a8a8a"))
        head = f"[{color}]{glyph} {status.upper()}[/]"
        if message:
            head += f"   [dim]{escape(message)}[/]"
        lines = [head]
        if status in ("starting", "ready"):
            lines += [
                "",
                f"  server   [b]{escape(cfg.server_url())}[/]",
                f"  openai   [b]{escape(cfg.base_url())}[/]   [dim]← Xcode base URL[/]",
                f"  health   [dim]{escape(cfg.health_url())}[/]",
            ]
        self.update("\n".join(lines))
        self.border_title = Content(f" {cfg.name} ")  # name may contain markup chars
