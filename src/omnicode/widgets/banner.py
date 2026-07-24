"""The welcome banner widget."""

from __future__ import annotations

from rich.markup import escape
from textual.widgets import Static

from ..theme import ACCENT, BANNER, TAGLINE


class Banner(Static):
    DEFAULT_CSS = """
    Banner {
        height: auto;
        padding: 1 2;
        margin: 1 0 0 0;
        border: round $primary;
        border-title-color: $primary;
        color: $primary;
    }
    """

    def __init__(self, **kwargs) -> None:
        content = f"[{ACCENT}]{escape(BANNER)}[/]\n[dim]{escape(TAGLINE)}[/]"
        super().__init__(content, **kwargs)
        self.border_title = "✻ welcome"
