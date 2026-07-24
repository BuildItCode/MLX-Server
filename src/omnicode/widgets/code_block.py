"""A fenced-code block with a syntax-highlighted body and a click-to-copy control."""

from __future__ import annotations

import re

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static


class _CopyLink(Static):
    """A small clickable 'copy' affordance that copies the block to the clipboard."""

    def __init__(self, code: str) -> None:
        super().__init__("❏ copy", classes="code-copy")
        self._code = code

    def on_click(self) -> None:
        self.app.copy_text(self._code)
        self.app.notify("Copied code block")


class CodeBlock(Vertical):
    def __init__(self, code: str, lang: str = "") -> None:
        super().__init__(classes="code-block")
        self._code = code
        self._lang = (lang or "text").lower()

    def compose(self) -> ComposeResult:
        with Horizontal(classes="code-head"):
            yield Static(self._lang, classes="code-lang")
            yield _CopyLink(self._code)
        yield Static(self._body(), classes="code-body")

    def _body(self) -> RichMarkdown:
        # Render via a single Markdown fence: Rich highlights the code, and a
        # Markdown renderable is what Textual's Static reliably accepts as content.
        # Size the fence longer than any backtick run inside the code.
        longest = max((len(m.group()) for m in re.finditer(r"`+", self._code)), default=2)
        fence = "`" * max(longest + 1, 3)
        return RichMarkdown(f"{fence}{self._lang}\n{self._code}\n{fence}")
