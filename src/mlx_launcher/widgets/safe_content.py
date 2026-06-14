"""Build Textual `Content` from arbitrary text WITHOUT markup-parsing it.

Textual's `Static`/`Label` parse their string content as console markup, and the
parser raises `MarkupError` on text containing patterns like `[w=600&h=400]`
(extremely common in model output, URLs, JSON, code). `escape()` does NOT
reliably prevent this. So for any externally-sourced text (model output, tool
results, user input, file paths, names, descriptions) build `Content` from
literal text + styles instead of interpolating into a markup string."""

from __future__ import annotations

from textual.content import Content


def plain(text: str) -> Content:
    """Literal text, never markup-parsed."""
    return Content(text or "")


def title_sub(title: str, subtitle: str, *, title_style: str = "bold", sub_style: str = "dim") -> Content:
    """A bold title over a dim subtitle — the common list-item shape."""
    return Content.assemble((title or "", title_style), "\n", (subtitle or "", sub_style))
