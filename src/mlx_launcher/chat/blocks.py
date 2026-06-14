"""Split assistant Markdown into prose runs and fenced code blocks, so each code
block can be rendered with its own copy control."""

from __future__ import annotations

import re

_OPEN = re.compile(r"^\s*```([\w+.-]*)\s*$")
_CLOSE = re.compile(r"^\s*```\s*$")


def split_blocks(text: str) -> list[tuple]:
    """Return a list of ('prose', text) and ('code', lang, code) tuples."""
    blocks: list[tuple] = []
    buf: list[str] = []
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        m = _OPEN.match(lines[i])
        if m:
            if buf:
                blocks.append(("prose", "\n".join(buf)))
                buf = []
            lang = m.group(1)
            code: list[str] = []
            i += 1
            while i < n and not _CLOSE.match(lines[i]):
                code.append(lines[i])
                i += 1
            i += 1  # consume closing fence (if present)
            blocks.append(("code", lang, "\n".join(code)))
        else:
            buf.append(lines[i])
            i += 1
    if buf:
        blocks.append(("prose", "\n".join(buf)))
    return blocks
