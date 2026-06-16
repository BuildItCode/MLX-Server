"""Split assistant Markdown into prose runs and fenced code blocks, so each code
block can be rendered with its own copy control."""

from __future__ import annotations

import re

_OPEN = re.compile(r"^\s*```([\w+.-]*)\s*$")
_CLOSE = re.compile(r"^\s*```\s*$")

_INLINE_CODE = re.compile(r"`[^`\n]*`")
# a bare http(s) URL not already inside link syntax (`[..](..)`, `<..>`, `[..]`),
# excluding trailing sentence punctuation
_BARE_URL = re.compile(r"""(?<![\[(<])\bhttps?://[^\s<>()\[\]`"']*[^\s<>()\[\]`"'.,;:!?]""")


def linkify_urls(text: str) -> str:
    """Wrap bare http(s) URLs in Markdown link syntax so they render as clickable links
    (Rich Markdown only links `[text](url)`, not the bare URLs LLMs commonly emit). Leaves
    URLs already inside link/autolink syntax and inside inline-code spans untouched."""
    def sub(s: str) -> str:
        return _BARE_URL.sub(lambda m: f"[{m.group(0)}]({m.group(0)})", s)

    out, pos = [], 0
    for code in _INLINE_CODE.finditer(text):  # skip inline-code spans verbatim
        out.append(sub(text[pos:code.start()]))
        out.append(code.group(0))
        pos = code.end()
    out.append(sub(text[pos:]))
    return "".join(out)


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
