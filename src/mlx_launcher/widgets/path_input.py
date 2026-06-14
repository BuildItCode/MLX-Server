"""Path input that understands drag-and-dropped model folders.

Terminals paste a dropped folder as its POSIX path — often quoted or with
backslash-escaped spaces, and sometimes with trailing whitespace. These helpers
normalize that; `resolve_path` additionally expands ~ and env vars.

`DropPathInput` goes a step further: when a real filesystem path is *pasted/
dropped* onto it (regardless of which field has focus), it emits `PathDropped`
instead of inserting the raw text, so the editor can route it to the model field."""

from __future__ import annotations

import os
import re

from textual import events
from textual.message import Message
from textual.widgets import Input

_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def sanitize_drag(raw: str) -> str:
    """Strip drag-and-drop artifacts (surrounding quotes, escaped spaces,
    whitespace). Idempotent and safe to apply on every change — it does NOT
    expand ~ so typing a path still works."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    elif "\\" in s:
        s = re.sub(r"\\(.)", r"\1", s)
    return s.strip()


def resolve_path(raw: str) -> str:
    s = sanitize_drag(raw)
    if s.startswith("~") or "$" in s:
        s = os.path.expanduser(os.path.expandvars(s))
    return s


def path_hint(raw: str) -> str:
    s = resolve_path(raw)
    if not s:
        return ""
    if os.path.isdir(s):
        return "✓ folder exists"
    if os.path.isfile(s):
        return "✓ file exists"
    if _REPO_RE.match(s):
        return "↯ treating as a HuggingFace repo id"
    return "✗ path not found (will still be passed to the server)"


class DropPathInput(Input):
    """An Input that recognizes a dropped/pasted filesystem path and reports it via
    a `PathDropped` message rather than inserting the raw text."""

    class PathDropped(Message):
        def __init__(self, path: str) -> None:
            self.path = path
            super().__init__()

    def _on_paste(self, event: events.Paste) -> None:
        text = event.text.splitlines()[0] if event.text else ""
        cleaned = resolve_path(text)
        if cleaned and os.path.isabs(cleaned) and os.path.exists(cleaned):
            event.stop()
            self.post_message(self.PathDropped(cleaned))
            return
        super()._on_paste(event)


class PathInput(DropPathInput):
    """A drop-aware Input whose value can be read back as a clean filesystem path."""

    def resolved(self) -> str:
        return resolve_path(self.value)

    def hint(self) -> str:
        return path_hint(self.value)
