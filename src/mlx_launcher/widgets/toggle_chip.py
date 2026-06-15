"""A compact click-to-toggle 'chip': the word itself is the control, and its
square lights up (accent background) when on. Replaces Label+Switch pairs."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import Static


class ToggleChip(Static):
    """`ToggleChip("web", "web")` — clicking flips it and posts `Changed(key, value)`.

    `set_value` syncs state without posting (when the screen drives it from a
    chat); `set_enabled(False)` locks it greyed-out (e.g. reason on a model that
    can't reason)."""

    class Changed(Message):
        def __init__(self, key: str, value: bool) -> None:
            self.key = key
            self.value = value
            super().__init__()

    def __init__(self, label: str, key: str, value: bool = False, **kwargs) -> None:
        super().__init__(label, **kwargs)  # label is a literal word — markup-safe
        self._key = key
        self._on = value
        self._locked = False
        self.add_class("chip")

    def on_mount(self) -> None:
        self._reflect()

    def on_click(self) -> None:
        if self._locked:
            return
        self._on = not self._on
        self._reflect()
        self.post_message(self.Changed(self._key, self._on))

    @property
    def value(self) -> bool:
        return self._on

    @property
    def key(self) -> str:
        return self._key

    def set_value(self, value: bool) -> None:
        """Set state programmatically (no Changed posted)."""
        self._on = bool(value)
        self._reflect()

    def set_enabled(self, enabled: bool) -> None:
        self._locked = not enabled
        if self._locked:
            self._on = False
        self._reflect()

    def _reflect(self) -> None:
        self.set_class(self._on, "-on")
        self.set_class(self._locked, "-disabled")
