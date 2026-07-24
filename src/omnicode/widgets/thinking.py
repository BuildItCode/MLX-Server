"""A small animated 'the model is thinking' placeholder shown in the assistant
bubble until the real reply streams in — a braille spinner plus a cycling,
whimsical verb (Claude-Code style)."""

from __future__ import annotations

from typing import Optional

from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static


class ThinkingIndicator(Static):
    """Animates itself via a timer. It is a ``Static`` subclass, so once the
    answer arrives the caller can ``stop()`` the animation and ``update()`` the
    same widget with content (or just remove it)."""

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # braille dots — smooth at ~11 fps
    WORDS = (
        "Thinking", "Pondering", "Percolating", "Cooking", "Conjuring",
        "Noodling", "Musing", "Brewing", "Scheming", "Crunching", "Vibing",
        "Ruminating", "Tinkering", "Daydreaming",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer: Optional[Timer] = None

    def on_mount(self) -> None:
        self._render_frame()
        self._timer = self.set_interval(0.09, self._advance)

    def _advance(self) -> None:
        self._frame += 1
        self._render_frame()

    def _render_frame(self) -> None:
        spin = self.SPINNER[self._frame % len(self.SPINNER)]
        word = self.WORDS[(self._frame // 22) % len(self.WORDS)]  # new verb ~every 2s
        dots = "." * ((self._frame // 5) % 4)
        # Content (not markup) so the verb/dots are never parsed as style tags.
        self.update(Content.assemble((spin, "bold"), "  ", (f"{word}{dots}", "dim")))

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def on_unmount(self) -> None:
        self.stop()
