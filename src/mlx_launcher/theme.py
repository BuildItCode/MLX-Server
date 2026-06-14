"""Visual identity: an ASCII banner, a warm Claude-Code-like theme, and the
status badge palette."""

from __future__ import annotations

from textual.theme import Theme

# Coral/amber wordmark, in the spirit of Claude Code's welcome box.
ACCENT = "#d97757"
AMBER = "#e6b450"

BANNER = r"""
███╗   ███╗██╗     ██╗  ██╗
████╗ ████║██║     ╚██╗██╔╝
██╔████╔██║██║      ╚███╔╝
██║╚██╔╝██║██║      ██╔██╗
██║ ╚═╝ ██║███████╗██╔╝ ██╗
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
""".strip("\n")

TAGLINE = "drag a model · launch a server · wire up Xcode 27"

MLX_THEME = Theme(
    name="mlx-dark",
    primary=ACCENT,
    secondary="#b3936b",
    accent=AMBER,
    foreground="#e8e6e3",
    background="#0e0e10",
    surface="#17171a",
    panel="#202024",
    success="#7fb069",
    warning=AMBER,
    error="#e06c75",
    dark=True,
)

# ServerStatus.value -> (glyph, color)
STATUS_BADGE: dict[str, tuple[str, str]] = {
    "idle": ("○", "#8a8a8a"),
    "starting": ("◐", AMBER),
    "ready": ("●", "#7fb069"),
    "stopped": ("○", "#8a8a8a"),
    "error": ("✗", "#e06c75"),
}
