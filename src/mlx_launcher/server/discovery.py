"""Re-export shim. Binary/port/GGUF discovery moved to :mod:`mlx_launcher.core.server.discovery`
(the backend server layer). Importing from ``mlx_launcher.server.discovery`` still works for
back-compat."""

from ..core.server.discovery import *  # noqa: F401,F403
