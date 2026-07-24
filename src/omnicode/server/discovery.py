"""Re-export shim. Binary/port/GGUF discovery moved to :mod:`omnicode.core.server.discovery`
(the backend server layer). Importing from ``omnicode.server.discovery`` still works for
back-compat."""

from ..core.server.discovery import *  # noqa: F401,F403
