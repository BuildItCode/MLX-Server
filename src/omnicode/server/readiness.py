"""Re-export shim. HTTP readiness probing moved to :mod:`omnicode.core.server.readiness`
(the backend server layer). Importing from ``omnicode.server.readiness`` still works for
back-compat."""

from ..core.server.readiness import *  # noqa: F401,F403
from ..core.server.readiness import wait_until_ready  # noqa: F401
