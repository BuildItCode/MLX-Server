"""Re-export shim. The model-server supervisor moved to :mod:`omnicode.core.server.manager`
(the backend server layer). Importing from ``omnicode.server.manager`` still works for
back-compat. The ``import *`` also re-exports the module's ``sys``/``os`` names, which a test
monkeypatches via ``omnicode.server.manager``."""

from ..core.server.manager import *  # noqa: F401,F403
from ..core.server.manager import (  # noqa: F401
    BinaryNotFound,
    LogCb,
    PortInUse,
    ServerManager,
    ServerStatus,
    StatusCb,
)
