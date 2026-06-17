"""Re-export shim. The engine command-line builder moved to
:mod:`mlx_launcher.core.server.flags` (the backend server layer). Importing from
``mlx_launcher.config.flags`` still works for back-compat."""

from ..core.server.flags import *  # noqa: F401,F403
from ..core.server.flags import build_args, build_argv, preview_command  # noqa: F401
