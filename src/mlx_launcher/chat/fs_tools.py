"""Re-export shim. The sandboxed filesystem tools moved to :mod:`mlx_launcher.core.tools.fs`
(the backend tool layer). Importing from ``mlx_launcher.chat.fs_tools`` still works for
back-compat."""

from ..core.tools.fs import *  # noqa: F401,F403
from ..core.tools.fs import (  # noqa: F401
    COMMAND_TIMEOUT,
    FS_TOOL_NAMES,
    MAX_OUTPUT,
    MAX_READ_BYTES,
    MUTATING_TOOLS,
    SYSTEM_NOTE,
    fs_specs,
    resolve_browser_target,
    run_fs_tool,
    system_note,
)
