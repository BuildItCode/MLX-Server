"""Re-export shim. MCP session management moved to :mod:`omnicode.core.tools.mcp` (the
backend tool layer). Importing from ``omnicode.chat.mcp_client`` still works for back-compat."""

from ..core.tools.mcp import *  # noqa: F401,F403
from ..core.tools.mcp import call_mcp, open_sessions, slug  # noqa: F401
