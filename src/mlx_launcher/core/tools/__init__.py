"""Backend tool layer: the agent's executable tools and their phrasing.

- ``web``  — DuckDuckGo web search (``web_search_spec`` / ``run_web_search``)
- ``fs``   — sandboxed filesystem tools confined to a project working dir
             (``fs_specs`` / ``run_fs_tool`` / ``FS_TOOL_NAMES`` / ``MUTATING_TOOLS``)
- ``mcp``  — Model Context Protocol sessions (``open_sessions`` / ``call_mcp`` / ``slug``)
- ``phrasing`` — human descriptions of tool calls / permission prompts

Tool-call execution and the permission gate live above this, in the agent loop. Module imports
here are cheap: the heavy deps (``ddgs``, ``mcp``) are imported lazily inside the functions."""

from . import fs, mcp, phrasing, web

__all__ = ["fs", "mcp", "phrasing", "web"]
