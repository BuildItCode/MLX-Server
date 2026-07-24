"""deepagents integration layer.

Adapts the LangChain/LangGraph-based ``deepagents`` library to omnicode's existing
OpenAI-engine + event-stream architecture. The hand-rolled agent loop
(``core/agent.py``) now delegates here.

Components:

* :mod:`model`  — wraps the existing :class:`~omnicode.engine.base.Engine` as a
  LangChain :class:`~langchain_core.language_models.BaseChatModel`, preserving all
  format-recovery (Harmony, prompted tools, text-dialect tool calls) and reasoning
  streaming that the old loop relied on.
* :mod:`tools`  — converts the existing web/fs/MCP tool specs into LangChain
  ``StructuredTool`` objects so ``create_deep_agent`` can bind them natively.
* :mod:`middleware` — a permission middleware (human-in-the-loop for mutating tools)
  and an event-translation middleware that bridges deepagent streaming events to
  omnicode's own :mod:`omnicode.core.events`.
* :mod:`adapter` — :func:`build_agent` (constructs the ``DeepAgent`` with subagents,
  memory, skills, filesystem tools) and :func:`run_turn` (drives one user turn and
  translates the LangGraph event stream into omnicode events).
"""

from .adapter import build_agent, run_turn
from .model import EngineChatModel
from .middleware import PermissionMiddleware, _ToolEventCapture

__all__ = [
    "build_agent",
    "run_turn",
    "EngineChatModel",
    "PermissionMiddleware",
    "_ToolEventCapture",
]
