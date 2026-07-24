"""Re-export shim. Inbound tool-call extraction moved to
:mod:`omnicode.engine.extract` (the format-engine layer). Importing from
``omnicode.chat.tool_calls`` still works for back-compat."""

from ..engine.extract import *  # noqa: F401,F403
from ..engine.extract import Extraction, extract_tool_calls  # noqa: F401
