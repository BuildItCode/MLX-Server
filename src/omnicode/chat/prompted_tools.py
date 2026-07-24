"""Re-export shim. The prompted (text-protocol) tool-call layer moved to
:mod:`omnicode.engine.prompted` (the format-engine layer). Importing from
``omnicode.chat.prompted_tools`` still works for back-compat."""

from ..engine.prompted import *  # noqa: F401,F403
from ..engine.prompted import (  # noqa: F401
    parse_tool_calls,
    parse_xml_tool_calls,
    strip_tool_calls,
    tool_instructions,
    tool_response,
)
