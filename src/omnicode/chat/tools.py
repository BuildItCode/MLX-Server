"""Re-export shim. The web_search tool moved to :mod:`omnicode.core.tools.web` (the backend
tool layer). Importing from ``omnicode.chat.tools`` still works for back-compat."""

from ..core.tools.web import *  # noqa: F401,F403
from ..core.tools.web import WEB_SEARCH_SPEC, run_web_search, web_search_spec  # noqa: F401
