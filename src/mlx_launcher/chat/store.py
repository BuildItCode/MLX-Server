"""Re-export shim. The chat store (atomic JSON at ~/.config/mlx-launcher/chats.json) moved to
:mod:`mlx_launcher.core.persistence.chats` (the backend persistence layer). Importing from
``mlx_launcher.chat.store`` still works for back-compat."""

from ..core.persistence.chats import *  # noqa: F401,F403
from ..core.persistence.chats import (  # noqa: F401
    chats_in,
    chats_path,
    delete_chat,
    delete_mcp,
    delete_project,
    delete_subagent,
    get_chat,
    get_project,
    get_subagent,
    load,
    mutate,
    save,
    upsert_chat,
    upsert_mcp,
    upsert_project,
    upsert_subagent,
)
