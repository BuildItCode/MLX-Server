"""Re-export shim. The config store (atomic JSON at ~/.config/omnicode/servers.json) moved to
:mod:`omnicode.core.persistence.config` (the backend persistence layer). Importing from
``omnicode.config.store`` still works for back-compat."""

from ..core.persistence.config import *  # noqa: F401,F403
from ..core.persistence.config import (  # noqa: F401
    atomic_write_text,
    backup_aside,
    config_dir,
    config_path,
    delete_server,
    find_server_by_id,
    get_server,
    load,
    mutate,
    save,
    upsert_server,
)
