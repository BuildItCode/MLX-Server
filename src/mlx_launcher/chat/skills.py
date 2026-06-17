"""Re-export shim. Skill discovery/parsing/install moved to :mod:`mlx_launcher.core.skills`
(the backend layer). Importing from ``mlx_launcher.chat.skills`` still works for back-compat.
Private helpers exercised by the test-suite are re-exported explicitly (``import *`` skips them)."""

from ..core.skills import *  # noqa: F401,F403
from ..core.skills import (  # noqa: F401
    _discover,
    _fm_value,
    _slugify,
    _strip_frontmatter,
    all_skills,
    create_custom_skill,
    delete_custom_skill,
    get_skill,
    instructions_for,
    update_custom_skill,
)
