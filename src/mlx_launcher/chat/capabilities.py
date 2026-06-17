"""Re-export shim. Capability heuristics + attachment encoding moved to
:mod:`mlx_launcher.engine.capabilities` (the format-engine layer). Importing from
``mlx_launcher.chat.capabilities`` still works for back-compat."""

from ..engine.capabilities import *  # noqa: F401,F403
from ..engine.capabilities import (  # noqa: F401  (private/explicit names import * skips)
    REASONING_EFFORTS,
    approx_tokens,
    classify,
    context_window,
    encode_image,
    estimate_prompt_tokens,
    image_mime,
    read_text_attachment,
    reasoning_template_kwargs,
    supports_reasoning,
    supports_vision,
)
