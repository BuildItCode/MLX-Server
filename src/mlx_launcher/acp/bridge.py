"""Re-export shim. The OpenAI-compatible HTTP adapter (``MlxBridge``) moved to
:mod:`mlx_launcher.engine.openai` (the format-engine layer). Importing from
``mlx_launcher.acp.bridge`` still works for back-compat."""

import httpx  # noqa: F401  — re-exported so tests can monkeypatch `acp.bridge.httpx.AsyncClient`

from ..engine.openai import (  # noqa: F401
    _CANCELLED,
    _http_error,
    _iter_sse_lines,
    MlxBridge,
    OpenAIEngine,
    fetch_models,
)
