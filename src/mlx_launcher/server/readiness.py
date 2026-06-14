"""Detect when an mlx_lm.server instance is ready to serve requests."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Callable, Optional

import httpx

# mlx_lm.server logs this once the HTTP server is bound.
STARTING_RE = re.compile(r"Starting httpd at .* on port (\d+)", re.IGNORECASE)


async def probe_once(server_url: str, *, timeout: float = 2.0) -> bool:
    """One readiness probe: /health, falling back to /v1/models."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ("/health", "/v1/models"):
            try:
                resp = await client.get(server_url + path)
                if resp.status_code < 500:
                    return True
            except Exception:
                continue
    return False


async def wait_until_ready(
    server_url: str,
    *,
    timeout: float = 180.0,
    interval: float = 0.3,
    should_continue: Optional[Callable[[], bool]] = None,
) -> bool:
    """Poll until the server responds or the timeout elapses.

    `should_continue` lets the caller abort early (e.g. the process died)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if should_continue is not None and not should_continue():
            return False
        if await probe_once(server_url):
            return True
        await asyncio.sleep(interval)
    return False
