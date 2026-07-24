"""Detect when an mlx_lm.server instance is ready to serve requests."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import httpx


async def probe_once(
    server_url: str, *, timeout: float = 2.0, client: Optional[httpx.AsyncClient] = None
) -> bool:
    """One readiness probe: /health, falling back to /v1/models. Reuses `client` when given
    (the poll loop shares one), otherwise opens and closes its own."""
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        for path in ("/health", "/v1/models"):
            try:
                resp = await client.get(server_url + path)
                if resp.status_code < 500:
                    return True
            except Exception:
                continue
        return False
    finally:
        if own:
            await client.aclose()


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
    # one client for the whole poll loop — readiness can poll ~3×/s for minutes while a large
    # model loads, so a fresh client (+ connection setup) per probe would be needless churn.
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if should_continue is not None and not should_continue():
                return False
            if await probe_once(server_url, client=client):
                return True
            await asyncio.sleep(interval)
    return False
