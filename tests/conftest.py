"""Test harness: give the app an in-process backend.

The frontend reaches the backend only through ``mlx_launcher.client``. In tests we patch
``client.connect`` to return a :class:`BackendClient` wired to an in-process Starlette app
(``httpx.ASGITransport``) instead of spawning a real ``lis-backend`` process. The service reads the
test's ``XDG_CONFIG_HOME`` store, so screens that list/edit data or drive runs work hermetically —
no subprocess, no network, same event loop."""

import httpx
import pytest

import mlx_launcher.client as client_mod
from mlx_launcher.core.service import create_app


@pytest.fixture(autouse=True)
def inproc_backend(monkeypatch):
    """Make ``app.backend()`` resolve to an in-process backend for every test."""
    holder: dict = {}

    async def _connect():
        if "client" not in holder:
            app = create_app(token=None)
            holder["app"] = app
            holder["client"] = client_mod.BackendClient(
                "http://test", token=None, transport=httpx.ASGITransport(app=app))
        return holder["client"]

    monkeypatch.setattr(client_mod, "connect", _connect)
    monkeypatch.setattr(client_mod, "discover", _connect)
    return holder
