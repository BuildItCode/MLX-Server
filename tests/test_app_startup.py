"""App startup paints the dashboard WITHOUT blocking on the backend, then fills it in.

Covers the first-paint fix: ``on_mount`` pushes the dashboard synchronously and connects to the
backend / loads config in a worker, and the dashboard reloads itself from the backend (so a
profile created server-side shows up once the wire answers). The in-process backend comes from the
autouse ``inproc_backend`` fixture in conftest."""

import asyncio
import sys

from mlx_launcher.core.persistence import config as config_store
from mlx_launcher.models import ServerConfig


def test_dashboard_paints_immediately_then_loads_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    async def go():
        # Seed a profile + a located binary path (so the setup screen isn't auto-pushed) in the
        # backend's store BEFORE the app starts — the frontend should pull this over the wire.
        await config_store.mutate(lambda f: config_store.upsert_server(
            f, ServerConfig(name="seeded-srv", model="llama", engine="llama-cpp",
                            host="127.0.0.1", port=8080)))
        await config_store.mutate(lambda f: setattr(f.settings, "mlx_server_path", sys.executable))

        from mlx_launcher.app import MlxLauncherApp
        from mlx_launcher.screens.dashboard import DashboardScreen, ServerItem

        app = MlxLauncherApp()
        async with app.run_test(size=(120, 40)) as pilot:
            # The dashboard is on screen right away — first paint did NOT wait on the backend.
            assert any(isinstance(s, DashboardScreen) for s in app.screen_stack)

            dash = next(s for s in app.screen_stack if isinstance(s, DashboardScreen))
            names = []
            for _ in range(60):  # let the startup + dashboard workers round-trip the backend
                await pilot.pause(0.05)
                names = [i.server.name for i in dash.query(ServerItem)]
                if "seeded-srv" in names:
                    break
            assert "seeded-srv" in names, names
            # config cache was populated from the backend, not the (empty) initial ConfigFile
            assert [s.name for s in app.config.servers] == ["seeded-srv"]

    asyncio.run(go())
