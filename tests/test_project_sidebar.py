"""Creating a project shows it in the chat sidebar.

Regression: the project editor writes projects backend-side, and ChatScreen.on_screen_resume must
reload self.data.projects so a newly-created project appears. It previously reloaded only subagents
+ mcp_servers, so a new project never showed up even though the comment promised "project edits".
The in-process backend (conftest's autouse fixture) shares the test's XDG store with the screen."""

import asyncio

from textual.widgets import ListView


def test_new_project_appears_in_sidebar_after_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    async def go():
        from mlx_launcher.app import MlxLauncherApp
        from mlx_launcher.screens.chat import ChatScreen, ProjectItem

        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            assert not any(p.name == "My new project" for p in screen.data.projects)

            # what the project editor does on save (backend-side, over the wire)
            client = await app.backend()
            await client.upsert_resource("projects", {"name": "My new project", "instructions": ""})

            # returning from the editor fires on_screen_resume → must pick up the new project
            screen.on_screen_resume()
            await pilot.pause(0.1)

            assert any(p.name == "My new project" for p in screen.data.projects), \
                "new project missing from the reloaded sidebar data"
            # and it rendered as a sidebar row (one ProjectItem per project, plus the 'All chats' row)
            rows = list(screen.query_one("#projects", ListView).query(ProjectItem))
            assert len(rows) == len(screen.data.projects) + 1

    asyncio.run(go())
