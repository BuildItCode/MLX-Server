"""The chat screen must not clobber turns the backend persisted during a run.

The backend appends a run's turns to chats.json on disk; the screen keeps a cached self.data. If the
screen later writes that cache back wholesale (e.g. creating a new chat), a stale snapshot would
overwrite the backend's appended turns — losing the conversation. Two guards are tested:
 * _resync_active_chat() reloads the cache and keeps self.chat attached after a run, and
 * _create_chat() re-reads before saving, so it's safe even if the cache is stale."""

import asyncio


def _append_reply(chats_store, chat_id):
    from mlx_launcher.models import ChatMessage

    def mutate(f):
        c = chats_store.get_chat(f, chat_id)
        c.messages.append(ChatMessage(role="user", text="hi"))
        c.messages.append(ChatMessage(role="assistant", text="important reply"))
    return mutate


def test_resync_picks_up_persisted_turns_and_stays_attached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    async def go():
        from mlx_launcher.app import MlxLauncherApp
        from mlx_launcher.core.persistence import chats as chats_store
        from mlx_launcher.screens.chat import ChatScreen

        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.2)
            screen = app.screen
            screen._create_chat()
            a_id = screen.chat.id

            await chats_store.mutate(_append_reply(chats_store, a_id))  # backend persists a run
            screen._resync_active_chat()

            assert any(m.text == "important reply" for m in screen.chat.messages)
            assert any(c is screen.chat for c in screen.data.chats), "self.chat detached from self.data"

    asyncio.run(go())


def test_creating_a_chat_does_not_clobber_prior_persisted_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    async def go():
        from mlx_launcher.app import MlxLauncherApp
        from mlx_launcher.core.persistence import chats as chats_store
        from mlx_launcher.screens.chat import ChatScreen

        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.2)
            screen = app.screen
            screen._create_chat()
            a_id = screen.chat.id

            # backend appends turns to A AFTER our cache was read → self.data is now stale
            await chats_store.mutate(_append_reply(chats_store, a_id))

            # create another chat WITHOUT a resync first — _create_chat must re-read, not clobber
            screen._create_chat()
            await pilot.pause(0.05)

            reloaded = chats_store.get_chat(chats_store.load(), a_id)
            assert reloaded is not None
            assert any(m.text == "important reply" for m in reloaded.messages), \
                "creating a new chat clobbered the prior chat's backend-persisted turns"

    asyncio.run(go())
