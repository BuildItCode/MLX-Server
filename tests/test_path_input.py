import os

from omnicode.widgets.path_input import path_hint, resolve_path, sanitize_drag


def test_sanitize_strips_quotes():
    assert sanitize_drag("  '/a/b c'  ") == "/a/b c"
    assert sanitize_drag('"/a/b"') == "/a/b"


def test_sanitize_unescapes_backslashes_when_unquoted():
    assert sanitize_drag("/Users/me/My\\ Model") == "/Users/me/My Model"


def test_sanitize_is_idempotent():
    once = sanitize_drag("/Users/me/My\\ Model")
    assert sanitize_drag(once) == once


def test_resolve_expands_user():
    assert resolve_path("~").startswith("/")


def test_hint_detects_kinds(tmp_path):
    assert "folder exists" in path_hint(str(tmp_path))
    assert "HuggingFace" in path_hint("mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert "not found" in path_hint("/no/such/path/xyz123")
    assert path_hint("") == ""

    f = tmp_path / "weights.bin"
    f.write_text("x")
    assert "file exists" in path_hint(str(f))


def test_working_dir_modal_accepts_drop_when_input_unfocused():
    """Regression (2026-07-24): the working-directory modal silently dropped paths pasted
    while focus was on OK/Cancel instead of the input — the drop worked in the add-model
    editor (which has a screen-level on_paste fallback) but not here. Both must behave
    the same: folder → folder, file → parent dir, non-path paste passes through."""
    import asyncio
    import os
    import tempfile

    from textual.app import App

    from omnicode.screens.chat import TextPromptModal

    class Host(App):
        pass

    class FakePaste:
        def __init__(self, text):
            self.text = text
            self.stopped = False

        def stop(self):
            self.stopped = True

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "My Project")
            os.makedirs(sub)
            f = os.path.join(sub, "page.html")
            open(f, "w").write("x")

            app = Host()
            async with app.run_test() as pilot:
                modal = TextPromptModal("Working directory")
                await app.push_screen(modal)
                await pilot.pause()

                # focus on a button, not the input — the previously lost case
                modal.query_one("#ok").focus()
                ev = FakePaste(sub + "/")
                modal.on_paste(ev)
                await pilot.pause()
                assert ev.stopped
                assert modal.query_one("#modal-input").value == sub

                # a dropped file resolves to its parent dir
                modal.query_one("#cancel").focus()
                ev2 = FakePaste(f"'{f}'")
                modal.on_paste(ev2)
                await pilot.pause()
                assert modal.query_one("#modal-input").value == sub

                # non-path text is not swallowed
                ev3 = FakePaste("hello world")
                modal.on_paste(ev3)
                assert not ev3.stopped

    asyncio.run(run())
