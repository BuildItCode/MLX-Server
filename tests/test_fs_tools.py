import asyncio
import os

from mlx_launcher.chat import fs_tools


def test_specs_cover_all_tool_names():
    names = {s["function"]["name"] for s in fs_tools.fs_specs()}
    assert names == fs_tools.FS_TOOL_NAMES


def test_write_read_edit_delete_list(tmp_path):
    root = str(tmp_path)

    async def go():
        assert "wrote" in await fs_tools.run_fs_tool(
            root, "write_file", {"path": "src/app.py", "content": "print('hi')\n"})
        assert (tmp_path / "src" / "app.py").read_text() == "print('hi')\n"  # really on disk
        assert "print('hi')" in await fs_tools.run_fs_tool(root, "read_file", {"path": "src/app.py"})
        assert "src/" in await fs_tools.run_fs_tool(root, "list_directory", {"path": "."})
        assert "edited" in await fs_tools.run_fs_tool(
            root, "edit_file", {"path": "src/app.py", "old_text": "hi", "new_text": "bye"})
        assert "bye" in (tmp_path / "src" / "app.py").read_text()
        assert "deleted" in await fs_tools.run_fs_tool(root, "delete_path", {"path": "src/app.py"})
        assert not (tmp_path / "src" / "app.py").exists()

    asyncio.run(go())


def test_path_escape_is_rejected(tmp_path):
    root = str(tmp_path / "work")
    os.makedirs(root)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")

    async def go():
        r1 = await fs_tools.run_fs_tool(root, "read_file", {"path": "../secret.txt"})
        assert "escapes" in r1 and "TOPSECRET" not in r1
        r2 = await fs_tools.run_fs_tool(root, "read_file", {"path": str(secret)})  # absolute
        assert "escapes" in r2 and "TOPSECRET" not in r2
        r3 = await fs_tools.run_fs_tool(root, "write_file", {"path": "../evil.txt", "content": "x"})
        assert "escapes" in r3
        assert not (tmp_path / "evil.txt").exists()
        r4 = await fs_tools.run_fs_tool(root, "delete_path", {"path": "../secret.txt"})
        assert "escapes" in r4 and secret.exists()

    asyncio.run(go())


def test_edit_missing_text_and_file(tmp_path):
    root = str(tmp_path)

    async def go():
        await fs_tools.run_fs_tool(root, "write_file", {"path": "a.txt", "content": "hello world"})
        assert "not found" in await fs_tools.run_fs_tool(
            root, "edit_file", {"path": "a.txt", "old_text": "zzz", "new_text": "q"})
        assert "not a file" in await fs_tools.run_fs_tool(
            root, "read_file", {"path": "missing.txt"})

    asyncio.run(go())


def test_system_note_instructs_agents_md(tmp_path):
    root = str(tmp_path)
    note = fs_tools.system_note(root)
    assert "AGENTS.md" in note and "read_file AGENTS.md" in note
    assert "HAS an AGENTS.md" not in note  # none present yet
    (tmp_path / "AGENTS.md").write_text("be terse")
    assert "HAS an AGENTS.md" in fs_tools.system_note(root)


def test_run_command_in_working_dir(tmp_path):
    root = str(tmp_path)

    async def go():
        await fs_tools.run_fs_tool(root, "write_file", {"path": "hello.txt", "content": "hi"})
        out = await fs_tools.run_fs_tool(root, "run_command", {"command": "ls"})
        assert "hello.txt" in out and "[exit 0]" in out

    asyncio.run(go())
