"""Filesystem tools for the chat's agentic loop, confined to a project's
**working directory**. Every path is resolved inside that root; attempts to
escape it (via ``..`` or absolute paths) are rejected. This is what lets a
project be used as a coding workspace — the model can list/read/create/edit/
delete files and run commands, all within the chosen folder.

`run_command` runs a real shell in the working directory (with a timeout); it is
only ever offered when the user has explicitly set a working directory."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

MAX_READ_BYTES = 64_000
MAX_OUTPUT = 8_000
COMMAND_TIMEOUT = 120.0

FS_TOOL_NAMES = {
    "list_directory",
    "read_file",
    "write_file",
    "edit_file",
    "delete_path",
    "run_command",
}

SYSTEM_NOTE = (
    "You have file tools scoped to this project's working directory:\n  {root}\n"
    "Use list_directory, read_file, write_file, edit_file, delete_path, and run_command "
    "to work in it. All paths are RELATIVE to the working directory.\n"
    "BEFORE starting any task, check for an AGENTS.md file in the working directory "
    "(read_file AGENTS.md) and follow its conventions and instructions if it exists.\n"
    "Inspect files before editing, make focused changes, and briefly explain what you did."
)


def system_note(root: str) -> str:
    """The filesystem system prompt for `root`, flagging AGENTS.md when present."""
    note = SYSTEM_NOTE.format(root=root)
    if os.path.isfile(os.path.join(os.path.expanduser(root), "AGENTS.md")):
        note += "\nNOTE: this project HAS an AGENTS.md — read it FIRST and follow it before anything else."
    return note


def fs_specs() -> list[dict]:
    """OpenAI function-tool specs for the filesystem tools."""

    def spec(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    return [
        spec("list_directory", "List files and folders. 'path' is relative to the working directory ('.' = root).",
             {"path": {"type": "string", "description": "relative directory path; default '.'"}}, []),
        spec("read_file", "Read a UTF-8 text file and return its contents.",
             {"path": {"type": "string", "description": "relative file path"}}, ["path"]),
        spec("write_file", "Create or overwrite a text file. Parent folders are created as needed.",
             {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        spec("edit_file", "Replace the first exact occurrence of old_text with new_text in a file.",
             {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
             ["path", "old_text", "new_text"]),
        spec("delete_path", "Delete a file, or a directory and its contents, inside the working directory.",
             {"path": {"type": "string"}}, ["path"]),
        spec("run_command", "Run a shell command in the working directory; returns its combined stdout/stderr.",
             {"command": {"type": "string"}}, ["command"]),
    ]


# --- path confinement ----------------------------------------------------

def _root(root: str) -> Path:
    return Path(os.path.expanduser(root)).resolve()


def _resolve(root: str, rel: str) -> Path:
    """Resolve `rel` under the working root, rejecting anything that escapes it."""
    base = _root(root)
    target = (base / (rel or ".")).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes the working directory: {rel!r}")
    return target


# --- operations ----------------------------------------------------------

def _list_directory(root: str, rel: str) -> str:
    p = _resolve(root, rel)
    if not p.exists():
        return f"not found: {rel}"
    if p.is_file():
        return f"{rel} is a file, not a directory"
    items = sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
    lines = [f"{c.name}/" if c.is_dir() else c.name for c in items]
    return "\n".join(lines) or "(empty directory)"


def _read_file(root: str, rel: str) -> str:
    p = _resolve(root, rel)
    if not p.is_file():
        return f"not a file: {rel}"
    raw = p.read_bytes()
    text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    if len(raw) > MAX_READ_BYTES:
        text += f"\n… (truncated at {MAX_READ_BYTES} bytes)"
    return text


def _write_file(root: str, rel: str, content: str) -> str:
    p = _resolve(root, rel)
    if p == _root(root):
        return "refusing to overwrite the working directory itself"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {rel} ({len(content)} chars)"


def _edit_file(root: str, rel: str, old: str, new: str) -> str:
    p = _resolve(root, rel)
    if not p.is_file():
        return f"not a file: {rel}"
    if not old:
        return "old_text must not be empty"
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        return f"old_text not found in {rel}"
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited {rel}" + ("" if count == 1 else f" (replaced the first of {count} matches)")


def _delete_path(root: str, rel: str) -> str:
    p = _resolve(root, rel)
    if p == _root(root):
        return "refusing to delete the working directory itself"
    if not p.exists():
        return f"not found: {rel}"
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"deleted {rel}"


async def _run_command(root: str, command: str) -> str:
    if not command.strip():
        return "empty command"
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(_root(root)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return f"command timed out after {int(COMMAND_TIMEOUT)}s"
    text = out.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + "\n… (output truncated)"
    return f"[exit {proc.returncode}]\n{text}".strip()


async def run_fs_tool(root: str, name: str, args: dict) -> str:
    """Dispatch one filesystem tool call, confined to `root`."""
    try:
        if name == "list_directory":
            return await asyncio.to_thread(_list_directory, root, args.get("path", ".") or ".")
        if name == "read_file":
            return await asyncio.to_thread(_read_file, root, args["path"])
        if name == "write_file":
            return await asyncio.to_thread(_write_file, root, args["path"], args.get("content", ""))
        if name == "edit_file":
            return await asyncio.to_thread(_edit_file, root, args["path"], args.get("old_text", ""), args.get("new_text", ""))
        if name == "delete_path":
            return await asyncio.to_thread(_delete_path, root, args["path"])
        if name == "run_command":
            return await _run_command(root, args.get("command", ""))
        return f"unknown filesystem tool: {name}"
    except KeyError as exc:
        return f"missing required argument: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
