"""Human-readable phrasing for tool calls (backend layer).

``_tool_phrase`` turns a tool call into a transcript line ("Reading src/app.py"); ``_perm_prompt``
builds the summary + detail shown before a mutating action. Pure presentation-of-domain helpers —
the backend computes them so every frontend renders the same wording. Moved out of
``screens/chat.py``; re-imported there so ``mlx_launcher.screens.chat._tool_phrase`` still resolves."""

from __future__ import annotations

import json


def _perm_prompt(name: str, args: dict) -> tuple[str, str]:
    """A human summary + detail preview for a file/command permission prompt."""
    if name == "write_file":
        content = args.get("content", "")
        return f"Write file  {args.get('path', '?')}  ({len(content)} chars)", content[:500]
    if name == "edit_file":
        return (f"Edit file  {args.get('path', '?')}",
                f"- {args.get('old_text', '')[:200]}\n+ {args.get('new_text', '')[:200]}")
    if name == "delete_path":
        return f"Delete  {args.get('path', '?')}", ""
    if name == "run_command":
        return "Run command", args.get("command", "")[:500]
    if name == "open_in_browser":
        return f"Open in browser  {args.get('path') or args.get('url', '?')}", ""
    return name, json.dumps(args)[:500]


def _tool_phrase(name: str, args: dict) -> str:
    """A natural-language description of a tool call for the transcript — 'Reading src/app.py'
    instead of 'read_file'. Falls back to a humanized identifier for MCP / unknown tools."""
    a = args or {}
    path = str(a.get("path") or "").strip()
    if name == "read_file":
        return f"Reading {path}" if path else "Reading a file"
    if name == "write_file":
        return f"Writing {path}" if path else "Writing a file"
    if name == "edit_file":
        return f"Editing {path}" if path else "Editing a file"
    if name == "list_directory":
        return f"Listing {path or '.'}"
    if name == "delete_path":
        return f"Deleting {path}" if path else "Deleting a path"
    if name == "run_command":
        cmd = str(a.get("command") or "").strip()
        return f"Running  {cmd[:60]}" if cmd else "Running a command"
    if name == "open_in_browser":
        target = str(a.get("path") or a.get("url") or "").strip()
        return f"Opening {target}" if target else "Opening in the browser"
    if name == "web_search":
        query = str(a.get("query") or "").strip()
        return f"Searching the web for “{query}”" if query else "Searching the web"
    label = (name or "").replace("_", " ").replace("-", " ").strip() or "a tool"  # MCP / unknown
    return label[:1].upper() + label[1:]
