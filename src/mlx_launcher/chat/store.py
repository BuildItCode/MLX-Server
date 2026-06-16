"""Persistence for chats + projects (~/.config/mlx-launcher/chats.json)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from ..config.store import config_dir
from .models import Chat, ChatStoreFile, McpServer, Project, Subagent


def chats_path() -> Path:
    return config_dir() / "chats.json"


def load() -> ChatStoreFile:
    path = chats_path()
    if not path.exists():
        return ChatStoreFile()
    try:
        return ChatStoreFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        try:
            path.rename(path.with_name(f"chats.corrupt-{int(time.time())}.json"))
        except Exception:
            pass
        return ChatStoreFile()


def save(data: ChatStoreFile) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = chats_path()
    tmp = path.with_name(path.name + ".tmp")
    # flush + fsync before the atomic rename so a crash/power-loss can't leave the temp
    # (and thus the renamed file) truncated — os.replace alone doesn't guarantee the data
    # blocks are on disk before the directory entry is swapped.
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# --- helpers -------------------------------------------------------------

def get_chat(data: ChatStoreFile, chat_id: str) -> Optional[Chat]:
    return next((c for c in data.chats if c.id == chat_id), None)


def upsert_chat(data: ChatStoreFile, chat: Chat) -> None:
    for i, c in enumerate(data.chats):
        if c.id == chat.id:
            data.chats[i] = chat
            return
    data.chats.insert(0, chat)


def delete_chat(data: ChatStoreFile, chat_id: str) -> None:
    data.chats = [c for c in data.chats if c.id != chat_id]


def get_project(data: ChatStoreFile, project_id: str) -> Optional[Project]:
    return next((p for p in data.projects if p.id == project_id), None)


def upsert_project(data: ChatStoreFile, project: Project) -> None:
    for i, p in enumerate(data.projects):
        if p.id == project.id:
            data.projects[i] = project
            return
    data.projects.insert(0, project)


def delete_project(data: ChatStoreFile, project_id: str) -> None:
    """Delete a project and detach (don't delete) its chats."""
    data.projects = [p for p in data.projects if p.id != project_id]
    for c in data.chats:
        if c.project_id == project_id:
            c.project_id = None


def upsert_mcp(data: ChatStoreFile, server: McpServer) -> None:
    for i, s in enumerate(data.mcp_servers):
        if s.id == server.id:
            data.mcp_servers[i] = server
            return
    data.mcp_servers.append(server)


def delete_mcp(data: ChatStoreFile, server_id: str) -> None:
    data.mcp_servers = [s for s in data.mcp_servers if s.id != server_id]


def get_subagent(data: ChatStoreFile, sub_id: str) -> Optional[Subagent]:
    return next((s for s in data.subagents if s.id == sub_id), None)


def upsert_subagent(data: ChatStoreFile, sub: Subagent) -> None:
    for i, s in enumerate(data.subagents):
        if s.id == sub.id:
            data.subagents[i] = sub
            return
    data.subagents.append(sub)


def delete_subagent(data: ChatStoreFile, sub_id: str) -> None:
    data.subagents = [s for s in data.subagents if s.id != sub_id]
    for c in data.chats:  # detach from every chat so no dangling ids remain
        if sub_id in c.subagent_ids:
            c.subagent_ids = [x for x in c.subagent_ids if x != sub_id]


def chats_in(data: ChatStoreFile, project_id: Optional[str]) -> list[Chat]:
    """Chats for a project, or all chats when project_id is None."""
    items = [c for c in data.chats if (project_id is None or c.project_id == project_id)]
    return sorted(items, key=lambda c: c.updated, reverse=True)
