"""Data models for the chat front-end: projects, chats, messages, attachments."""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


class Attachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    name: str = ""
    kind: Literal["image", "text"] = "text"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    text: str = ""
    reasoning: str = ""  # stored "thinking" content, if any
    attachments: list[Attachment] = Field(default_factory=list)
    ts: float = Field(default_factory=_now)
    # generation stats (assistant messages)
    tps: Optional[float] = None
    n_tokens: Optional[int] = None
    elapsed: Optional[float] = None


class Chat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    title: str = "New chat"
    project_id: Optional[str] = None
    server_id: Optional[str] = None  # the ServerConfig this chat targets
    base_url: str = ""
    model: str = ""
    skill_id: Optional[str] = None  # active skill injected as system guidance
    reasoning: bool = False  # show/stream the model's thinking
    web_search: bool = False  # allow the model to call the web_search tool
    tools: bool = False  # allow the model to call MCP server tools
    plan_mode: bool = False  # plan-only: produce a plan for approval, take no actions
    messages: list[ChatMessage] = Field(default_factory=list)
    created: float = Field(default_factory=_now)
    updated: float = Field(default_factory=_now)


class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    name: str = "Untitled project"
    instructions: str = ""  # used as a system prompt for its chats
    working_dir: Optional[str] = None  # if set, chats get file tools scoped here
    created: float = Field(default_factory=_now)


class McpServer(BaseModel):
    """A Model Context Protocol server the chat models can call tools on."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    name: str = "server"
    enabled: bool = True
    transport: Literal["stdio", "sse"] = "stdio"
    command: str = ""  # stdio: executable
    args: str = ""  # stdio: shlex-split arguments
    env: str = ""  # stdio: "KEY=VALUE KEY2=VALUE2"
    url: str = ""  # sse: endpoint URL


class ChatStoreFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    projects: list[Project] = Field(default_factory=list)
    chats: list[Chat] = Field(default_factory=list)
    mcp_servers: list[McpServer] = Field(default_factory=list)
