"""Data models for the chat front-end: projects, chats, messages, attachments."""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    role: Literal["user", "assistant", "system", "tool"]
    text: str = ""
    reasoning: str = ""  # stored "thinking" content, if any
    attachments: list[Attachment] = Field(default_factory=list)
    ts: float = Field(default_factory=_now)
    # generation stats (assistant messages)
    tps: Optional[float] = None
    n_tokens: Optional[int] = None
    elapsed: Optional[float] = None
    # agentic tool steps, persisted so a follow-up turn ("continue") keeps the work context.
    # On an assistant turn: the calls it made, as [{"name": str, "arguments": dict}, ...].
    # On a `tool` turn: the result, with tool_name naming the tool that produced it.
    tool_calls: Optional[list[dict]] = None
    tool_name: Optional[str] = None


class Chat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    title: str = "New chat"
    project_id: Optional[str] = None  # legacy — migrated to working_dir + instructions
    server_id: Optional[str] = None  # the ServerConfig the main agent targets
    subagent_server_id: Optional[str] = None  # the ServerConfig subagents use (None = inherit main)
    base_url: str = ""
    model: str = ""
    skill_id: Optional[str] = None  # active skill injected as system guidance
    reasoning: bool = False  # show/stream the model's thinking
    reasoning_effort: Optional[str] = None  # off|low|medium|high; None = model/template default
    web_search: bool = False  # allow the model to call the web_search tool
    # build = make changes, ask before each file/command action; plan = propose a plan, take no
    # actions; auto = make changes and run tools WITHOUT asking (always auto-approve).
    mode: Literal["build", "plan", "auto"] = "build"
    # NOTE: ``tools`` (MCP) and ``coding`` (senior-engineer persona) used to be opt-in toggles.
    # They are now ALWAYS ON for every model, so the fields were removed and the chips dropped
    # from the chat screen. MCP tools are always wired up when a connector is configured.
    working_dir: Optional[str] = None  # if set, the agent gets file tools scoped here
    instructions: str = ""  # custom system-prompt instructions for this chat
    subagent_ids: list[str] = Field(default_factory=list)  # legacy; kept for back-compat (unused)
    messages: list[ChatMessage] = Field(default_factory=list)
    created: float = Field(default_factory=_now)
    updated: float = Field(default_factory=_now)

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data):
        if not isinstance(data, dict):
            return data
        # back-compat: chats saved before the 3-way mode used a `plan_mode` bool
        if "mode" not in data and data.get("plan_mode"):
            data = {**data, "mode": "plan"}
        return data


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


class Subagent(BaseModel):
    """A named specialist that the main model can delegate to via the ``task`` tool
    (deepagents subagent). The main agent decides when to delegate based on the
    ``description``; the subagent runs in an isolated context window with its own
    model (``server_id``), system prompt, and capabilities, then returns only its
    final result to the main agent's context."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    name: str = "subagent"  # unique identifier (used as the task tool's subagent_type)
    description: str = ""  # what this subagent does — the main model uses this to decide delegation
    system_prompt: str = ""  # specialized instruction for this subagent
    server_id: Optional[str] = None  # the model this subagent runs on (None = inherit chat's subagent_server_id or main)
    web_search: bool = False  # may call the built-in web_search tool
    tools: bool = False  # may call MCP connections
    mcp_server_ids: list[str] = Field(default_factory=list)  # which MCP servers it uses
    skill_ids: list[str] = Field(default_factory=list)  # skills injected into its prompt


class ChatStoreFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    projects: list[Project] = Field(default_factory=list)
    chats: list[Chat] = Field(default_factory=list)
    mcp_servers: list[McpServer] = Field(default_factory=list)
    subagents: list[Subagent] = Field(default_factory=list)
