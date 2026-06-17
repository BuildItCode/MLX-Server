"""Typed events emitted by the unified agent loop (:mod:`mlx_launcher.core.agent`).

The loop is a transport-agnostic async generator: it *yields* these observational events and never
touches a UI. Each frontend renders what it cares about — the TUI mounts widgets, the ACP agent
maps them to session updates, and the HTTP service serializes them as Server-Sent Events (the
``type`` field is the SSE event name; :meth:`Event.to_dict` is the JSON payload).

The two *interactive* decisions (permission to run a mutating tool, opening a URL) are NOT events —
they are injected async callbacks on the runner, because the loop must block on the answer. See
:mod:`mlx_launcher.core.agent`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union


@dataclass
class RunStarted:
    """A run (one user turn → loop) began."""

    kind: str = "chat"  # chat | compact | regenerate
    type: str = field(default="run_start", init=False)


@dataclass
class ContentDelta:
    """A chunk of the assistant's answer."""

    text: str
    type: str = field(default="content", init=False)


@dataclass
class ReasonDelta:
    """A chunk of the model's reasoning / thinking."""

    text: str
    type: str = field(default="reason", init=False)


@dataclass
class ToolStarted:
    """A tool call is about to run. ``phrase`` is the human description (see core.tools.phrasing)."""

    tool_id: str
    name: str
    phrase: str
    args: dict = field(default_factory=dict)
    type: str = field(default="tool_start", init=False)


@dataclass
class ToolFinished:
    """A tool call finished. ``status`` is ok | error | denied; ``preview`` is a truncated result."""

    tool_id: str
    name: str
    result: str
    status: str = "ok"
    preview: str = ""
    type: str = field(default="tool_end", init=False)


@dataclass
class UsageUpdated:
    """Context-window usage after the turn (for a usage bar). ``window`` is None when unknown."""

    used: int
    window: Optional[int] = None
    type: str = field(default="usage", init=False)


@dataclass
class Notice:
    """A transient message for the user (e.g. 'native tools failed — using prompted tools')."""

    level: str  # info | warning | error
    text: str
    type: str = field(default="notice", init=False)


@dataclass
class PermissionRequest:
    """The loop wants to run a mutating tool and needs the user's decision. Emitted by the service
    onto the run's event stream; answered by POSTing ``{"decision": "once"|"all"|"deny"}`` to the
    correlated ``…/permissions/{id}`` endpoint."""

    id: str
    name: str
    summary: str
    detail: str = ""
    type: str = field(default="permission_request", init=False)


@dataclass
class OpenUrlRequested:
    """The model asked to open a URL (open_in_browser). A frontend opens it on its UI thread; a
    headless client ignores it."""

    url: str
    type: str = field(default="open_url", init=False)


@dataclass
class TurnFinished:
    """The run completed. ``reason`` is an OpenAI-style finish reason
    (stop | length | cancelled | content_filter | tool_calls)."""

    text: str
    reason: str = "stop"
    n_tool_calls: int = 0
    elapsed: float = 0.0
    reasoning: str = ""
    type: str = field(default="finish", init=False)


@dataclass
class TurnFailed:
    """The run errored. ``fatal`` distinguishes an engine crash (OOM/reshape) from a recoverable
    error, mirroring the frontend's ``_is_fatal_generation_error`` gate."""

    error: str
    fatal: bool = False
    type: str = field(default="error", init=False)


Event = Union[
    RunStarted,
    ContentDelta,
    ReasonDelta,
    ToolStarted,
    ToolFinished,
    UsageUpdated,
    Notice,
    TurnFinished,
    TurnFailed,
]


def to_dict(event: Event) -> dict:
    """The JSON payload for an event (its ``type`` plus its fields), for the SSE wire format."""
    return asdict(event)
