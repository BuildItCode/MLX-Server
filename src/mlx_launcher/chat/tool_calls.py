"""The inbound tool-call "protocol" layer — the ONE place that understands every way a model or
server can express a tool call, so the agentic loop stays format-agnostic.

Good servers (and cloud providers) return STRUCTURED OpenAI ``tool_calls`` and this is trivial.
Weaker local servers (mlx_lm / mlx-vlm running gpt-oss, MiniMax, …) have no tool parser for those
models and return the call as TEXT in the message content, so we recover it. The loop calls
``extract_tool_calls()`` once and only has to decide native-vs-text feedback — it never touches a
format regex itself.

Formats handled, in order (each lives in a focused, separately-tested function):
  • native        — the server returned OpenAI ``tool_calls``                     (best case)
  • Hermes/MiniMax — ``<tool_call>{json}</tool_call>`` or MiniMax ``<invoke>`` XML  (prompted_tools)
  • Harmony       — gpt-oss commentary channel ``to=functions.x<|message|>{…}``    (client)
  • loose         — a known tool name immediately followed by a JSON object         (client)
  • json          — a drifted bare ``{"name": "<known tool>", "arguments": {…}}``   (client)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from . import prompted_tools
from .client import (
    parse_harmony,
    parse_harmony_tool_calls,
    recover_json_tool_calls,
    recover_loose_tool_calls,
)


@dataclass
class Extraction:
    """What a model's reply contained: the tool calls it made (if any), its prose, and HOW the
    calls arrived — structured ``tool_calls`` from the server (``is_native`` → feed results back as
    the ``tool`` role) vs. recovered from text (→ feed results back as user ``<tool_response>``)."""

    calls: list           # normalized [{"name", "arguments"}]; empty ⇒ the model gave a final answer
    content: str          # the model's prose (Harmony channels split out; tool markup still in `content`)
    reason: str           # reasoning / analysis text
    native: list          # the raw OpenAI tool_calls when the server returned them structured, else []
    finish: Optional[str]  # finish_reason ("length" ⇒ the turn was truncated mid-output)

    @property
    def is_native(self) -> bool:
        return bool(self.native)


def _from_native(call: dict) -> dict:
    fn = call.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return {"name": fn.get("name", ""), "arguments": args}


def extract_tool_calls(message: dict, finish: Optional[str], tool_names: list) -> Extraction:
    """Read one chat-completion ``message`` into normalized tool calls. Prefers the server's
    structured ``tool_calls``; otherwise recovers the call from the content text in any known
    format. Returns an Extraction with empty ``calls`` when the model gave a plain answer."""
    raw = message.get("content") or ""
    native = message.get("tool_calls") or []
    content, reason = parse_harmony(raw)
    if native:
        return Extraction([_from_native(c) for c in native], content, reason, native, finish)
    recovered = (prompted_tools.parse_tool_calls(content)            # Hermes <tool_call> / MiniMax XML
                 or parse_harmony_tool_calls(raw)                    # gpt-oss Harmony commentary channel
                 or recover_loose_tool_calls(content, tool_names)    # bare `name {json}`
                 or recover_json_tool_calls(content, tool_names))    # drifted {"name":…,"arguments":…}
    return Extraction(list(recovered), content, reason, [], finish)
