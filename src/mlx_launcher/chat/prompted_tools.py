"""Prompted (text-protocol) tool calling — a model-agnostic fallback.

Native function calling depends on the model's chat template rendering the
OpenAI `tools` param and the server parsing the model's tool-call format back.
When that fails for a given model/template, we fall back to *instructing* the
model: describe the tools in the system prompt and ask it to emit Hermes-style
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` tags, which we parse
ourselves. This works for any instruction-following model — no template tool
support required."""

from __future__ import annotations

import json
import re

_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.DOTALL)


def tool_instructions(specs: list[dict]) -> str:
    """A system-prompt block describing the tools and the call protocol."""
    lines = [
        "# Tools",
        "You can call tools. To call one, output a tag EXACTLY like this (you may emit several):",
        '<tool_call>{"name": "<tool_name>", "arguments": {<json args>}}</tool_call>',
        "Then STOP and wait — each result returns as a <tool_response> message. "
        "When you are finished, reply normally with NO <tool_call> tag.",
        "",
        "Available tools:",
    ]
    for s in specs:
        fn = s.get("function") or {}
        params = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        arglist = ", ".join(f"{k}" if k in required else f"{k}?" for k in params)
        lines.append(f"- {fn.get('name', '?')}({arglist}): {fn.get('description', '')}")
    return "\n".join(lines)


def _coerce(obj: dict) -> dict | None:
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments")
    if args is None:
        args = obj.get("args") if obj.get("args") is not None else obj.get("parameters")
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if isinstance(name, str) and name and isinstance(args, dict):
        return {"name": name, "arguments": args}
    return None


def parse_tool_calls(text: str) -> list[dict]:
    """Extract [{name, arguments}] from a model's text reply. Tolerant of the
    `<tool_call>` tag and, failing that, a bare fenced JSON object."""
    out: list[dict] = []
    matches = _TAG_RE.findall(text or "") or _FENCE_RE.findall(text or "")
    for raw in matches:
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        call = _coerce(obj) if isinstance(obj, dict) else None
        if call:
            out.append(call)
    return out


def strip_tool_calls(text: str) -> str:
    """Remove the tool-call tags so only the model's prose remains."""
    return _TAG_RE.sub("", text or "").strip()


def tool_response(name: str, result: str) -> str:
    """Format a tool result to feed back to the model."""
    return f'<tool_response name="{name}">\n{result}\n</tool_response>'
