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

# MiniMax-M2 emits its own XML tool-call format (it ignores the Hermes <tool_call> instruction and
# uses what it was trained on), e.g.
#     <minimax:tool_call>
#     <invoke name="read_file"><parameter name="path">app.py</parameter></invoke>
#     </minimax:tool_call>
# Multiple <invoke> blocks may appear in one wrapper; parameter values are plain text that may be
# JSON (arrays/objects) or a bare string. We scan <invoke> globally so a missing/garbled wrapper
# (some servers drop it) still parses.
_MINIMAX_BLOCK_RE = re.compile(r"<minimax:tool_call>.*?</minimax:tool_call>", re.DOTALL | re.IGNORECASE)
_INVOKE_RE = re.compile(r'<invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</invoke>', re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(r'<parameter\s+name="(?P<name>[^"]+)"\s*>(?P<value>.*?)</parameter>', re.DOTALL | re.IGNORECASE)


def _coerce_param(raw: str):
    """A MiniMax <parameter> value: parsed as JSON when it parses (arrays/objects/numbers/bools),
    else the literal trimmed string — matching the format's "plain string or JSON" contract."""
    s = (raw or "").strip()
    try:
        return json.loads(s)
    except ValueError:
        return s


def parse_xml_tool_calls(text: str) -> list[dict]:
    """[{name, arguments}] from MiniMax-M2's <invoke name=…><parameter name=…>…</invoke> XML.
    Empty list when the text isn't that shape, so ordinary prose is never misread as a call."""
    out: list[dict] = []
    for inv in _INVOKE_RE.finditer(text or ""):
        name = inv.group("name").strip()
        if not name:
            continue
        args = {p.group("name").strip(): _coerce_param(p.group("value"))
                for p in _PARAM_RE.finditer(inv.group("body"))}
        out.append({"name": name, "arguments": args})
    return out


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
    """Extract [{name, arguments}] from a model's text reply. Tolerant of the `<tool_call>` tag,
    a bare fenced JSON object, and MiniMax-M2's `<invoke>` XML form."""
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
    return out or parse_xml_tool_calls(text)  # fall back to MiniMax's XML tool-call format


def strip_tool_calls(text: str) -> str:
    """Remove the tool-call tags (Hermes <tool_call> and MiniMax XML) so only prose remains."""
    text = _TAG_RE.sub("", text or "")
    text = _MINIMAX_BLOCK_RE.sub("", text)
    text = _INVOKE_RE.sub("", text)  # a stray <invoke> block whose wrapper was dropped
    return text.strip()


def tool_response(name: str, result: str) -> str:
    """Format a tool result to feed back to the model."""
    return f'<tool_response name="{name}">\n{result}\n</tool_response>'
