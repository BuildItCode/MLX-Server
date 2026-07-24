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

# Qwen / DeepSeek / gpt-oss models trained on the "function=" protocol emit:
#     <tool_call>
#     <function=mcp__playwright__browser_snapshot>{"url": "…"}</function>
#     </tool_call>
# (args may be absent for zero-arg tools, and some servers drop the closing
# </function> — so we also accept </tool_call> as the terminator). This is
# NOT Hermes JSON-in-<tool_call> and NOT MiniMax XML — parse it separately.
_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[A-Za-z0-9_.\-]+)>(?P<args>.*?)(?:</function>|</tool_call>)",
    re.DOTALL | re.IGNORECASE,
)


def parse_function_tool_calls(text: str) -> list[dict]:
    """[{name, arguments}] from the Qwen/DeepSeek `<function=name>{json args?}` format.
    Args are optional (zero-arg tools like browser_snapshot emit none) and must be a
    JSON object when present — malformed args degrade to {} rather than dropping the call."""
    out: list[dict] = []
    for m in _FUNCTION_RE.finditer(text or ""):
        name = m.group("name").strip()
        if not name:
            continue
        raw = (m.group("args") or "").strip()
        args: dict = {}
        if raw:
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                args = {}
        out.append({"name": name, "arguments": args})
    return out

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


def _type_annotation(schema: dict) -> str:
    """JSON Schema property → short type string for the model (``str``, ``int``, ``bool``, …)."""
    _MAP = {"string": "str", "integer": "int", "number": "float",
            "boolean": "bool", "array": "list", "object": "dict", "null": "None"}
    return _MAP.get((schema or {}).get("type", ""), (schema or {}).get("type", ""))


def tool_instructions(specs: list[dict]) -> str:
    """A system-prompt block describing the tools and the call protocol.

    Each tool is rendered with parameter *types* (``path: str``, ``fullPage?: bool``)
    and inline per-parameter descriptions, so the model can emit correctly-shaped
    arguments instead of guessing — critical for MCP servers whose schemas carry
    rich type/enum/description metadata that was previously discarded.
    """
    lines = [
        "# Tools",
        "You can call tools. To call one, output a tag EXACTLY like this (you may emit several):",
        '<tool_call>{"name": "<tool_name>", "arguments": {<json args>}}</tool_call>',
        "Each result returns as a <tool_response> message. When you are finished "
        "with all steps, reply normally with NO <tool_call> tag.",
        "",
        "CRITICAL: a reply with no <tool_call> tag ENDS your turn — nothing further "
        "happens until the user replies. Never narrate what you are ABOUT to do "
        "(\"Let me click…\", \"I will now open…\") and then stop: emit the <tool_call> "
        "tag for that action in the SAME reply, right after any prose. If your plan "
        "has more steps, keep going — one <tool_call> per step, several tags per "
        "reply is fine — and only write a plain reply when the whole task is done.",
        "NEVER claim a step succeeded or report what a page/tool showed before that "
        "step's <tool_response> has actually arrived — in a reply that contains "
        "<tool_call> tags, prose describes WHAT you are doing, not outcomes. Your "
        "final answer must be based ONLY on real tool results, never on what you "
        "expected to happen.",
        "WEB/BROWSER RULE: every web or browser action — navigating to a URL, opening "
        "or reading a page, clicking, typing, scrolling, snapshotting, screenshotting, "
        "or evaluating JavaScript — MUST go through the Playwright MCP tools "
        "(mcp__playwright__browser_*). There is NO direct browser tool here: never "
        "fetch or render pages with run_command (curl/wget), never open an http(s) URL "
        "with open_in_browser (it opens the user's OS browser and is for local files "
        "you created only), and never use browser tools from any non-playwright MCP "
        "server. web_search returns search-result snippets ONLY — to open or read "
        "anything it finds, navigate there with mcp__playwright__browser_navigate.",
        "",
        "Available tools:",
    ]
    for s in specs:
        fn = s.get("function") or {}
        params = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        # Render: path: str, fullPage?: bool  (was: path, fullPage?)
        arg_parts = []
        for pname, pschema in params.items():
            ann = _type_annotation(pschema)
            suffix = "" if pname in required else "?"
            arg_parts.append(f"{pname}{suffix}: {ann}" if ann else f"{pname}{suffix}")
        arglist = ", ".join(arg_parts)
        desc = fn.get("description", "")
        lines.append(f"- {fn.get('name', '?')}({arglist}): {desc}")
        # Per-parameter descriptions + enum values (so the model knows what each arg expects)
        for pname, pschema in params.items():
            pdesc = (pschema or {}).get("description", "")
            enum_vals = (pschema or {}).get("enum")
            if enum_vals:
                lines.append(f"    {pname}: one of {enum_vals}")
            elif pdesc:
                lines.append(f"    {pname}: {pdesc}")
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
    elif isinstance(args, list):
        # Some models emit arguments as a positional list instead of a dict — wrap it
        # so the call is not silently dropped (the model clearly intended to pass args).
        args = {"items": args}
    if isinstance(name, str) and name and isinstance(args, dict):
        return {"name": name, "arguments": args}
    return None


def parse_tool_calls(text: str) -> list[dict]:
    """Extract [{name, arguments}] from a model's text reply. Tolerant of the `<tool_call>` tag,
    a bare fenced JSON object, the Qwen `<function=name>` protocol, and MiniMax-M2's `<invoke>`
    XML form."""
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
    return out or parse_function_tool_calls(text) or parse_xml_tool_calls(text)


def strip_tool_calls(text: str) -> str:
    """Remove the tool-call tags (Hermes <tool_call>, Qwen <function=…>, and MiniMax XML)
    so only prose remains."""
    text = _TAG_RE.sub("", text or "")
    text = _MINIMAX_BLOCK_RE.sub("", text)
    text = _INVOKE_RE.sub("", text)  # a stray <invoke> block whose wrapper was dropped
    text = _FUNCTION_RE.sub("", text)  # Qwen <function=name> blocks
    # Orphaned <tool_call> wrapper tags left behind when the body was a Qwen
    # <function=…> block (the JSON Hermes regex above can't match those).
    text = re.sub(r"</?tool_call\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def tool_response(name: str, result: str) -> str:
    """Format a tool result to feed back to the model.

    The result text is NOT XML-escaped — the model needs to see it verbatim (code,
    JSON, HTML, snapshot trees). A literal ``</tool_response>`` inside the result would
    break parsing, but that string essentially never occurs in real tool output. We
    accept the risk rather than escaping (which would make JSON/code unreadable to
    the model) — the parser's ``re.DOTALL`` non-greedy match means only the FIRST
    ``</tool_response>`` ends the block, so a real closing tag followed by trailing
    content is the only failure shape, and that content is always the next turn's
    user message which has its own role boundary.
    """
    return f'<tool_response name="{name}">\n{result}\n</tool_response>'
