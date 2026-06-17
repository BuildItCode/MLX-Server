"""Streaming / format parsers for OpenAI-compatible model output.

Separates a model's "thinking" from its answer — inline ``<think>…</think>`` (ThinkSplitter)
and OpenAI **Harmony** channel markup (HarmonyParser, gpt-oss) — and recovers tool calls that
weaker servers emit as plain text instead of structured ``tool_calls``. Pure functions, no I/O;
the engine's HTTP adapter (``engine.openai``) and the agent loop layer above feed text through
these. Moved out of ``chat/client.py``; re-exported there for back-compat."""

from __future__ import annotations

import json
import re
from typing import Optional


class ThinkSplitter:
    """Incrementally splits a content stream on <think>…</think>, holding back a
    partial closing/opening tag that may straddle two chunks."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.pending = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.pending += text
        out: list[tuple[str, str]] = []
        while self.pending:
            tag = self.CLOSE if self.in_think else self.OPEN
            idx = self.pending.find(tag)
            if idx != -1:
                before = self.pending[:idx]
                if before:
                    out.append(("reason" if self.in_think else "content", before))
                self.pending = self.pending[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            hold = self._partial_suffix(self.pending, tag)
            emit = self.pending[: len(self.pending) - hold]
            if emit:
                out.append(("reason" if self.in_think else "content", emit))
            self.pending = self.pending[len(self.pending) - hold:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.pending:
            return []
        kind = "reason" if self.in_think else "content"
        out = [(kind, self.pending)]
        self.pending = ""
        return out

    @staticmethod
    def _partial_suffix(s: str, tag: str) -> int:
        """Length of the longest suffix of s that is a prefix of tag."""
        for k in range(min(len(s), len(tag) - 1), 0, -1):
            if s[-k:] == tag[:k]:
                return k
        return 0


class HarmonyParser:
    """Incrementally parse OpenAI **Harmony** channel markup (gpt-oss models) that
    some servers stream verbatim instead of parsing, e.g.

        <|channel|>analysis<|message|>…reasoning…<|end|>
        <|start|>assistant<|channel|>final<|message|>…answer…<|return|>

    Routes the ``final`` channel to ``content`` and ``analysis``/``commentary`` to
    ``reason``, stripping every control token. Text containing no Harmony tokens
    passes straight through as ``content``, so ordinary models are unaffected."""

    _TOKENS = (
        "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
        "<|constrain|>", "<|return|>", "<|call|>",
    )
    _ROLE = "\x00role"  # sentinel: a message header's body (echoed role turn) → dropped

    def __init__(self) -> None:
        self.pending = ""
        self.mode = "body"  # body | role | channel | constrain | none
        self.channel: Optional[str] = None
        self._meta = ""  # accumulates a channel name across chunks

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.pending += text
        out: list[tuple[str, str]] = []
        while self.pending:
            tok, idx = self._next_token(self.pending)
            if tok is None:
                hold = self._partial_suffix(self.pending)
                emit = self.pending[: len(self.pending) - hold]
                self._consume(emit, out)
                self.pending = self.pending[len(self.pending) - hold:]
                break
            self._consume(self.pending[:idx], out)
            self._apply(tok)
            self.pending = self.pending[idx + len(tok):]
        return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.pending:
            self._consume(self.pending, out)
            self.pending = ""
        return out

    def _consume(self, text: str, out: list[tuple[str, str]]) -> None:
        if not text:
            return
        if self.mode == "body":
            if self.channel in (None, "final"):
                out.append(("content", text))  # None = pre-Harmony passthrough (normal models)
            elif self.channel == self._ROLE:
                pass  # echoed system/user/assistant header turn — drop it
            else:
                out.append(("reason", text))  # analysis / commentary
        elif self.mode == "channel":
            self._meta += text  # building up the channel name

    def _apply(self, tok: str) -> None:
        if self.mode == "channel":  # a token ends the channel name we were reading
            self.channel = self._meta.strip() or self.channel
            self._meta = ""
        if tok == "<|channel|>":
            self.mode = "channel"
            self._meta = ""
        elif tok == "<|message|>":
            self.mode = "body"
        elif tok == "<|constrain|>":
            self.mode = "constrain"
        elif tok == "<|start|>":
            self.mode = "role"
            self.channel = self._ROLE  # body before an explicit channel is a role echo
        else:  # <|end|> / <|return|> / <|call|>
            self.mode = "none"
            self.channel = None

    def _next_token(self, s: str) -> tuple[Optional[str], Optional[int]]:
        best_tok, best_idx = None, None
        for t in self._TOKENS:
            i = s.find(t)
            if i != -1 and (best_idx is None or i < best_idx):
                best_tok, best_idx = t, i
        return best_tok, best_idx

    def _partial_suffix(self, s: str) -> int:
        """Longest suffix of s that is a prefix of some control token — held back
        in case the token completes in the next chunk."""
        hold = 0
        for t in self._TOKENS:
            for k in range(min(len(s), len(t) - 1), 0, -1):
                if s[-k:] == t[:k]:
                    hold = max(hold, k)
                    break
        return hold


_STRIPPED_LEAD_RE = re.compile(r"\s*(?:analysis|commentary)(?=\S)")


def recover_stripped_harmony(text: str) -> Optional[tuple[str, str]]:
    """Some servers decode gpt-oss Harmony with the ``<|...|>`` control tokens removed
    but the channel/role NAMES left glued inline, e.g.
        ``analysis{reasoning}assistantfinal{answer}``
    so the reasoning leaks into the answer. Recover (content, reasoning) from that,
    or return None when the text isn't that shape (so normal prose is untouched)."""
    if not text or "<|" in text:  # literal-token form → HarmonyParser handles it
        return None
    has_lead = bool(_STRIPPED_LEAD_RE.match(text))
    # 'assistantfinal' glued is unambiguous; allow a single space only with a lead name.
    # We require the explicit 'assistant…final' role marker (the real stripped form always
    # has it) rather than splitting on a bare 'final' — that fallback corrupted ordinary
    # answers opening with "analysis" that contain a word like "finalize"/"finalist"
    # (e.g. "analysis. The finalists were chosen." → answer cut mid-word).
    m = re.search(r"assistant ?final", text) if has_lead else re.search(r"assistantfinal", text)
    if m:
        reason = _STRIPPED_LEAD_RE.sub("", text[:m.start()], count=1)
        return text[m.end():].strip(), reason.strip()
    return None


def parse_harmony(text: str) -> tuple[str, str]:
    """One-shot: split a full Harmony string into (final_content, reasoning).
    Handles both the literal ``<|channel|>`` form and the token-stripped
    ``analysis…assistantfinal…`` form. Returns (text, "") for plain prose."""
    p = HarmonyParser()
    pieces = p.feed(text) + p.flush()
    content = "".join(t for k, t in pieces if k == "content")
    reason = "".join(t for k, t in pieces if k == "reason")
    if "<|" not in (text or ""):  # no real control tokens — maybe the stripped form
        stripped = recover_stripped_harmony(text or "")
        if stripped is not None:
            return stripped
    return content, reason


# gpt-oss emits tool calls in the Harmony *commentary* channel, e.g.
#   <|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{"query":"…"}<|call|>
# mlx_lm.server has no gpt-oss tool parser (has_tool_calling=False), so it returns
# this verbatim in `content` instead of as a native `tool_calls` entry — leaving the
# call unexecuted and the answer empty. We recover the call from the raw text.
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.(?P<name>[A-Za-z0-9_.\-]+)"          # recipient: functions.<name>
    r".*?"                                                # rest of header (<|constrain|>json, ws)
    r"<\|message\|>(?P<args>.*?)"                         # the JSON arguments
    r"(?=<\|call\|>|<\|end\|>|<\|return\|>|<\|start\|>|\Z)",
    re.DOTALL,
)


def _loads_lenient(raw: str) -> dict:
    """Parse a JSON object, salvaging the first balanced {...} if there's trailing junk."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        pass
    start = raw.find("{")
    if start == -1:
        return {}
    # raw_decode parses ONE JSON value from `start` and ignores trailing junk — and it's
    # string-aware, so a `}` inside a string value (e.g. {"content": "if (x) { }"}) doesn't
    # prematurely end the object the way naive brace-counting did.
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw, start)
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_harmony_tool_calls(text: str) -> list[dict]:
    """Extract [{name, arguments}] from Harmony `commentary to=functions.*` tool
    calls that a server returned as plain text. Empty list when there is none."""
    out: list[dict] = []
    for m in _HARMONY_CALL_RE.finditer(text or ""):
        out.append({"name": m.group("name"), "arguments": _loads_lenient(m.group("args"))})
    return out


# Last-resort recovery: a known tool name immediately followed by a JSON object, for
# servers that strip the Harmony delimiters. gpt-oss on mlx_lm can leave its call as
# e.g.  "…Use web_search function.{\"query\": …}"  with no <|call|> token to match.
# The bridge between the name and the `{` may be ONLY whitespace/punctuation and the
# optional literal word "function" (gpt-oss's stripped artifact) — never arbitrary
# prose — so a sentence that merely explains a tool ("call read_file with a path like
# {…}") is not misread as a real, executable call.
_LOOSE_BRIDGE = re.compile(r"""[\s(]*(?:function)?[\s.:=)"'`]*\{""")


def recover_loose_tool_calls(text: str, tool_names: list[str]) -> list[dict]:
    """[{name, arguments}] for any KNOWN tool name followed immediately by a parseable
    JSON object. Deliberately conservative — only whitespace/punctuation (and gpt-oss's
    "function" artifact) may bridge to the `{`, known names only, and the JSON must
    actually parse — so ordinary prose that merely mentions a tool isn't misread as a
    call. Empty list when nothing fits."""
    text = text or ""
    out: list[dict] = []
    for name in tool_names:
        if not name:
            continue
        # Scan EVERY occurrence of the name, not just the first: gpt-oss commonly narrates
        # ("I'll use web_search …") before emitting the actual `web_search{…}` call, so the
        # first mention is prose and the real call is later. Take the first that parses.
        start = 0
        while True:
            idx = text.find(name, start)
            if idx == -1:
                break
            start = idx + len(name)
            tail = text[idx + len(name):]
            bridge = _LOOSE_BRIDGE.match(tail)
            if bridge is None:
                continue
            obj = _loads_lenient(tail[bridge.end() - 1:])  # from the '{'
            if obj:
                out.append({"name": name, "arguments": obj})
                break
    return out


def recover_json_tool_calls(text: str, tool_names: list[str]) -> list[dict]:
    """Last-resort: bare JSON tool-call objects ``{"name": "<known tool>", "arguments": {…}}``
    found anywhere in the text — for a model that drifts out of its tagged format into a loose
    shape like ``[calling tool: {"name": …, "arguments": …}]``. Keyed on KNOWN tool names and
    requiring an arguments object, so prose / example JSON isn't misread. Empty list when none fit."""
    names = {n for n in (tool_names or []) if n}
    if not names:
        return []
    out: list[dict] = []
    dec = json.JSONDecoder()
    s = text or ""
    i = 0
    while True:
        idx = s.find("{", i)
        if idx == -1:
            break
        try:  # string-aware: a '{' inside a string value won't trip up the scan
            obj, end = dec.raw_decode(s, idx)
        except ValueError:
            i = idx + 1
            continue
        i = max(end, idx + 1)
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments")
            if args is None:
                args = obj.get("args") if obj.get("args") is not None else obj.get("parameters")
            if name in names and isinstance(args, dict):
                out.append({"name": name, "arguments": args})
    return out


def _render_tool_calls(calls: list[dict]) -> str:
    """Persisted tool calls → the text-protocol <tool_call> tags the loop understands."""
    return "\n".join(
        "<tool_call>" + json.dumps({"name": c.get("name", ""), "arguments": c.get("arguments") or {}})
        + "</tool_call>"
        for c in (calls or [])
    )
