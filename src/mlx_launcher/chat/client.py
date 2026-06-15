"""Streaming chat client for the front-end.

Wraps MlxBridge.stream_chat, separating the model's "thinking" from its answer:
reasoning arrives either as OpenAI `reasoning_content` deltas (handled by the
bridge) or as inline `<think>…</think>` in the content (handled here by an
incremental splitter)."""

from __future__ import annotations

import json
import re
from typing import AsyncIterator, Callable, Optional

from ..acp.bridge import MlxBridge
from . import capabilities
from .models import Chat, ChatMessage, Project


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
    # 'assistantfinal' glued is unambiguous; allow a single space only with a lead name
    m = re.search(r"assistant ?final", text) if has_lead else re.search(r"assistantfinal", text)
    if m:
        reason = _STRIPPED_LEAD_RE.sub("", text[:m.start()], count=1)
        return text[m.end():].strip(), reason.strip()
    if has_lead:  # leading channel name, no assistant marker → split on a glued 'final'
        fm = re.search(r"final(?=\S)", text)
        if fm and fm.start() > 0:
            reason = _STRIPPED_LEAD_RE.sub("", text[:fm.start()], count=1)
            return text[fm.end():].strip(), reason.strip()
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
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else {}
                except ValueError:
                    return {}
    return {}


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
_LOOSE_BRIDGE = re.compile(r"[\s.:=)\"'`a-zA-Z_]{0,40}\{")


def recover_loose_tool_calls(text: str, tool_names: list[str]) -> list[dict]:
    """[{name, arguments}] for any KNOWN tool name followed closely by a parseable
    JSON object. Deliberately conservative — a short, word/punctuation-only bridge to
    the `{`, known names only, and the JSON must actually parse — so ordinary prose
    that merely mentions a tool isn't misread as a call. Empty list when nothing fits."""
    text = text or ""
    out: list[dict] = []
    for name in tool_names:
        if not name:
            continue
        idx = text.find(name)
        if idx == -1:
            continue
        tail = text[idx + len(name):]
        bridge = _LOOSE_BRIDGE.match(tail)
        if bridge is None:
            continue
        obj = _loads_lenient(tail[bridge.end() - 1:])  # from the '{'
        if obj:
            out.append({"name": name, "arguments": obj})
    return out


def _message_to_openai(m: ChatMessage) -> dict:
    if m.role == "assistant":
        return {"role": "assistant", "content": m.text}

    text = m.text
    for att in m.attachments:
        if att.kind == "text":
            body = capabilities.read_text_attachment(att.path)
            text += f'\n\n<file name="{att.name or att.path}">\n{body}\n</file>'

    images = [a for a in m.attachments if a.kind == "image"]
    if images:
        parts: list[dict] = [{"type": "text", "text": text}]
        for att in images:
            url = capabilities.encode_image(att.path)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": m.role, "content": parts}
    return {"role": m.role, "content": text}


PLAN_MODE_INSTRUCTIONS = (
    "You are in PLAN MODE — like a senior engineer scoping work before touching anything.\n"
    "- Do NOT make changes, write or edit files, or call tools that modify state. "
    "Read-only investigation is fine.\n"
    "- Think the task through, then present a clear, step-by-step PLAN: what you would do, "
    "which files/commands are involved, and any trade-offs or open questions.\n"
    "- If the request is ambiguous, ask brief clarifying questions before presenting the plan.\n"
    "- END by asking the user to approve the plan or tell you what to change. Do NOT begin "
    "implementing until the user explicitly approves."
)

CODING_MODE_INSTRUCTIONS = (
    "You are a senior software engineer. Write correct, idiomatic, production-quality code "
    "that matches the surrounding style, naming, and conventions of the codebase.\n"
    "- VALIDATE before you claim something works. When a working directory and tools are "
    "available, run the project's own checks — type-check, lint, build, and tests "
    "(e.g. `tsc --noEmit`, `npm run lint`, `npm test`, `pytest`, `cargo check`, `go vet`) — "
    "and FIX everything they surface. Never report success on code you have not verified.\n"
    "- Reuse existing functions, utilities, and patterns instead of adding new ones; read the "
    "relevant code before changing it.\n"
    "- Handle errors and edge cases. Do not leave TODOs, stubs, or placeholder implementations "
    "unless the user asks for them.\n"
    "- Keep changes focused and minimal; don't refactor unrelated code.\n"
    "- If a requirement is ambiguous or you must assume something, state the assumption briefly. "
    "Explain only non-obvious decisions, and keep explanations concise."
)


def prepend_system(messages: list[dict], note: str) -> list[dict]:
    """Fold `note` into the system prompt as ONE leading system message — merging
    into an existing system message rather than adding a second one. Many chat
    templates (e.g. Qwen) raise "System message must be at the beginning" — a 500
    from mlx_lm.server — when given two leading system turns, so we never emit two."""
    if not note:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{note}\n\n---\n\n{messages[0]['content']}"}
    else:
        messages.insert(0, {"role": "system", "content": note})
    return messages


def build_openai_messages(
    chat: Chat,
    project: Optional[Project] = None,
    skill_instructions: Optional[str] = None,
) -> list[dict]:
    msgs: list[dict] = []
    parts = [
        p.strip()
        for p in (skill_instructions, project.instructions if project else "")
        if p and p.strip()
    ]
    if getattr(chat, "coding", False):
        parts.append(CODING_MODE_INSTRUCTIONS)
    if getattr(chat, "plan_mode", False):
        parts.append(PLAN_MODE_INSTRUCTIONS)  # last = the most salient framing
    if parts:
        msgs.append({"role": "system", "content": "\n\n---\n\n".join(parts)})
    for m in chat.messages:
        msgs.append(_message_to_openai(m))
    return msgs


# A reasoning model spends its token budget on the analysis channel *before* the
# answer, so the server's 512-token default leaves nothing for the reply. Ask for
# a real budget; a profile's own --max-tokens (if set) overrides this in chat.py.
DEFAULT_MAX_TOKENS = 16384


class ChatClient:
    def __init__(
        self, base_url: str, model: str, api_key: str = "not-needed", max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        self.bridge = MlxBridge(base_url, model, api_key, max_tokens=max_tokens)

    async def stream(
        self,
        messages: list[dict],
        *,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield ('reason', text) / ('content', text) / ('finish', reason).

        Content passes through a Harmony parser (gpt-oss channels) and then the
        <think> splitter, so both reasoning conventions land in the thinking
        panel and only the real answer shows as content."""
        harmony = HarmonyParser()
        think = ThinkSplitter()

        def route(piece: str) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            for hk, htext in harmony.feed(piece):
                if hk == "reason":
                    out.append(("reason", htext))
                else:
                    out.extend(think.feed(htext))
            return out

        async for kind, chunk in self.bridge.stream_chat(messages, cancel=cancel):
            if kind == "content":
                for item in route(chunk):
                    yield item
            elif kind == "reason":
                yield ("reason", chunk)
            elif kind == "finish":
                for hk, htext in harmony.flush():
                    if hk == "reason":
                        yield ("reason", htext)
                    else:
                        for item in think.feed(htext):
                            yield item
                for item in think.flush():
                    yield item
                yield ("finish", chunk)
