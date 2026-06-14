"""Streaming chat client for the front-end.

Wraps MlxBridge.stream_chat, separating the model's "thinking" from its answer:
reasoning arrives either as OpenAI `reasoning_content` deltas (handled by the
bridge) or as inline `<think>…</think>` in the content (handled here by an
incremental splitter)."""

from __future__ import annotations

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


def parse_harmony(text: str) -> tuple[str, str]:
    """One-shot: split a full Harmony string into (final_content, reasoning).
    Returns (text, "") unchanged when there is no Harmony markup."""
    p = HarmonyParser()
    pieces = p.feed(text) + p.flush()
    content = "".join(t for k, t in pieces if k == "content")
    reason = "".join(t for k, t in pieces if k == "reason")
    return content, reason


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
    if getattr(chat, "plan_mode", False):
        parts.append(PLAN_MODE_INSTRUCTIONS)  # last = the most salient framing
    if parts:
        msgs.append({"role": "system", "content": "\n\n---\n\n".join(parts)})
    for m in chat.messages:
        msgs.append(_message_to_openai(m))
    return msgs


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed") -> None:
        self.bridge = MlxBridge(base_url, model, api_key)

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
