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


def build_openai_messages(chat: Chat, project: Optional[Project] = None) -> list[dict]:
    msgs: list[dict] = []
    instructions = (project.instructions if project else "").strip()
    if instructions:
        msgs.append({"role": "system", "content": instructions})
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
        """Yield ('reason', text) / ('content', text) / ('finish', reason)."""
        splitter = ThinkSplitter()
        async for kind, chunk in self.bridge.stream_chat(messages, cancel=cancel):
            if kind == "content":
                for k, piece in splitter.feed(chunk):
                    yield (k, piece)
            elif kind == "reason":
                yield ("reason", chunk)
            elif kind == "finish":
                for k, piece in splitter.flush():
                    yield (k, piece)
                yield ("finish", chunk)
