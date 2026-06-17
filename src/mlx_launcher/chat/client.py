"""Streaming chat client facade for the front-end.

Re-export shim. The format parsers now live in :mod:`mlx_launcher.engine.streaming`, the OpenAI
HTTP adapter in :mod:`mlx_launcher.engine.openai`, and the OpenAI-message assembly + token
budgeting in :mod:`mlx_launcher.core.messages` / :mod:`mlx_launcher.core.instructions`. This
module re-exports them and keeps the :class:`ChatClient` streaming facade so existing
``mlx_launcher.chat.client`` imports keep working."""

from __future__ import annotations

from typing import AsyncIterator, Callable, Optional

from ..core.instructions import CODING_MODE_INSTRUCTIONS, PLAN_MODE_INSTRUCTIONS  # noqa: F401
from ..core.messages import (  # noqa: F401
    DEFAULT_MAX_TOKENS,
    _coalesce_roles,
    _message_to_openai,
    build_openai_messages,
    prepend_system,
    scaled_max_tokens,
)
from ..engine.openai import MlxBridge
from ..engine.streaming import (  # noqa: F401
    HarmonyParser,
    ThinkSplitter,
    _loads_lenient,
    _render_tool_calls,
    parse_harmony,
    parse_harmony_tool_calls,
    recover_json_tool_calls,
    recover_loose_tool_calls,
    recover_stripped_harmony,
)
# Bind to the engine modules (NOT the chat shims) so `cl.capabilities` is the SAME object
# that core.messages uses — a test monkeypatching `cl.capabilities.context_window` must reach
# scaled_max_tokens, which now lives in core.messages.
from ..engine import capabilities, prompted as prompted_tools  # noqa: F401



class ChatClient:
    def __init__(
        self, base_url: str, model: str, api_key: str = "not-needed",
        max_tokens: int = DEFAULT_MAX_TOKENS, chat_template_kwargs: Optional[dict] = None,
        sampling: Optional[dict] = None,
    ) -> None:
        self.bridge = MlxBridge(
            base_url, model, api_key, max_tokens=max_tokens,
            chat_template_kwargs=chat_template_kwargs, sampling=sampling,
        )

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
