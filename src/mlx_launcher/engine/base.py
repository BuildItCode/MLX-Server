"""The format-engine seam.

``Engine`` is the structural contract the backend depends on, so the agent loop never names a
concrete client. ``EngineConfig`` captures everything needed to talk to one model on one server,
and ``build_engine`` constructs the default OpenAI-compatible adapter from it. To target a
different OpenAI-compatible server, only ``base_url`` changes; to target a non-OpenAI engine,
provide another object satisfying ``Engine`` and a matching ``build_engine`` branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional, Protocol, runtime_checkable


@dataclass
class EngineConfig:
    """Per-request connection + sampling config for one model on one server."""

    base_url: str
    model: str
    api_key: str = "not-needed"
    max_tokens: Optional[int] = None
    chat_template_kwargs: Optional[dict] = None
    sampling: Optional[dict] = None


@runtime_checkable
class Engine(Protocol):
    """A streaming + non-streaming chat-completions client.

    The default implementation is :class:`mlx_launcher.engine.openai.OpenAIEngine`; any object
    exposing these two coroutines is a drop-in (that is what makes the engine swappable)."""

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield ``('content'|'reason', text)`` chunks, then a final ``('finish', reason)``."""
        ...

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        *,
        read_timeout: float = 600.0,
    ) -> dict:
        """One non-streaming completion; returns the parsed OpenAI response JSON."""
        ...


def build_engine(cfg: EngineConfig) -> Engine:
    """Construct the default OpenAI-compatible engine from a config."""
    from .openai import OpenAIEngine

    return OpenAIEngine(
        cfg.base_url,
        cfg.model,
        cfg.api_key,
        max_tokens=cfg.max_tokens,
        chat_template_kwargs=cfg.chat_template_kwargs,
        sampling=cfg.sampling,
    )
