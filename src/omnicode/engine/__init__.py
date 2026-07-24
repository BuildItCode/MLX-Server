"""The **format engine**: an OpenAI-compatible adapter plus the format-quirk handling
(Harmony / ``<think>`` parsing, tool-call extraction in every dialect, the prompted-protocol
fallback, and model-capability heuristics).

This is the bottom layer. It is swappable to any OpenAI-compatible server via ``base_url`` and,
behind the :class:`Engine` protocol, to non-OpenAI adapters later. It imports nothing from the
``core`` (backend) or frontend layers — the dependency arrows point only inward."""

from .base import Engine, EngineConfig, build_engine
from .extract import Extraction, extract_tool_calls
from .openai import MlxBridge, OpenAIEngine, fetch_models

__all__ = [
    "Engine",
    "EngineConfig",
    "build_engine",
    "Extraction",
    "extract_tool_calls",
    "MlxBridge",
    "OpenAIEngine",
    "fetch_models",
]
