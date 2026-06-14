"""Heuristics for model capabilities + attachment encoding.

Capability detection from a model name is necessarily a heuristic — the UI uses
it only to set sensible defaults; the user can always toggle reasoning and attach
files regardless."""

from __future__ import annotations

import base64
import json
import os
import re

# Substrings that suggest a vision-capable (multimodal) model.
_VISION_HINTS = (
    "vl", "vision", "llava", "pixtral", "internvl", "smolvlm", "moondream",
    "gemma-3", "idefics", "molmo", "aria", "multimodal", "janus", "minicpm-v",
    "llama-3.2-11b", "llama-3.2-90b", "phi-3.5-vision", "phi-4-multimodal",
)

# Substrings that suggest a reasoning / "thinking" model.
_REASONING_HINTS = (
    "deepseek-r1", "r1-distill", "-r1", "qwq", "qwen3", "magistral",
    "openthinker", "glm-z1", "skywork-o1", "marco-o1", "phi-4-reasoning",
    "phi-4-mini-reasoning", "thinking", "reasoner",
)

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _norm(model: str) -> str:
    return (model or "").lower()


def supports_vision(model: str) -> bool:
    m = _norm(model)
    return any(h in m for h in _VISION_HINTS)


def supports_reasoning(model: str) -> bool:
    m = _norm(model)
    return any(h in m for h in _REASONING_HINTS)


def image_mime(path: str) -> str | None:
    return _IMAGE_MIME.get(os.path.splitext(path)[1].lower())


def classify(path: str) -> str:
    return "image" if image_mime(path) else "text"


def encode_image(path: str) -> str | None:
    """Return a data: URL for an image file, or None if unreadable/not an image."""
    mime = image_mime(path)
    if not mime or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def read_text_attachment(path: str, limit: int = 200_000) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return "[could not read file]"


# --- context window -----------------------------------------------------

_CTX_CONFIG_KEYS = (
    "max_position_embeddings", "max_sequence_length", "n_positions",
    "seq_length", "model_max_length", "max_seq_len", "n_ctx",
)


def context_window(model: str) -> int | None:
    """Best-effort context window (in tokens), or None if it can't be determined.

    Reads a local model's `config.json` first (authoritative), then falls back to
    a `…-128k…` style hint in the name. None ⇒ the UI hides the context bar."""
    path = os.path.expanduser(model or "")
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
            for src in (data, data.get("text_config") or {}, data.get("llm_config") or {}):
                for key in _CTX_CONFIG_KEYS:
                    val = src.get(key)
                    if isinstance(val, int) and 0 < val <= 100_000_000:
                        return val
        except (OSError, ValueError, AttributeError):
            pass
    m = re.search(r"(\d+)\s*k(?![a-z])", _norm(model))  # e.g. "qwen2.5-7b-128k"
    if m:
        return int(m.group(1)) * 1024
    return None


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — no tokenizer dependency."""
    return (len(text) + 3) // 4


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Approximate the tokens a chat-completions request would use. Image parts
    are counted at a flat per-image cost (they dominate but vary by model)."""
    total = 0
    for m in messages:
        total += 4  # per-message framing overhead
        content = m.get("content")
        if isinstance(content, str):
            total += approx_tokens(content)
        elif isinstance(content, list):  # multimodal parts
            for part in content:
                if part.get("type") == "text":
                    total += approx_tokens(part.get("text", ""))
                else:
                    total += 800  # rough image cost
    return total
