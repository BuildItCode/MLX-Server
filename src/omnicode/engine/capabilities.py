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
    "phi-4-mini-reasoning", "thinking", "reasoner", "gpt-oss", "gpt_oss",
)

# Models whose chat template gates thinking via an `enable_thinking` bool (Qwen3-style),
# as opposed to gpt-oss's graded `reasoning_effort`.
_ENABLE_THINKING_HINTS = ("qwen3", "qwen-3")

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


# Reasoning-effort cycle exposed in the chat UI. None = "auto" (the model/template default).
REASONING_EFFORTS = (None, "off", "low", "medium", "high")


def reasoning_template_kwargs(model: str, effort: str | None) -> dict:
    """Translate a reasoning-effort choice into the `chat_template_kwargs` the model's own
    chat template understands. Sent in the request body — mlx_lm.server / vLLM forward it to
    `apply_chat_template`. Returns {} only for 'auto' (the model/template default).

    An EXPLICIT effort is sent even when the name-based heuristic doesn't flag the model as a
    reasoner — a kwarg a template doesn't reference is harmless (just unused Jinja context), and
    this way reasoning models we don't recognize by name (e.g. Step) still respond to the control.
    gpt-oss uses a graded `reasoning_effort` (low|medium|high); Qwen3-style templates use an
    `enable_thinking` bool; everything else gets `reasoning_effort`."""
    if not effort:
        return {}
    m = _norm(model)
    if "gpt-oss" in m or "gpt_oss" in m:
        return {"reasoning_effort": "low" if effort == "off" else effort}  # can't fully disable
    if any(h in m for h in _ENABLE_THINKING_HINTS):
        return {"enable_thinking": effort != "off"}
    return {} if effort == "off" else {"reasoning_effort": effort}


def image_mime(path: str) -> str | None:
    return _IMAGE_MIME.get(os.path.splitext(path)[1].lower())


def classify(path: str) -> str:
    return "image" if image_mime(path) else "text"


_MAX_IMAGE_BYTES = 16_000_000  # skip absurdly large images rather than bloating the request


def encode_image(path: str) -> str | None:
    """Return a data: URL for an image file, or None if unreadable/not an image/too big."""
    mime = image_mime(path)
    if not mime or not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) > _MAX_IMAGE_BYTES:
            return None
        with open(path, "rb") as f:
            data = f.read(_MAX_IMAGE_BYTES)
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
    # a "…-128k…" style hint: digits bounded by non-alnum on both sides, then 'k'
    m = re.search(r"(?<![a-z0-9])(\d{1,4})\s*k(?![a-z0-9])", _norm(model))  # e.g. "qwen2.5-7b-128k"
    if m:
        win = int(m.group(1)) * 1024
        if 1024 <= win <= 100_000_000:
            return win
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
                if isinstance(part, str):
                    total += approx_tokens(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    total += approx_tokens(part.get("text", ""))
                else:
                    total += 800  # rough image cost
    return total
