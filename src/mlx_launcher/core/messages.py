"""OpenAI-message assembly + token budgeting (backend layer).

Turns a :class:`~mlx_launcher.models.chat.Chat` (history, mode, project + skill instructions,
attachments) into the OpenAI ``messages`` list the engine sends, folds all system guidance into
ONE leading system message, and scales the per-request ``max_tokens`` to the model's context
window. Pure, transport-agnostic; depends only on ``engine`` + ``models`` + ``core.instructions``.
Moved out of ``chat/client.py``; re-exported there for back-compat."""

from __future__ import annotations

from typing import Optional

from ..engine import capabilities
from ..engine import prompted as prompted_tools
from ..engine.streaming import _render_tool_calls
from ..models.chat import Chat, ChatMessage, Project
from .instructions import CODING_MODE_INSTRUCTIONS, PLAN_MODE_INSTRUCTIONS


def _message_to_openai(m: ChatMessage) -> dict:
    if m.role == "tool":
        # A persisted tool result → replayed as the text-protocol <tool_response>. Every chat
        # template renders user turns, so this works regardless of native tool support, and the
        # loop accepts native/harmony/text calls alike on the way back out.
        return {"role": "user", "content": prompted_tools.tool_response(m.tool_name or "tool", m.text)}
    if m.role == "assistant":
        content = m.text
        if m.tool_calls:  # an agentic turn that called tools — keep the calls in the history
            tags = _render_tool_calls(m.tool_calls)
            content = f"{content}\n{tags}".strip() if content.strip() else tags
        return {"role": "assistant", "content": content}

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


def prepend_system(messages: list[dict], note: str) -> list[dict]:
    """Fold `note` into the system prompt as ONE leading system message — merging
    into an existing system message rather than adding a second one. Many chat
    templates (e.g. Qwen) raise "System message must be at the beginning" — a 500
    from mlx_lm.server — when given two leading system turns, so we never emit two."""
    if not note:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{note}\n\n---\n\n{messages[0].get('content', '')}"}
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
    if getattr(chat, "mode", "build") == "plan":
        parts.append(PLAN_MODE_INSTRUCTIONS)  # last = the most salient framing
    if parts:
        msgs.append({"role": "system", "content": "\n\n---\n\n".join(parts)})
    for m in chat.messages:
        msgs.append(_message_to_openai(m))
    return _coalesce_roles(msgs)


def _coalesce_roles(msgs: list[dict]) -> list[dict]:
    """Merge adjacent same-role turns with plain-text content into one. The agentic loop
    persists tool steps as assistant/user turns, which can leave two consecutive user turns
    (a tool result, then the next user message); strict role-alternating templates (Qwen) 500
    on that. Multimodal (list) content is never merged."""
    out: list[dict] = []
    for m in msgs:
        prev = out[-1] if out else None
        if (prev and prev["role"] == m["role"]
                and isinstance(prev.get("content"), str) and isinstance(m.get("content"), str)):
            out[-1] = {**prev, "content": f"{prev['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    return out


# A reasoning model spends its token budget on the analysis channel *before* the
# answer, so the server's 512-token default leaves nothing for the reply. Ask for
# a real budget; a profile's own --max-tokens (if set) overrides this in chat.py.
DEFAULT_MAX_TOKENS = 16384  # fallback only — used when the context window can't be determined

# The generation budget scales with the available context instead of a fixed 16k. Bounds keep it
# sane at the extremes: a reasoning model isn't starved on a small context, and a huge context
# can't license a turn that generates for minutes.
_MIN_SCALED_MAX_TOKENS = 4096    # floor: leave room for a reasoning model to actually answer
_MAX_SCALED_MAX_TOKENS = 65536   # ceiling: bound a single turn (truncation continues across turns)


def scaled_max_tokens(model: str, context_cap: Optional[int] = None) -> int:
    """Per-request ``max_tokens`` scaled to the context window: ~1/4 of an explicit KV-cache / ctx
    cap the user configured, else ~1/6 of the model's max context. Floored and capped (see above),
    never larger than the window itself, and DEFAULT_MAX_TOKENS when the window is unknown."""
    model_max = capabilities.context_window(model)
    if context_cap:
        window = min(context_cap, model_max) if model_max else context_cap
        budget = window // 4
    elif model_max:
        window = model_max
        budget = window // 6
    else:
        return DEFAULT_MAX_TOKENS
    budget = max(_MIN_SCALED_MAX_TOKENS, min(budget, _MAX_SCALED_MAX_TOKENS))
    return min(budget, window)  # can't generate more than the whole window
