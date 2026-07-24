"""Context compaction (backend layer).

When a conversation approaches the model's context window, summarize the earlier turns into a
compact brief and REPLACE them, so the chat can keep going. Pure helpers + one engine call;
transport-agnostic (the frontend decides *when* to trigger between runs and persists the result)."""

from __future__ import annotations

from typing import Optional

from .instructions import COMPACT_INSTRUCTIONS, COMPACT_USER_MARKER

# Trigger auto-compaction once the prompt fills this fraction of the window (between runs only).
AUTOCOMPACT_THRESHOLD = 0.95
# Don't bother compacting a tiny window or an almost-empty chat.
_MIN_WINDOW = 2000
_MIN_TURNS = 3


def should_autocompact(used: int, window: Optional[int], n_turns: int) -> bool:
    """True when the prompt has grown past the threshold of a known, non-trivial window and there
    are enough real turns to be worth summarizing. The caller checks this between runs."""
    if not window or window < _MIN_WINDOW or n_turns < _MIN_TURNS:
        return False
    return used >= AUTOCOMPACT_THRESHOLD * window


def summary_request(messages: list[dict]) -> list[dict]:
    """The message list to send for a compaction summary: the conversation so far + a final user
    turn asking for the brief."""
    return [*messages, {"role": "user", "content": COMPACT_INSTRUCTIONS}]


def replacement_turns(summary: str) -> list[dict]:
    """The two turns that REPLACE the compacted history — a visible user marker + the summary as an
    assistant turn (a valid user→assistant pair keeps alternating-role templates happy)."""
    return [
        {"role": "user", "content": COMPACT_USER_MARKER},
        {"role": "assistant", "content": summary},
    ]


async def summarize(engine, messages: list[dict]) -> str:
    """Ask the engine to summarize ``messages`` into a standalone brief. Returns the summary text
    (empty string if the model returns nothing)."""
    data = await engine.chat(summary_request(messages))
    choice = (data.get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""
