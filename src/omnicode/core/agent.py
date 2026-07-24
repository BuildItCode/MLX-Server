"""The unified agent loop — now powered by ``langchain-ai/deepagents``.

``AgentRunner.run(messages)`` yields :mod:`omnicode.core.events` (same contract as
before) but internally drives a ``deepagents`` agent (a compiled LangGraph StateGraph)
instead of the old hand-rolled loop. The public interface (``AgentRunner``, ``ToolSet``,
``RunPolicy``, ``ToolOutcome``) is unchanged, so ``service.py`` and ``acp/agent.py`` need
no changes.

What deepagents gives us that the old loop didn't have:
  * **Subagents** — the ``task`` tool lets the model delegate to specialist agents running
    in isolated context windows.
  * **Memory** — AGENTS.md / MEMORY.md files loaded into context via the memory middleware.
  * **Context management** — the summarization middleware auto-compacts long conversations
    (replacing the manual ``/compact`` + auto-compaction logic).
  * **Todo list** — the model can track multi-step work.
  * **Filesystem tools** — read/write/edit/glob/grep via the filesystem middleware.

What's preserved from the old loop:
  * The ``Engine`` protocol (any OpenAI-compatible local server).
  * The native→prompted tool-call fallback (weak servers that reject the ``tools`` param).
  * The permission gate (``once`` / ``all`` / ``deny`` for mutating tools).
  * The event stream contract (``RunStarted`` → ``ContentDelta`` / ``ToolStarted`` / … →
    ``TurnFinished``).
  * The ``turns`` attribute (this run's conversation turns for persistence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from . import events as ev
from .deepagents.adapter import build_agent, run_turn


async def _allow_all(name: str, args: dict) -> str:
    return "all"


@dataclass
class ToolOutcome:
    """The result of running one tool: the text fed back to the model, and whether it
    succeeded. ``denied`` marks a permission refusal (the loop sets this; the executor
    never sees it)."""

    text: str
    ok: bool = True


# (name, args) -> the text to feed back to the model + whether it succeeded.
ToolExecutor = Callable[[str, dict], Awaitable[ToolOutcome]]
# (name, args) -> "once" | "all" | "deny". Only called for mutating tools when not auto-approving.
PermissionPolicy = Callable[[str, dict], Awaitable[str]]


@dataclass
class ToolSet:
    """The tools available to a run: their OpenAI specs, an async executor, and which names
    mutate state (so the loop can gate them behind a permission prompt)."""

    specs: list[dict] = field(default_factory=list)
    execute: Optional[ToolExecutor] = None
    mutating: frozenset[str] = frozenset()

    @property
    def names(self) -> list[str]:
        return [(s.get("function") or {}).get("name") for s in self.specs]

    def is_mutating(self, name: str) -> bool:
        return name in self.mutating


@dataclass
class RunPolicy:
    """Loop bounds + behavior flags (preserved for back-compat; deepagents handles
    iteration limits internally via the recursion limit)."""

    max_iters: int = 24
    max_tool_calls: Optional[int] = None
    continue_on_truncation: bool = True
    force_final_answer: bool = True
    native_tools: bool = True  # start native; the EngineChatModel downgrades on 4xx


# Kept for back-compat (imported by service.py and tests).
_FATAL_GENERATION_MARKERS = (
    "out of memory", "metal", "reshape", "shape", "exceeds", "context length",
    "maximum context", "n_ctx", "kv cache", "cannot allocate", "too many tokens",
)


def is_fatal_generation_error(exc: Exception) -> bool:
    """True for an engine error that another (prompted) retry can't fix."""
    msg = str(exc).lower()
    return any(m in msg for m in _FATAL_GENERATION_MARKERS)


class AgentRunner:
    """Drives one user turn to completion, yielding events — now via deepagents.

    The public interface matches the old runner: ``run(messages)`` is an async generator
    of :mod:`~omnicode.core.events`, ``turns`` holds this run's conversation turns
    for persistence, and ``used_prompted`` records whether the prompted protocol was used.
    """

    def __init__(
        self,
        engine,
        *,
        tools: Optional[ToolSet] = None,
        policy: Optional[RunPolicy] = None,
        permission: PermissionPolicy = _allow_all,
        system_note: Optional[str] = None,
        cancel: Optional[Callable[[], bool]] = None,
        subagents: Optional[list[dict]] = None,
        memory_sources: Optional[list[str]] = None,
        skill_sources: Optional[list[str]] = None,
        fs_root: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.tools = tools or ToolSet()
        self.policy = policy or RunPolicy()
        self.permission = permission
        self.system_note = system_note
        self._cancel = cancel or (lambda: False)
        self.subagents = subagents
        self.memory_sources = memory_sources
        self.skill_sources = skill_sources
        self.fs_root = fs_root
        self.used_prompted = not self.policy.native_tools
        self.turns: list[dict] = []

    async def run(self, messages: list[dict]):
        """Drive the turn to completion, yielding events. ``messages`` is the OpenAI
        message list; it is copied, not mutated. The final assistant text is on the
        closing ``TurnFinished`` event."""
        messages = [dict(m) for m in messages]

        # No tools AND no working dir → stream one turn live (the old _run_streaming
        # path). This keeps plain chat responsive (token-by-token). A working dir
        # implies deepagents' built-in fs tools exist even when `tools.specs` is empty,
        # so that path must NOT take the tool-free shortcut.
        if not self.tools.specs and not self.fs_root:
            if self.system_note:
                from .messages import prepend_system

                prepend_system(messages, self.system_note)
            yield ev.RunStarted()
            async for event in self._run_streaming(messages):
                yield event
            return

        # Tools offered → the deepagents agent loop. For KV-cache stability:
        # extract the system message from the messages list and pass it as the
        # system_prompt param to deepagents. This way the full system prompt
        # (our guidance + deepagents' BASE_AGENT_PROMPT) stays in a FIXED position
        # at the front of the context, so the prefix is cacheable across turns.
        # The messages list (conversation history) follows after, with no system turn.
        system_prompt_parts: list[str] = []
        if messages and messages[0].get("role") == "system":
            system_prompt_parts.append(messages[0].get("content", ""))
            messages = messages[1:]  # strip — deepagents injects it via middleware
        if self.system_note:
            system_prompt_parts.append(self.system_note)
        if self.fs_root:
            # Belt-and-braces for path format: the middleware already normalizes/rejects
            # absolute paths server-side, but without this line models KEEP trying them —
            # each attempt costs a wasted tool call and shows a scary ERROR in the log.
            system_prompt_parts.append(
                f"PATH RULE: every file/shell tool path must be RELATIVE to the working "
                f"directory \"{self.fs_root}\" (e.g. \"index.html\", \"src/app.py\"). "
                "NEVER pass absolute paths."
            )
        system_prompt = "\n\n---\n\n".join(p for p in system_prompt_parts if p and p.strip()) or None

        agent, capture = build_agent(
            self.engine,
            self.tools,
            system_prompt=system_prompt,
            permission=self.permission,
            max_iters=self.policy.max_iters,
            fs_root=self.fs_root,
            subagents=self.subagents,
            memory_sources=self.memory_sources,
            skill_sources=self.skill_sources,
        )

        final_text = ""
        async for event in run_turn(agent, capture, messages,
                                    max_iters=self.policy.max_iters,
                                    cancel=self._cancel):
            yield event
            if isinstance(event, ev.TurnFinished):
                final_text = event.text

        # Record the turns this run produced for persistence (the assistant turn).
        if final_text:
            self.turns = [{"role": "assistant", "content": final_text}]

    async def _run_streaming(self, messages: list[dict]):
        """No tools → stream one turn live, yielding content/reason deltas."""
        import time

        from ..engine.streaming import recover_stripped_harmony

        t0 = time.monotonic()
        content_acc: list[str] = []
        reason_acc: list[str] = []
        finish = "stop"
        try:
            async for kind, chunk in self.engine.stream_chat(messages, cancel=self._cancel):
                if kind == "content":
                    content_acc.append(chunk)
                    yield ev.ContentDelta(chunk)
                elif kind == "reason":
                    reason_acc.append(chunk)
                    yield ev.ReasonDelta(chunk)
                elif kind == "finish":
                    finish = chunk
        except Exception as exc:  # noqa: BLE001
            yield ev.TurnFailed(str(exc), fatal=is_fatal_generation_error(exc))
            return
        final = "".join(content_acc)
        recovered = recover_stripped_harmony(final)
        if recovered is not None:
            final, leaked = recovered
            if leaked:
                reason_acc.append(leaked)
        if self._cancel():
            finish = "cancelled"
        if final:
            self.turns = [{"role": "assistant", "content": final}]
        yield ev.TurnFinished(final, reason=finish, n_tool_calls=0,
                              elapsed=round(time.monotonic() - t0, 1), reasoning="".join(reason_acc))
