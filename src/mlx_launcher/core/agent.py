"""The unified agent loop — one transport-agnostic async generator that replaces the three
copies that used to live in the frontends (the TUI's ``_generate_tools``, the ACP agent's
``_agentic_prompt``, and the subagent side-pane's ``_subagent_tool_loop``).

``AgentRunner.run(messages)`` yields :mod:`mlx_launcher.core.events` and performs no I/O of its
own beyond the injected ``engine`` (an :class:`mlx_launcher.engine.base.Engine`) and ``tools``. The
two interactive decisions are injected async callbacks so the loop can block on the answer without
knowing the transport:

* ``permission(name, args) -> 'once'|'all'|'deny'`` — asked before each *mutating* tool.
* the tool executor (``ToolSet.execute``) owns running web/fs/MCP tools (and opening URLs), so the
  loop never branches on transport: the TUI/local-server supply a server-side executor, the ACP
  agent supplies one that delegates to the editor.

It is engine-agnostic (any OpenAI-compatible server via the Engine protocol) and frontend-agnostic
(any consumer of the event stream)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from ..engine import prompted
from ..engine.extract import extract_tool_calls
from ..engine.streaming import parse_harmony, recover_stripped_harmony
from . import events as ev
from .instructions import CONTINUE_TRUNCATED_PROMPT, WRAP_UP_PROMPT
from .messages import prepend_system
from .tools.phrasing import _tool_phrase


@dataclass
class ToolOutcome:
    """The result of running one tool: the text fed back to the model, and whether it succeeded.
    ``denied`` marks a permission refusal (the loop sets this; the executor never sees it)."""

    text: str
    ok: bool = True


# (name, args) -> the text to feed back to the model + whether it succeeded.
ToolExecutor = Callable[[str, dict], Awaitable[ToolOutcome]]
# (name, args) -> "once" | "all" | "deny". Only called for mutating tools when not auto-approving.
PermissionPolicy = Callable[[str, dict], Awaitable[str]]


async def _allow_all(name: str, args: dict) -> str:
    return "all"


@dataclass
class ToolSet:
    """The tools available to a run: their OpenAI specs, an async executor, and which names mutate
    state (so the loop can gate them behind a permission prompt)."""

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
    """Loop bounds + behavior flags. The TUI uses 24 iters / uncapped calls for file tools and
    8 / 8 for web+MCP; the ACP agent used 12; the subagent used 8."""

    max_iters: int = 12
    max_tool_calls: Optional[int] = None
    continue_on_truncation: bool = True
    force_final_answer: bool = True
    native_tools: bool = True  # start by sending the native `tools` param (auto-downgrades on 4xx)


# A model-level error that is NOT a tool-template rejection (OOM / reshape / context overflow) —
# retrying in prompted mode just wastes another long call, so the loop surfaces it instead.
_FATAL_GENERATION_MARKERS = (
    "out of memory", "metal", "reshape", "shape", "exceeds", "context length",
    "maximum context", "n_ctx", "kv cache", "cannot allocate", "too many tokens",
)


def is_fatal_generation_error(exc: Exception) -> bool:
    """True for an engine error that another (prompted) retry can't fix — so the loop stops
    instead of paying for a second long call. Mirrors the frontend's old gate."""
    msg = str(exc).lower()
    return any(m in msg for m in _FATAL_GENERATION_MARKERS)


def _tool_call_echo(content: str, reason: str, calls: list[dict]) -> str:
    """The assistant turn to put back into the in-flight history for a *text-recovered* tool call.

    MiniMax's native ``<minimax:tool_call>`` XML is echoed VERBATIM (re-rendering it as Hermes
    JSON makes MiniMax drift); everything else is rebuilt as clean prose + ``<tool_call>`` tags,
    which nudges a drifted/loose form back toward the instructed protocol and avoids gpt-oss
    Harmony's raw ``<|...|>`` control tokens."""
    c = content or ""
    if "<minimax:tool_call>" in c or "<invoke name=" in c:
        return content
    prose = (prompted.strip_tool_calls(content) or reason or "").strip()
    tags = "\n".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)
    return f"{prose}\n{tags}".strip() if prose else tags


class AgentRunner:
    """Drives one user turn to completion, yielding events. Re-entrant: all per-run state is local
    to :meth:`run`, so two runs (e.g. two TUI panes) can proceed concurrently with separate
    runners — nothing is stored on ``self`` that a second run could corrupt."""

    def __init__(
        self,
        engine,
        *,
        tools: Optional[ToolSet] = None,
        policy: Optional[RunPolicy] = None,
        permission: PermissionPolicy = _allow_all,
        system_note: Optional[str] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.engine = engine
        self.tools = tools or ToolSet()
        self.policy = policy or RunPolicy()
        self.permission = permission
        self.system_note = system_note
        self._cancel = cancel or (lambda: False)
        # Set after run(): whether the loop fell back to the prompted protocol (the caller may
        # remember this to start prompted next time for the same server).
        self.used_prompted = not self.policy.native_tools
        # Set after run(): the conversation turns this run produced (assistant tool-call turns, tool
        # results, and the final assistant turn) — what a frontend appends to persist the exchange.
        self.turns: list[dict] = []

    # --- internals -------------------------------------------------------

    async def _turn(self, messages: list[dict], *, native: bool) -> Optional[dict]:
        """One non-streaming completion that aborts promptly on cancel. Returns the OpenAI
        response dict, or None if cancelled. Engine errors propagate."""
        import asyncio

        specs = self.tools.specs if (native and self.tools.specs) else None
        task = asyncio.ensure_future(self.engine.chat(messages, tools=specs))
        while not task.done():
            if self._cancel():
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001
                    pass
                return None
            await asyncio.sleep(0.1)
        return task.result()

    async def _run_calls(self, ext, messages: list[dict]):
        """Echo the model's tool-call turn, execute each call (gating mutating ones behind the
        permission policy), feed results back, and yield tool events. Mutates ``messages``; yields
        events and returns the count of calls executed via a trailing ``('count', n)`` tuple."""
        n = 0
        auto = False  # an "approve all" decision sticks for the rest of this run
        if ext.is_native:
            messages.append({"role": "assistant", "content": ext.content or None, "tool_calls": ext.native})
            feed = lambda call, raw, result: messages.append(
                {"role": "tool", "tool_call_id": raw.get("id", ""), "content": result[:8000]})
            pairs = list(zip(ext.native, ext.calls))
        else:
            messages.append({"role": "assistant", "content": _tool_call_echo(ext.content, ext.reason, ext.calls)})
            feed = lambda call, raw, result: messages.append(
                {"role": "user", "content": prompted.tool_response(call["name"], result[:8000])})
            pairs = [(None, call) for call in ext.calls]

        for raw, call in pairs:
            name, args = call["name"], call["arguments"]
            tool_id = (raw or {}).get("id") or f"call_{n}"
            phrase = _tool_phrase(name, args)
            # permission gate for mutating tools
            if self.tools.is_mutating(name) and not auto:
                decision = await self.permission(name, args)
                if decision == "all":
                    auto = True
                elif decision != "once":
                    # Emit start+end so the tool still renders (every tool_end has a matching
                    # tool_start in the contract — frontends mount on start, update on end).
                    yield ev.ToolStarted(tool_id, name, phrase, args)
                    yield ev.ToolFinished(tool_id, name, "", status="denied",
                                          preview="denied by the user")
                    denial = "The user DENIED this action. Do not retry it; ask how to proceed."
                    feed(call, raw, denial)
                    n += 1
                    continue
            yield ev.ToolStarted(tool_id, name, phrase, args)
            try:
                outcome = await self.tools.execute(name, args) if self.tools.execute else ToolOutcome(
                    f"Unknown tool: {name}", ok=False)
            except Exception as exc:  # noqa: BLE001
                outcome = ToolOutcome(f"tool error: {exc}", ok=False)
            preview = outcome.text if len(outcome.text) <= 500 else outcome.text[:500] + " …"
            yield ev.ToolFinished(tool_id, name, outcome.text, status="ok" if outcome.ok else "error",
                                  preview=preview)
            feed(call, raw, outcome.text)
            n += 1
        yield ("count", n)  # sentinel: not an Event — the caller unpacks it

    # --- public ----------------------------------------------------------

    async def run(self, messages: list[dict]):
        """Drive the turn to completion, yielding :mod:`mlx_launcher.core.events`. ``messages`` is
        the OpenAI message list (see core.messages.build_openai_messages); it is copied, not
        mutated. The final assistant text is on the closing ``TurnFinished`` event."""
        messages = [dict(m) for m in messages]
        if self.system_note:
            prepend_system(messages, self.system_note)
        if self.tools.specs:
            # Describe the tools in the prompt so the model SEES them even when a server silently
            # ignores the native `tools` param; extract_tool_calls accepts native/Harmony/text alike.
            prepend_system(messages, prompted.tool_instructions(self.tools.specs))
        yield ev.RunStarted()

        if not self.tools.specs:
            async for e in self._run_streaming(messages):
                yield e
            return
        async for e in self._run_tool_loop(messages):
            yield e

    async def _run_streaming(self, messages: list[dict]):
        """No tools → stream one turn live, yielding content/reason deltas (the old _stream_into /
        ACP _chat_prompt path)."""
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
        # Some gpt-oss servers strip the Harmony tokens but leak the channel names — recover.
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

    async def _run_tool_loop(self, messages: list[dict]):
        """Tools offered → the function-calling loop (the old _generate_tools path)."""
        t0 = time.monotonic()
        base_len = len(messages)  # turns appended past here are this run's conversation turns
        n_calls = 0
        final_text = ""
        truncating = False
        native = self.policy.native_tools
        names = self.tools.names
        try:
            for _ in range(self.policy.max_iters):
                if self._cancel():
                    break
                if self.policy.max_tool_calls is not None and n_calls >= self.policy.max_tool_calls:
                    yield ev.Notice("warning",
                                    f"Reached the {self.policy.max_tool_calls}-call limit — "
                                    "answering with what I found")
                    break
                try:
                    data = await self._turn(messages, native=native)
                except Exception as exc:  # noqa: BLE001
                    # native tools rejected → switch to prompted and retry (tools are also described
                    # in the prompt); a fatal generation error is surfaced instead.
                    if self.tools.specs and native and not is_fatal_generation_error(exc):
                        native = False
                        self.used_prompted = True
                        yield ev.Notice("warning",
                                        "Native tool-calling failed — using prompted tools for this model.")
                        continue
                    raise
                if data is None:  # cancelled
                    break
                choice = (data.get("choices") or [{}])[0]
                ext = extract_tool_calls(choice.get("message") or {}, choice.get("finish_reason"), names)
                if not ext.calls:
                    clean = prompted.strip_tool_calls(ext.content) or ext.content or ext.reason
                    if self.policy.continue_on_truncation and ext.finish == "length" and (clean or "").strip():
                        messages.append({"role": "assistant", "content": clean})
                        messages.append({"role": "user", "content": CONTINUE_TRUNCATED_PROMPT})
                        final_text += clean
                        truncating = True
                        continue
                    final_text = (final_text + clean) if truncating else clean
                    truncating = False
                    messages.append({"role": "assistant", "content": ext.content})
                    break
                truncating = False
                final_text = ""  # the real answer comes after the tool work
                async for item in self._run_calls(ext, messages):
                    if isinstance(item, tuple) and item[0] == "count":
                        n_calls += item[1]
                    else:
                        yield item
            # ran out of iterations still calling tools but never answered → one more turn with NO
            # tools so the user gets an answer, not "(no answer)".
            if self.policy.force_final_answer and not final_text and not self._cancel() and n_calls > 0:
                messages.append({"role": "user", "content": WRAP_UP_PROMPT})
                data = await self._turn(messages, native=False)
                if data:
                    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    content, reason = parse_harmony(raw)
                    final_text = prompted.strip_tool_calls(content) or content or reason
        except Exception as exc:  # noqa: BLE001
            yield ev.TurnFailed(str(exc), fatal=is_fatal_generation_error(exc))
            return
        # The in-flight turns this run appended (assistant tool-call turns, tool results, final
        # assistant). Internal nudges (continue/wrap-up) are excluded so a frontend can persist this.
        self.turns = [m for m in messages[base_len:]
                      if m.get("content") not in (WRAP_UP_PROMPT, CONTINUE_TRUNCATED_PROMPT)]
        reason = "cancelled" if self._cancel() else "stop"
        yield ev.TurnFinished(final_text, reason=reason, n_tool_calls=n_calls,
                              elapsed=round(time.monotonic() - t0, 1))
