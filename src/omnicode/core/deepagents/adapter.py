"""The adapter: builds a ``deepagents`` agent from a omnicode session and drives one turn,
translating the LangGraph event stream into omnicode's own :mod:`~omnicode.core.events`.

:func:`build_agent` constructs a :class:`DeepAgent` (a compiled LangGraph StateGraph)
from a :class:`~omnicode.core.session.Session`, wiring:
  * The session's engine as the model (:class:`EngineChatModel`).
  * The session's tools (web/fs/MCP) as LangChain tools.
  * Deepagents' built-in features: subagents, memory, skills, filesystem middleware,
    and the permission + event middleware for omnicode's human-in-the-loop + event stream.
  * The prompted-protocol fallback (via ``EngineChatModel``).

:func:`run_turn` drives one user turn through the agent's ``astream_events``, translating
LangGraph streaming events (``on_chat_model_stream``, ``on_tool_start``, ``on_tool_end``)
into omnicode events (``ContentDelta``, ``ToolStarted``, ``ToolFinished``, ``TurnFinished``),
and yielding them as an async generator — the same contract the old ``AgentRunner.run``
had, so callers (service.py, acp/agent.py) need no changes.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Callable, Optional

from .middleware import PermissionMiddleware, _LISFileToolsMiddleware, _ToolEventCapture
from .model import EngineChatModel
from .tools import toolset_to_langchain_tools, mutating_tool_names
from .. import events as ev
from ..tools.phrasing import _tool_phrase

# NOTE: ToolSet / PermissionPolicy from ..agent are imported lazily inside functions
# to avoid a circular import (agent.py → deepagents.adapter → ..agent).
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import ToolSet, PermissionPolicy


def build_agent(
    engine: Any,
    tools: "ToolSet",
    *,
    system_prompt: Optional[str] = None,
    permission: "PermissionPolicy" = None,
    max_iters: int = 24,
    fs_root: Optional[str] = None,
    open_url: Optional[Callable] = None,
    subagents: Optional[list[dict]] = None,
    memory_sources: Optional[list[str]] = None,
    skill_sources: Optional[list[str]] = None,
) -> tuple[Any, _ToolEventCapture]:
    """Construct a deepagents agent from a omnicode engine + toolset.

    Returns ``(agent, capture)`` where ``agent`` is the compiled DeepAgent (a LangGraph
    CompiledStateGraph) and ``capture`` is the :class:`_ToolEventCapture`.

    The agent is configured with:
      * ``EngineChatModel`` wrapping the omnicode engine (with native→prompted fallback).
      * omnicode tools as LangChain tools (web/fs/MCP with the existing executor closures).
      * Permission gating built into the executor (so all tool events flow through
        LangGraph's on_tool_start/on_tool_end — even denied ones).
      * Deepagents' built-in subagent delegation (``task`` tool), memory, summarization,
        and skills — but WITHOUT its filesystem tools (which would conflict with omnicode's
        sandboxed versions scoped to the working directory).

    ``system_prompt`` is the user/system guidance for this chat (skill + instructions +
    coding/plan mode framing). It's placed BEFORE deepagents' own base prompt so the
    prefix stays stable for KV-cache reuse across turns.

    ``subagents`` is a list of deepagents ``SubAgent`` dicts (name/description/
    system_prompt/model) that the main model can delegate to via the ``task`` tool.

    ``memory_sources`` is a list of file paths (e.g. AGENTS.md) loaded into context.

    ``skill_sources`` is a list of skill directory paths for on-demand skill loading.

    ``fs_root`` is the chat's working directory. When set, a ``LocalShellBackend``
    (scoped to that root, ``virtual_mode=True``) is wired in so deepagents' built-in
    filesystem + ``execute`` tools do the file work (path-confined by the backend), a
    :class:`_LISFileToolsMiddleware` gates mutating file/shell ops behind
    ``permission``, and the colliding fs tool names are DROPPED from the user-tool list
    so the built-ins win.
    """
    from deepagents import create_deep_agent, HarnessProfile, register_harness_profile

    # Register (idempotently) a harness profile that excludes deepagents' built-in
    # todo tools — omnicode doesn't surface the task list. The provider key is the lowercase
    # class name ('enginechatmodel'); deepagents matches on this.
    if not getattr(build_agent, "_profile_registered", False):
        register_harness_profile(
            "enginechatmodel",
            HarnessProfile(
                excluded_tools=frozenset({
                    "write_todos", "read_todos",     # todo list — not needed in omnicode
                }),
            ),
        )
        build_agent._profile_registered = True

    # When a working dir is set, deepagents' FilesystemMiddleware injects its own
    # ls/read_file/write_file/edit_file/glob/grep/execute tools. The name-based
    # `_ToolExclusionMiddleware` strips tools from the WHOLE request by name, so our
    # same-named user tools must be dropped from the list here — otherwise the model
    # would emit the built-in's args (absolute paths, offset/limit) at our executor,
    # which speaks the old relative-path schema.
    fs_names = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
    user_tools = tools
    if fs_root:
        from ..agent import ToolSet as _TS

        kept = [s for s in tools.specs if (s.get("function") or {}).get("name") not in fs_names]
        user_tools = _TS(specs=kept, execute=tools.execute, mutating=tools.mutating)

    # Wrap the (non-fs) user tools' executor with permission gating. This ensures
    # LangGraph's on_tool_start / on_tool_end fire for EVERY tool call (including denied
    # ones), so the omnicode event stream is complete. The fs built-ins are gated separately
    # by _LISFileToolsMiddleware below (they're deepagents' own tools, not ours).
    capture = _ToolEventCapture()
    if user_tools.mutating and permission is not None and user_tools.execute is not None:
        _execute = user_tools.execute
        _mutating = user_tools.mutating

        async def gated_execute(name: str, args: dict):
            from ..agent import ToolOutcome

            if name in _mutating and not capture.auto_approved:
                decision = await permission(name, args)
                if decision == "all":
                    capture.auto_approved = True
                elif decision == "deny":
                    return ToolOutcome(
                        "The user DENIED this action. Do not retry it; ask how to proceed.",
                        ok=False,
                    )
            return await _execute(name, args)

        from ..agent import ToolSet as _TS

        user_tools = _TS(specs=user_tools.specs, execute=gated_execute, mutating=user_tools.mutating)

    lc_tools = toolset_to_langchain_tools(user_tools)
    model = EngineChatModel(engine=engine, tool_specs=[], tool_names=[], model_name=getattr(engine, "model", ""))

    backend = None
    middleware: list = []
    if fs_root:
        import os

        from deepagents.backends.local_shell import LocalShellBackend

        # virtual_mode=True confines every built-in fs op to the working dir (blocks
        # '..', '~', and absolute paths outside root_dir) — the same sandbox our old
        # hand-rolled fs tools enforced. LocalShellBackend also implements
        # SandboxBackendProtocol, so deepagents' `execute` runs a real shell in root.
        backend = LocalShellBackend(root_dir=os.path.expanduser(fs_root), virtual_mode=True)
        middleware.append(_LISFileToolsMiddleware(
            permission=permission, capture=capture, fs_root=os.path.expanduser(fs_root),
        ))

    agent = create_deep_agent(
        model=model,
        tools=lc_tools,
        system_prompt=system_prompt or "",
        backend=backend,
        middleware=middleware,
        subagents=subagents or None,
        memory=memory_sources or None,
        skills=skill_sources or None,
        name="lis-agent",
    )
    return agent, capture


def _openai_to_langchain(messages: list[dict]) -> list:
    """Convert OpenAI message dicts to LangChain message objects.

    Handles system/user/assistant (with tool_calls) and tool roles. This replaces the
    ``convert_openai_messages_to_langchain`` function that older langchain_core versions
    provided — it was removed in langchain-core 1.x.
    """
    import json

    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.messages.tool import tool_call as create_tool_call

    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content or ""))
        elif role == "user":
            out.append(HumanMessage(content=content or ""))
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            tc_list = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (ValueError, TypeError):
                    args = {}
                tc_list.append(create_tool_call(
                    name=fn.get("name", ""), args=args, id=tc.get("id") or "",
                ))
            out.append(AIMessage(content=content or "", tool_calls=tc_list))
        elif role == "tool":
            out.append(ToolMessage(
                content=content or "",
                tool_call_id=m.get("tool_call_id") or "",
                name=m.get("name") or "",
            ))
        else:
            out.append(HumanMessage(content=content or ""))
    return out


async def run_turn(
    agent: Any,
    capture: _ToolEventCapture,
    messages: list[dict],
    *,
    max_iters: int = 24,
    cancel: Optional[Callable[[], bool]] = None,
) -> AsyncIterator:
    """Drive one user turn through the agent, yielding omnicode events.

    ``messages`` is the OpenAI message list (same as the old ``AgentRunner.run``).
    We feed it to ``agent.astream_events`` and translate the LangGraph event stream into
    omnicode events. The contract (yielding :mod:`~omnicode.core.events`) matches the old
    ``AgentRunner.run`` so ``service.py`` and ``acp/agent.py`` need no changes.

    ``max_iters`` sets the LangGraph recursion_limit (each model+tool cycle is 2 steps, so
    ``recursion_limit = max_iters * 2 + 2``). Default 24 → limit 50 (was hard-coded 100).
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    t0 = time.monotonic()
    lc_messages = _openai_to_langchain(messages)
    yield ev.RunStarted()

    content_acc: list[str] = []
    reason_acc: list[str] = []
    finish_reason = "stop"
    n_tool_calls = 0
    peak_input_tokens = 0  # largest prompt_tokens the server reported for any LLM call this turn

    try:
        config = {"recursion_limit": max(2, max_iters * 2 + 2)}
        async for event in agent.astream_events(
            {"messages": lc_messages}, config=config, version="v2",
        ):
            # Check cancel ONLY at the start of a new model call (not between tool events),
            # so a cancel that arrives during a permission prompt still lets the tool's
            # denial event flow through (the old loop checked cancel at the top of each
            # iteration, not between every sub-event).
            kind = event.get("event", "")
            if cancel and cancel() and kind == "on_chat_model_start":
                finish_reason = "cancelled"
                break

            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                # A chunk carrying token counts (emitted at the end of each model call —
                # see model.py). Merged into the peak like on_chat_model_end's AIMessage.
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage:
                    peak_input_tokens = max(
                        peak_input_tokens, int(chunk_usage.get("input_tokens") or 0)
                    )
                text = getattr(chunk, "content", "")
                if text:
                    content_acc.append(text)
                    yield ev.ContentDelta(text)
                reasoning = (getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content", "")
                if reasoning:
                    reason_acc.append(reasoning)
                    yield ev.ReasonDelta(reasoning)

            elif kind == "on_tool_start":
                name = event.get("name", "")
                # Extract args from the run input
                run_input = data.get("input", {})
                if isinstance(run_input, dict):
                    args = run_input.get("args", run_input)
                else:
                    args = {}
                tool_id = event.get("run_id", "")
                phrase = _tool_phrase(name, args if isinstance(args, dict) else {})
                yield ev.ToolStarted(tool_id, name, phrase, args if isinstance(args, dict) else {})
                n_tool_calls += 1

            elif kind == "on_chat_model_end":
                # Real token usage from the engine (mlx-lm / llama.cpp / LM Studio all return
                # OpenAI-style `usage`). The LARGEST input_tokens across the turn's LLM calls is
                # the conversation's true context footprint — it already includes the system
                # prompt, tool schemas/instructions, and the deepagents base prompt, which the
                # TUI's char-based estimate never sees.
                output = data.get("output")
                usage = getattr(output, "usage_metadata", None)
                if not usage and isinstance(output, dict):
                    gens = (output.get("generations") or [[]])[0]
                    usage = getattr(gens[0].message, "usage_metadata", None) if gens else None
                if usage:
                    peak_input_tokens = max(
                        peak_input_tokens, int(usage.get("input_tokens") or 0)
                    )

            elif kind == "on_tool_end":
                name = event.get("name", "")
                output = data.get("output")
                if output is not None:
                    text = getattr(output, "content", str(output))
                else:
                    text = ""
                tool_id = event.get("run_id", "")
                # Detect permission denials and errors from the middleware pipeline
                status = "ok"
                if text.startswith("The user DENIED"):
                    status = "denied"
                elif text.startswith("tool error:") or text.startswith("ERROR"):
                    status = "error"
                preview = text if len(text) <= 500 else text[:500] + " …"
                yield ev.ToolFinished(tool_id, name, text, status=status, preview=preview)

            elif kind == "on_custom_event" and event.get("name") == "lis_tool_denied":
                # Emitted by _LISFileToolsMiddleware on a permission DENY: LangChain
                # suppresses the tool node's events for middleware-short-circuited
                # calls, so the middleware dispatches both halves of the pair itself.
                d = data or {}
                text = d.get("text", "")
                preview = text if len(text) <= 500 else text[:500] + " …"
                tid = d.get("id", "")
                tname = d.get("name", "")
                args = d.get("args") or {}
                phrase = _tool_phrase(tname, args if isinstance(args, dict) else {})
                yield ev.ToolStarted(tid, tname, phrase, args if isinstance(args, dict) else {})
                yield ev.ToolFinished(tid, tname, text, status="denied", preview=preview)

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                # Final state — extract the last AI message (usage too, as a last resort
                # when the streaming path didn't surface it).
                output = data.get("output", {})
                if isinstance(output, dict):
                    msgs = output.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        usage = getattr(last, "usage_metadata", None)
                        if usage:
                            peak_input_tokens = max(
                                peak_input_tokens, int(usage.get("input_tokens") or 0)
                            )
                        final_text = getattr(last, "content", "")
                        if final_text and not content_acc:
                            content_acc.append(final_text)
                            yield ev.ContentDelta(final_text)

    except Exception as exc:  # noqa: BLE001
        from .model import is_fatal_generation_error

        yield ev.TurnFailed(str(exc), fatal=is_fatal_generation_error(exc))
        return

    final_text = "".join(content_acc)
    reasoning = "".join(reason_acc)
    if peak_input_tokens:
        yield ev.UsageUpdated(used=peak_input_tokens)
    yield ev.TurnFinished(
        final_text,
        reason=finish_reason,
        n_tool_calls=n_tool_calls,
        elapsed=round(time.monotonic() - t0, 1),
        reasoning=reasoning,
    )
