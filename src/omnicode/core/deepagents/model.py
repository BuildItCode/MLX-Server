"""Adapts omnicode's OpenAI-compatible :class:`~omnicode.engine.base.Engine` as a
LangChain :class:`~langchain_core.language_models.BaseChatModel`.

This is the bridge that lets ``deepagents`` (built on LangChain/LangGraph) talk to
any of omnicode's local model servers (mlx-lm, mlx-vlm, vllm-mlx, llama.cpp) through the
same HTTP path the old loop used. All the format-recovery logic (Harmony channel
parsing, text-dialect tool-call extraction, the native→prompted fallback) that
``engine/extract.py`` and ``engine/streaming.py`` provide is applied here, so
deepagents gets clean ``AIMessage`` objects regardless of how broken the local
server's tool support is.

Design:

* ``_generate`` / ``_agenerate`` convert LangChain messages → OpenAI dicts, call
  ``engine.chat()`` (non-streaming, used by the tool loop), then convert the
  response back to a LangChain ``AIMessage`` with proper ``tool_calls``.
* ``_stream`` / ``_astream`` convert → call ``engine.stream_chat()`` → yield
  ``ChatGenerationChunk`` objects, splitting reasoning from content using the
  existing ThinkSplitter / HarmonyParser.
* ``bind_tools`` stores the tool specs (same OpenAI function-tool dicts the old
  loop built) and a native-vs-prompted flag. When the server rejects native tools
  (4xx), the model falls back to the prompted protocol (tools described in the
  system prompt, calls extracted from text) — the same behaviour the old loop did.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, ClassVar, Iterator, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.tool import (
    InvalidToolCall,
    ToolCall,
    tool_call as create_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ...engine import prompted
from ...engine.extract import extract_tool_calls
from ...engine.streaming import parse_harmony, recover_stripped_harmony

# Error markers that mean another (prompted) retry can't fix it — the old loop surfaced
# these instead of paying for a second long call.
_FATAL_MARKERS = (
    "out of memory", "metal", "reshape", "shape", "exceeds", "context length",
    "maximum context", "n_ctx", "kv cache", "cannot allocate", "too many tokens",
)


def is_fatal_generation_error(exc: Exception) -> bool:
    """True for an engine error that another (prompted) retry can't fix."""
    msg = str(exc).lower()
    return any(m in msg for m in _FATAL_MARKERS)


# --- message conversion --------------------------------------------------

def _tool_call_json_or_empty(raw) -> dict:
    """A native tool_calls entry's `arguments` (a JSON string, or already a dict) → dict."""
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _text_protocol_history(messages: list[dict]) -> list[dict]:
    """Rewrite structured tool-call history into the TEXT protocol the prompted tool
    instructions promise.

    LangGraph's in-turn message list is always structured: assistant turns carry
    ``tool_calls`` arrays and results come back as ``role: "tool"`` turns. But the
    prompted protocol we teach the model says: emit ``<tool_call>{…}</tool_call>``
    tags in the assistant text, and each result returns as a ``<tool_response>``
    message — and ``_message_to_openai`` (persistence) replays history exactly that
    way. A local server whose template has no ``tool`` role (or mis-renders one)
    sees a turn shape it was never taught, gets confused about what just happened,
    and answers as if the turn were over — the "stops after every step" symptom.

    So, until the server has proven native tool support (it returned a structured
    ``tool_calls`` array itself — ``native_ok``), the history is presented in the
    text dialect the instructions describe; once native is confirmed, the structured
    form is passed through untouched (the server's own template handles it best).
    """
    from ...engine.prompted import tool_response as _tool_response
    from ...engine.streaming import _render_tool_calls

    out: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = [
                {"name": (fn := (tc.get("function") or {})).get("name", ""),
                 "arguments": _tool_call_json_or_empty(fn.get("arguments"))}
                for tc in m.get("tool_calls") or []
            ]
            tags = _render_tool_calls(calls)
            content = m.get("content") or ""
            m = {k: v for k, v in m.items() if k != "tool_calls"}
            m["content"] = f"{content}\n{tags}".strip() if content.strip() else tags
            out.append(m)
        elif m.get("role") == "tool":
            block = _tool_response(m.get("name") or "tool", m.get("content") or "")
            # Consecutive tool results (parallel calls) fold into ONE user turn — chat
            # templates that require role alternation reject user,user adjacency.
            if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), str):
                out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" + block}
            else:
                out.append({"role": "user", "content": block})
        else:
            out.append(m)
    return out


def lc_to_openai(messages: Sequence[BaseMessage]) -> list[dict]:
    """LangChain messages → OpenAI message dicts (for ``engine.chat`` / ``stream_chat``).

    SystemMessage → system, HumanMessage → user, AIMessage → assistant (with tool_calls),
    ToolMessage → tool. Reuses LangChain's own converter so we match its dialect exactly.
    """
    from langchain_core.messages import convert_to_openai_messages

    return convert_to_openai_messages(list(messages))


def _lc_usage(raw: Optional[dict]) -> dict:
    """OpenAI ``usage`` dict → LangChain's standard ``usage_metadata`` shape."""
    u = raw or {}
    return {
        "input_tokens": u.get("prompt_tokens", 0),
        "output_tokens": u.get("completion_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def _finish_to_langchain(
    raw_content: str,
    raw_reasoning: str,
    tool_calls: list[dict],
    native_calls: list[dict],
    finish_reason: Optional[str],
    tool_names: list[str],
) -> AIMessage:
    """Build a LangChain ``AIMessage`` from one extracted model reply.

    ``native_calls`` are the raw OpenAI ``tool_calls`` entries (when the server returned
    them structured). ``tool_calls`` is the normalized list (name + arguments dict) for
    all formats. When the server returned structured calls we use those; otherwise we
    build ToolCall objects from the text-recovered calls.
    """
    if native_calls:
        tc_list: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []
        for nc in native_calls:
            fn = nc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args)  # validate
                tc_list.append(create_tool_call(name=name, args=parsed_args, id=nc.get("id") or ""))
            except (ValueError, TypeError):
                invalid.append(InvalidToolCall(
                    name=name, args=raw_args, id=nc.get("id") or "",
                    error="malformed arguments",
                ))
        return AIMessage(
            content=raw_content,
            tool_calls=tc_list,
            invalid_tool_calls=invalid,
            additional_kwargs={"reasoning": raw_reasoning} if raw_reasoning else {},
        )
    # Text-recovered calls → synthesize ToolCall objects so deepagents executes them.
    if tool_calls:
        tc_list = []
        for i, call in enumerate(tool_calls):
            tc_list.append(create_tool_call(
                name=call["name"], args=call["arguments"], id=f"call_{i}",
            ))
        # The prompted protocol: clean the tool-call markup from the visible content.
        clean = prompted.strip_tool_calls(raw_content) or raw_reasoning
        return AIMessage(
            content=clean,
            tool_calls=tc_list,
            additional_kwargs={"reasoning": raw_reasoning} if raw_reasoning else {},
        )
    return AIMessage(
        content=raw_content,
        additional_kwargs={"reasoning": raw_reasoning} if raw_reasoning else {},
    )


class EngineChatModel(BaseChatModel):
    """A LangChain chat model backed by omnicode's OpenAI-compatible Engine.

    Not a ``BaseChatOpenAI`` — it speaks raw HTTP via the existing Engine protocol, so
    there's no extra ``openai`` dependency, and the format-recovery layer
    (Harmony/Hermes/text tool-call parsing) runs inside ``_generate``. The native→prompted
    fallback that the old loop did is preserved: if the server rejects the ``tools`` param,
    the model retries with tools described in the system prompt and calls extracted from text.
    """

    engine: Any  # omnicode.engine.base.Engine
    tool_specs: list[dict] = []          # the OpenAI function-tool dicts (set by bind_tools)
    tool_names: list[str] = []
    use_native_tools: bool = True         # start native; downgrade on 4xx
    model_name: str = ""

    # Servers that have RETURNED structured tool_calls (key: (base_url, model)) — their
    # template has a real `tool` role, so history can stay in the native shape. A CLASS
    # attribute because bind_tools copies the model (deepagents' middleware re-binds per
    # model call): an instance flag would never survive to the next LLM invocation.
    _native_ok_keys: ClassVar[set] = set()

    # Servers that REJECTED the native ``tools`` param (4xx) — skip the futile native
    # attempt on every subsequent call to avoid paying for a failed HTTP round-trip each
    # turn. Same class-level reason as ``_native_ok_keys``.
    _native_fail_keys: ClassVar[set] = set()

    def _native_key(self):
        key = getattr(self.engine, "base_url", None) or id(self.engine)
        return (key, self.model_name or getattr(self.engine, "model", ""))

    def _note_native_ok(self) -> None:
        EngineChatModel._native_ok_keys.add(self._native_key())

    def _native_proven(self) -> bool:
        return self._native_key() in EngineChatModel._native_ok_keys

    def _note_native_fail(self) -> None:
        EngineChatModel._native_fail_keys.add(self._native_key())

    def _native_fail_proven(self) -> bool:
        return self._native_key() in EngineChatModel._native_fail_keys

    # ---- metadata ----

    @property
    def _llm_type(self) -> str:
        return "lis-openai-engine"

    @property
    def lc_secrets(self) -> dict:
        return {}

    def _identify_params(self) -> dict:
        return {"model_name": self.model_name}

    # ---- tool binding ----

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "EngineChatModel":
        """Return a copy with tool specs attached.

        ``tools`` may be BaseTool objects, callables, or OpenAI tool dicts. We extract
        the OpenAI function specs (the same dicts the old loop built in
        ``core/tools/*.py``) and store them. ``_generate`` / ``_agenerate`` then either
        pass them as the native ``tools`` param or embed them in the system prompt
        (prompted fallback), depending on ``use_native_tools``.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        specs: list[dict] = []
        for t in tools:
            if isinstance(t, dict) and "type" in t:
                specs.append(t)  # already an OpenAI tool dict
            else:
                specs.append(convert_to_openai_tool(t))
        names = [(s.get("function") or {}).get("name", "") for s in specs]
        return self.model_copy(update={
            "tool_specs": specs,
            "tool_names": names,
            "use_native_tools": True,
        })

    # ---- non-streaming generation ----

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio

        return asyncio.run(self._agenerate(messages, stop, None, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_msgs = self._build_messages(lc_to_openai(messages))
        data = await self._call_with_fallback(openai_msgs)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        ext = extract_tool_calls(msg, choice.get("finish_reason"), self.tool_names)
        if ext.native:
            # The server parsed tool calls itself → its template has a real `tool` role;
            # feed history back in the native shape from here on (see _build_messages).
            self._note_native_ok()
        ai = _finish_to_langchain(
            ext.content, ext.reason, ext.calls, ext.native, ext.finish, self.tool_names,
        )
        lc_usage = _lc_usage(data.get("usage"))
        generation = ChatGeneration(message=ai)
        if lc_usage.get("total_tokens") or lc_usage.get("input_tokens"):
            # Standard LangChain slot — lets deepagents (and our on_chat_model_end hook in
            # adapter.run_turn) read real token counts off the AIMessage directly.
            ai.usage_metadata = lc_usage
        generation.generation_info = {
            "finish_reason": ext.finish,
            "usage": lc_usage,
        }
        return ChatResult(generations=[generation], llm_output={"usage": lc_usage})

    async def _call_with_fallback(self, messages: list[dict]) -> dict:
        """One non-streaming completion with native→prompted fallback.

        Native tools → server returns structured ``tool_calls`` (best case).
        If the server rejects the ``tools`` param (4xx), retry WITHOUT tools:
        tools are described in the system prompt, calls extracted from text.
        A fatal error (OOM / context overflow) propagates — retrying can't fix it.

        Once a server has rejected native tools, ``_native_fail_proven`` skips the
        futile native attempt on every subsequent call (avoiding a wasted HTTP
        round-trip per turn). The tool instructions are ALREADY in ``messages``
        (prepended by ``_build_messages``), so the retry sends them as-is — no
        double injection.
        """
        native = self.use_native_tools and bool(self.tool_specs) and not self._native_fail_proven()
        try:
            return await self.engine.chat(messages, tools=self.tool_specs if native else None)
        except Exception as exc:
            if native and self.tool_specs and not is_fatal_generation_error(exc):
                self._note_native_fail()  # remember so we don't retry native next time
                return await self.engine.chat(messages, tools=None)
            raise

    # ---- streaming ----

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import asyncio

        # LangChain's generate_from_stream handles sync iteration of an async generator
        # by running it in an event loop — but _stream is expected to be synchronous.
        # Since our engine is async-only, we bridge via run_until_complete.
        loop = asyncio.new_event_loop()
        try:
            agen = self._astream_chunks(lc_to_openai(messages))
            while True:
                try:
                    kind, text = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                if kind == "usage":
                    continue  # token counts, not content — handled by the async path
                if text:
                    if kind == "reason":
                        chunk = ChatGenerationChunk(message=AIMessageChunk(
                            content="", additional_kwargs={"reasoning_content": text}))
                    else:
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(text)
        finally:
            loop.close()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # When tools are bound, the engine's stream_chat can't pass the `tools` param, so
        # we use the non-streaming engine.chat() (with native→prompted fallback) and yield
        # the result as chunks. This matches the old loop's behaviour: the tool loop was
        # always non-streaming (engine.chat), only plain chat was streamed (stream_chat).
        if self.tool_specs:
            async for chunk in self._astream_with_tools(messages):
                yield chunk
                if run_manager and chunk.message.content:
                    await run_manager.on_llm_new_token(chunk.message.content)
            return
        # No tools → live streaming for a responsive TUI.
        usage: Optional[dict] = None
        async for kind, text in self._astream_chunks(lc_to_openai(messages)):
            if kind == "usage":
                try:
                    usage = json.loads(text)
                except (ValueError, TypeError):
                    pass
                continue
            if not text:
                continue
            if kind == "reason":
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": text}))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                await run_manager.on_llm_new_token(text)
        lc_usage = _lc_usage(usage)
        if lc_usage.get("total_tokens") or lc_usage.get("input_tokens"):
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=lc_usage))

    async def _astream_with_tools(
        self, messages: list[BaseMessage],
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Non-streaming generation (with tools) yielded as a single chunk pair.

        The engine's ``chat()`` returns the full response at once; we yield the content
        and reasoning as chunks, plus the tool_calls as part of the AIMessageChunk. This
        lets deepagents' streaming event loop observe the model's tool-call decision.
        """
        openai_msgs = self._build_messages(lc_to_openai(messages))
        data = await self._call_with_fallback(openai_msgs)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        ext = extract_tool_calls(msg, choice.get("finish_reason"), self.tool_names)
        if ext.native:
            self._note_native_ok()  # see _agenerate / _build_messages

        lc_usage = _lc_usage(data.get("usage"))
        # Final chunk carries the real token counts — run_turn's on_chat_model_end hook
        # reads usage_metadata off the merged AIMessage for the context bar.
        if lc_usage.get("total_tokens") or lc_usage.get("input_tokens"):
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=lc_usage))

        # Yield reasoning first (if any), then content.
        if ext.reason:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": ext.reason}))

        if ext.calls:
            # Tool calls: yield content (stripped of markup) + tool_call chunks
            clean = prompted.strip_tool_calls(ext.content) if ext.content else ""
            if clean:
                yield ChatGenerationChunk(message=AIMessageChunk(content=clean))
            for i, call in enumerate(ext.calls):
                # Build a tool_call chunk so deepagents' tool node picks it up.
                # args must be a JSON string (LangChain re-parses it into tool_calls).
                call_id = (ext.native[i].get("id") if i < len(ext.native) else None) or f"call_{i}"
                tc_chunk = {
                    "name": call["name"],
                    "args": json.dumps(call["arguments"]),
                    "id": call_id,
                    "index": i,
                }
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="", tool_call_chunks=[tc_chunk]))
        else:
            # Plain answer
            if ext.content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=ext.content))

    async def _astream_chunks(
        self, openai_msgs: list[dict],
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield ``('content'|'reason', text)`` chunks from the streaming engine."""
        msgs = self._build_messages(openai_msgs)
        async for kind, chunk in self.engine.stream_chat(msgs):
            if kind in ("content", "reason", "usage"):
                yield kind, chunk
            elif kind == "finish":
                return

    # ---- helpers ----

    def _build_messages(self, openai_msgs: list[dict]) -> list[dict]:
        """Inject the text-protocol tool instructions whenever tools are bound.

        ALWAYS injected (not only in prompted mode): many local servers silently
        IGNORE the native ``tools`` param without returning a 4xx, so the model only
        ever learns how to call tools from the prompt. When native calls do come back,
        ``extract_tool_calls`` prefers them, so this costs nothing — and when the
        server drops the param, the model still emits ``<tool_call>`` tags that the
        extractor recovers. This mirrors the old hand-rolled loop, which prepended the
        same instructions on every tool-enabled turn.

        Matching dialect for history: until the server has PROVEN native support (it
        returned a structured ``tool_calls`` array — ``_native_proven()``), in-turn tool
        history (assistant tool_calls + role:"tool" results) is rewritten into the
        text protocol the instructions promise (``<tool_call>`` tags + user-role
        ``<tool_response>`` blocks). Local servers whose template has no real
        ``tool`` role otherwise see a turn shape they were never taught and answer as
        if the turn were finished — the "stops after every step" symptom.
        """
        msgs = [dict(m) for m in openai_msgs]
        if not self.tool_specs:
            return msgs
        if not self._native_proven():
            msgs = _text_protocol_history(msgs)
        from ..messages import prepend_system

        prepend_system(msgs, prompted.tool_instructions(self.tool_specs))
        return msgs
