"""Unit tests for the deepagents-powered agent loop (omnicode.core.agent.AgentRunner).

A scriptable fake Engine (satisfies engine.base.Engine: chat + stream_chat) drives the loop
offline. Covers: the streaming no-tools path, the native tool loop, text-protocol recovery,
the permission-deny gate, the native→prompted downgrade, and fatal-error surfacing.

The agent loop is now powered by deepagents (LangChain/LangGraph). Tests verify the same
event-stream contract (RunStarted → ContentDelta / ToolStarted / ToolFinished → TurnFinished)
so service.py and acp/agent.py work unchanged.
"""

import json

from omnicode.core import events as ev
from omnicode.core.agent import AgentRunner, RunPolicy, ToolOutcome, ToolSet


def _fn_spec(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}


def _native_call(name, args, finish="tool_calls", call_id="c1"):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        "finish_reason": finish}]}


def _final(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


class FakeEngine:
    """Scriptable Engine: `chat` pops queued responses; `stream_chat` replays a (kind, chunk) script.
    `fail_native_once` raises the first time a native `tools` param is sent (a template rejection)."""

    def __init__(self, chat_responses=None, stream_script=None, fail_native_once=False):
        self.chat_responses = list(chat_responses or [])
        self.stream_script = list(stream_script or [])
        self.fail_native_once = fail_native_once
        self.calls = []  # one {"tools": ...} per chat() call

    async def chat(self, messages, tools=None, *, read_timeout=600.0):
        self.calls.append({"tools": tools, "messages": [dict(m) for m in messages]})
        if self.fail_native_once and tools is not None:
            self.fail_native_once = False
            raise RuntimeError("server returned HTTP 400: this model has no tool parser")
        return self.chat_responses.pop(0)

    async def stream_chat(self, messages, *, cancel=None):
        for item in self.stream_script:
            yield item


async def _collect(runner, messages):
    return [e async for e in runner.run(messages)]


def _kinds(events):
    return [type(e).__name__ for e in events]


def _finish(events):
    return next(e for e in events if isinstance(e, ev.TurnFinished))


async def test_streaming_path_no_tools():
    """No tools → stream one turn live via engine.stream_chat (not deepagents' non-streaming path)."""
    eng = FakeEngine(stream_script=[("reason", "hmm "), ("content", "hello "), ("content", "world"),
                                    ("finish", "stop")])
    out = await _collect(AgentRunner(eng), [{"role": "user", "content": "hi"}])
    assert isinstance(out[0], ev.RunStarted)
    assert "".join(e.text for e in out if isinstance(e, ev.ContentDelta)) == "hello world"
    assert "".join(e.text for e in out if isinstance(e, ev.ReasonDelta)) == "hmm "
    fin = _finish(out)
    assert fin.text == "hello world" and fin.reasoning == "hmm " and fin.n_tool_calls == 0


async def test_native_tool_loop_executes_and_answers():
    """Tools offered → the deepagents tool loop: model calls a tool, gets the result, answers."""
    eng = FakeEngine(chat_responses=[_native_call("web_search", {"query": "vllm"}), _final("the answer")])
    ran = []
    async def execute(name, args):
        ran.append((name, args))
        return ToolOutcome("search results", ok=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    assert ("web_search", {"query": "vllm"}) in ran
    assert any(isinstance(e, ev.ToolStarted) and e.name == "web_search" for e in out)
    assert any(isinstance(e, ev.ToolFinished) and e.status == "ok" for e in out)
    fin = _finish(out)
    assert fin.text == "the answer" and fin.n_tool_calls >= 1


async def test_permission_deny_skips_execution():
    """A denied mutating tool is never executed; the model gets a denial message instead."""
    eng = FakeEngine(chat_responses=[_native_call("write_file", {"path": "a.py", "content": "x"}),
                                     _final("ok, asked")])
    ran = []
    async def execute(name, args):
        ran.append(name)
        return ToolOutcome("wrote")
    async def deny(name, args):
        return "deny"
    tools = ToolSet(specs=[_fn_spec("write_file")], execute=execute, mutating=frozenset({"write_file"}))
    out = await _collect(AgentRunner(eng, tools=tools, permission=deny), [{"role": "user", "content": "q"}])
    assert ran == []  # denied → never executed
    assert any(isinstance(e, ev.ToolFinished) and e.status == "denied" for e in out)
    assert _finish(out).text == "ok, asked"


async def test_permission_all_approves_rest_of_run():
    """'all' auto-approves the rest of the run (no further permission prompts)."""
    eng = FakeEngine(chat_responses=[
        _native_call("write_file", {"path": "a.py", "content": "x"}, call_id="c1"),
        _native_call("write_file", {"path": "b.py", "content": "y"}, call_id="c2"),
        _final("done"),
    ])
    prompts = []
    async def permission(name, args):
        prompts.append(name)
        return "all" if len(prompts) == 1 else "deny"  # only first prompt; rest auto-approved
    ran = []
    async def execute(name, args):
        ran.append(name)
        return ToolOutcome("ok")
    tools = ToolSet(specs=[_fn_spec("write_file")], execute=execute, mutating=frozenset({"write_file"}))
    out = await _collect(AgentRunner(eng, tools=tools, permission=permission), [{"role": "user", "content": "q"}])
    assert len(prompts) == 1  # only prompted once
    assert len(ran) == 2      # both writes executed (auto-approved after "all")


async def test_fatal_engine_error_surfaces_without_retry():
    """An OOM/reshape error is surfaced as TurnFailed(fatal=True), not retried in prompted mode."""
    class FatalEngine:
        async def chat(self, messages, tools=None, *, read_timeout=600.0):
            raise RuntimeError("Metal: out of memory while reshaping")
        async def stream_chat(self, messages, *, cancel=None):
            yield ("finish", "stop")

    async def execute(name, args):
        return ToolOutcome("")

    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(FatalEngine(), tools=tools), [{"role": "user", "content": "q"}])
    failed = [e for e in out if isinstance(e, ev.TurnFailed)]
    assert failed and failed[0].fatal is True


async def test_native_to_prompted_fallback_recovers():
    """A server that rejects native tools (4xx) falls back to prompted mode and still answers.
    The EngineChatModel catches the error internally and retries without the `tools` param."""
    eng = FakeEngine(chat_responses=[_final("answer after fallback")], fail_native_once=True)
    ran = []
    async def execute(name, args):
        ran.append(name)
        return ToolOutcome("results")
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    runner = AgentRunner(eng, tools=tools)
    out = await _collect(runner, [{"role": "user", "content": "q"}])
    assert _finish(out).text == "answer after fallback"


async def test_runner_turns_recorded_for_persistence():
    """The runner records the final assistant turn for persistence (service.py appends it)."""
    eng = FakeEngine(chat_responses=[_native_call("web_search", {"query": "x"}), _final("the answer")])
    async def execute(name, args):
        return ToolOutcome("results")
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    runner = AgentRunner(eng, tools=tools)
    await _collect(runner, [{"role": "user", "content": "q"}])
    assert runner.turns
    assert runner.turns[-1]["role"] == "assistant"
    assert runner.turns[-1]["content"] == "the answer"


async def test_cancel_stops_the_run():
    """A cancel flag stops the run gracefully."""
    eng = FakeEngine(stream_script=[("content", "hello"), ("finish", "stop")])
    cancelled = [False]
    def cancel():
        cancelled[0] = True
        return cancelled[0]
    runner = AgentRunner(eng, cancel=cancel)
    out = await _collect(runner, [{"role": "user", "content": "hi"}])
    fin = _finish(out)
    assert fin.reason == "cancelled"


async def test_tool_result_fed_back_to_model():
    """The tool's result text is fed back to the model (it appears in the next chat() call's messages)."""
    eng = FakeEngine(chat_responses=[_native_call("web_search", {"query": "x"}), _final("based on results")])
    async def execute(name, args):
        return ToolOutcome("THE_SEARCH_RESULT", ok=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    # The second chat() call (the final answer) should contain the tool result in its messages
    assert len(eng.calls) >= 2
    second_messages = eng.calls[1]["messages"]
    all_text = json.dumps(second_messages)
    assert "THE_SEARCH_RESULT" in all_text


# --- "stops after every step" fix: text-protocol history until native is proven ---

def _text_call(name, args, content=""):
    """A reply whose tool call arrives as TEXT (<tool_call> tags), not structured tool_calls —
    what local servers without a tool parser return."""
    return {"choices": [{"message": {
        "content": content + '<tool_call>{"name": "%s", "arguments": %s}</tool_call>'
        % (name, json.dumps(args))}, "finish_reason": "stop"}]}


async def test_prompted_history_uses_text_protocol_until_native_proven():
    """Reported failure: with a local server that has no tool parser, the model does ONE step
    and stops — the follow-up arrives as LangChain's structured history (assistant tool_calls
    array + role:'tool' result), which a text-protocol-trained model reads as 'turn over'.
    Until the server PROVES native support (returns structured tool_calls itself), in-turn
    history must be presented in the text dialect the tool instructions promise:
    <tool_call> tags on the assistant turn, results as user-role <tool_response> blocks."""
    eng = FakeEngine(chat_responses=[
        _text_call("web_search", {"query": "x"}),
        _final("the answer"),
    ])
    async def execute(name, args):
        return ToolOutcome("THE_RESULT", ok=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    assert _finish(out).text == "the answer"
    assert len(eng.calls) >= 2
    second = eng.calls[1]["messages"]
    # No structured tool history leaks through while unproven
    assert not any(m.get("role") == "tool" for m in second)
    assert not any(m.get("tool_calls") for m in second)
    # The call and its result are present in the TEXT protocol instead
    assistant = next(m for m in second if m.get("role") == "assistant")
    assert '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>' in assistant["content"]
    results = [m for m in second if m.get("role") == "user" and "<tool_response" in (m.get("content") or "")]
    assert results and "THE_RESULT" in results[-1]["content"]


async def test_native_history_stays_structured_once_native_proven():
    """A server that DID return structured tool_calls gets native-shaped history from the
    second model call on — the text rewrite is only a bridge for unproven servers."""
    eng = FakeEngine(chat_responses=[
        _native_call("web_search", {"query": "x"}),
        _final("the answer"),
    ])
    async def execute(name, args):
        return ToolOutcome("THE_RESULT", ok=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    assert len(eng.calls) >= 2
    second = eng.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "THE_RESULT" in (m.get("content") or "") for m in second)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)


async def test_parallel_tool_results_fold_into_one_user_turn():
    """Two tool results in a row must not become two adjacent user turns (strict
    role-alternating templates 500 on that) — they fold into one."""
    from omnicode.core.deepagents.model import _text_protocol_history

    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
            {"id": "c2", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}]},
        {"role": "tool", "name": "read_file", "content": "AAA", "tool_call_id": "c1"},
        {"role": "tool", "name": "read_file", "content": "BBB", "tool_call_id": "c2"},
    ]
    out = _text_protocol_history(msgs)
    assert len(out) == 2
    assert out[0]["role"] == "assistant" and "tool_calls" not in out[0]
    assert out[1]["role"] == "user"
    assert "AAA" in out[1]["content"] and "BBB" in out[1]["content"]
