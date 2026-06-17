"""Unit tests for the unified agent loop (mlx_launcher.core.agent.AgentRunner).

A scriptable fake Engine (satisfies engine.base.Engine: chat + stream_chat) drives the loop
offline. Covers: the streaming no-tools path, the native tool loop, text-protocol recovery,
the permission-deny gate, the native→prompted downgrade, and truncation-continue."""

import json

from mlx_launcher.core import events as ev
from mlx_launcher.core.agent import AgentRunner, RunPolicy, ToolOutcome, ToolSet


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
    eng = FakeEngine(stream_script=[("reason", "hmm "), ("content", "hello "), ("content", "world"),
                                    ("finish", "stop")])
    out = await _collect(AgentRunner(eng), [{"role": "user", "content": "hi"}])
    assert isinstance(out[0], ev.RunStarted)
    assert "".join(e.text for e in out if isinstance(e, ev.ContentDelta)) == "hello world"
    assert "".join(e.text for e in out if isinstance(e, ev.ReasonDelta)) == "hmm "
    fin = _finish(out)
    assert fin.text == "hello world" and fin.reasoning == "hmm " and fin.n_tool_calls == 0


async def test_native_tool_loop_executes_and_answers():
    eng = FakeEngine(chat_responses=[_native_call("web_search", {"query": "vllm"}), _final("the answer")])
    ran = []
    async def execute(name, args):
        ran.append((name, args))
        return ToolOutcome("search results", ok=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    assert ran == [("web_search", {"query": "vllm"})]
    assert any(isinstance(e, ev.ToolStarted) and e.name == "web_search" for e in out)
    assert any(isinstance(e, ev.ToolFinished) and e.status == "ok" for e in out)
    fin = _finish(out)
    assert fin.text == "the answer" and fin.n_tool_calls == 1
    assert eng.calls[0]["tools"] is not None  # native param sent


async def test_text_protocol_recovery_when_prompted():
    # A server with no native tool parser returns the call as <tool_call> text in content.
    call = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'
    eng = FakeEngine(chat_responses=[_final(call), _final("done")])
    ran = []
    async def execute(name, args):
        ran.append(name)
        return ToolOutcome("results")
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools, policy=RunPolicy(native_tools=False)),
                         [{"role": "user", "content": "q"}])
    assert ran == ["web_search"]
    assert eng.calls[0]["tools"] is None  # prompted → native param NOT sent
    assert _finish(out).text == "done"


async def test_permission_deny_skips_execution():
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


async def test_native_to_prompted_downgrade():
    eng = FakeEngine(chat_responses=[_final("answer after downgrade")], fail_native_once=True)
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=lambda n, a: None)
    runner = AgentRunner(eng, tools=tools)
    out = await _collect(runner, [{"role": "user", "content": "q"}])
    assert any(isinstance(e, ev.Notice) and "prompted" in e.text.lower() for e in out)
    assert runner.used_prompted is True
    assert _finish(out).text == "answer after downgrade"


async def test_max_tool_calls_cap_then_wraps_up():
    # a model that keeps searching must stop at the cap, then a no-tools wrap-up forces an answer.
    eng = FakeEngine(chat_responses=[_native_call("web_search", {"query": "a"}),
                                     _native_call("web_search", {"query": "b"}),
                                     _final("final after cap")])

    async def execute(name, args):
        return ToolOutcome("r")

    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools, policy=RunPolicy(max_iters=8, max_tool_calls=2)),
                         [{"role": "user", "content": "q"}])
    fin = _finish(out)
    assert fin.n_tool_calls == 2 and fin.text == "final after cap"
    assert any(isinstance(e, ev.Notice) for e in out)  # "reached the N-call limit"


async def test_fatal_engine_error_is_not_retried_in_prompted_mode():
    # an OOM/reshape error is NOT a tools-template rejection — don't waste a second prompted retry.
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
    assert not any(isinstance(e, ev.Notice) and "prompted" in e.text.lower() for e in out)


async def test_empty_final_channel_falls_back_to_reasoning():
    # gpt-oss sometimes puts its post-tool summary in the ANALYSIS channel, leaving `final` empty.
    # After real tool work that must surface as the answer, not "(no answer)".
    call = '<|channel|>commentary to=functions.web_search<|message|>{"query":"x"}<|call|>'
    summary = "<|channel|>analysis<|message|>Found 2 issues: A and B<|end|>"
    eng = FakeEngine(chat_responses=[_final(call), _final(summary)])

    async def execute(name, args):
        return ToolOutcome("a result")

    tools = ToolSet(specs=[_fn_spec("web_search")], execute=execute)
    out = await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "check"}])
    assert "Found 2 issues" in _finish(out).text


async def test_truncation_continues_across_turns():
    # A 'length' finish with no tool call → push the partial + a continue nudge and resume, rather
    # than treating the truncated turn as done. Each piece is strip()'d (faithful to the original
    # loop's `strip_tool_calls`), so the accumulation concatenates without an inserted separator.
    eng = FakeEngine(chat_responses=[_final("partial.", finish="length"), _final("the rest.")])
    tools = ToolSet(specs=[_fn_spec("web_search")], execute=lambda n, a: None)
    out = await _collect(AgentRunner(eng, tools=tools), [{"role": "user", "content": "q"}])
    assert _finish(out).text == "partial.the rest."  # accumulated across the two turns
