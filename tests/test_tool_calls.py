"""The consolidated inbound protocol layer: one extract_tool_calls() that reads a tool call out of
a model reply regardless of how it was expressed (native / Harmony / MiniMax / Hermes / loose)."""

from mlx_launcher.chat.tool_calls import extract_tool_calls

NAMES = ["read_file", "list_directory", "web_search"]


def test_native_structured_tool_calls():
    msg = {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]}
    ext = extract_tool_calls(msg, "tool_calls", NAMES)
    assert ext.is_native is True
    assert ext.calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]
    assert ext.native[0]["id"] == "c1"  # raw kept so results can echo back as the `tool` role


def test_hermes_text_tool_call():
    msg = {"content": 'Sure.\n<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'}
    ext = extract_tool_calls(msg, None, NAMES)
    assert ext.is_native is False
    assert ext.calls == [{"name": "list_directory", "arguments": {"path": "."}}]


def test_minimax_xml_tool_call():
    msg = {"content": ('<minimax:tool_call>\n<invoke name="read_file">'
                       '<parameter name="path">a.py</parameter></invoke>\n</minimax:tool_call>')}
    ext = extract_tool_calls(msg, None, NAMES)
    assert ext.is_native is False
    assert ext.calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]


def test_gpt_oss_harmony_commentary_call():
    raw = '<|channel|>commentary to=functions.web_search<|message|>{"query": "x"}<|call|>'
    ext = extract_tool_calls({"content": raw}, None, NAMES)
    assert ext.is_native is False
    assert ext.calls == [{"name": "web_search", "arguments": {"query": "x"}}]


def test_loose_and_drifted_json_recovery():
    # a model that narrates then emits a bare `name {json}` call
    loose = extract_tool_calls({"content": 'I will read it.\nread_file{"path": "a.py"}'}, None, NAMES)
    assert loose.calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]
    # the "tool call became text" drift: a bare {"name":…} object in a loose wrapper
    drift = extract_tool_calls(
        {"content": '[calling tool: {"name": "list_directory", "arguments": {"path": "src"}}]'}, None, NAMES)
    assert drift.calls == [{"name": "list_directory", "arguments": {"path": "src"}}]


def test_plain_answer_has_no_calls_and_propagates_finish():
    ext = extract_tool_calls({"content": "Here is the answer."}, "length", NAMES)
    assert ext.calls == []
    assert ext.content == "Here is the answer."
    assert ext.finish == "length"  # truncation signal passes through for the loop to continue
