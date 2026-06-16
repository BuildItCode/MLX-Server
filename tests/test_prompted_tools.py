from mlx_launcher.chat import fs_tools
from mlx_launcher.chat import prompted_tools as pt


def test_parse_single_tool_call_and_strip():
    text = 'Sure.\n<tool_call>{"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}</tool_call>'
    assert pt.parse_tool_calls(text) == [{"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}]
    assert pt.strip_tool_calls(text) == "Sure."


def test_parse_multiple_calls_and_key_aliases():
    text = ('<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'
            '<tool_call>{"tool":"list_directory","args":{"path":"."}}</tool_call>')
    assert pt.parse_tool_calls(text) == [
        {"name": "read_file", "arguments": {"path": "a"}},
        {"name": "list_directory", "arguments": {"path": "."}},
    ]


def test_parse_arguments_given_as_json_string():
    text = '<tool_call>{"name":"x","arguments":"{\\"k\\":1}"}</tool_call>'
    assert pt.parse_tool_calls(text) == [{"name": "x", "arguments": {"k": 1}}]


def test_parse_none_and_fenced_fallback():
    assert pt.parse_tool_calls("just a normal answer, no tools") == []
    fenced = '```json\n{"name":"read_file","arguments":{"path":"a"}}\n```'
    assert pt.parse_tool_calls(fenced) == [{"name": "read_file", "arguments": {"path": "a"}}]
    # malformed JSON inside a tag is ignored, not crashed on
    assert pt.parse_tool_calls("<tool_call>{not json}</tool_call>") == []


def test_tool_instructions_describe_the_protocol_and_tools():
    instr = pt.tool_instructions(fs_tools.fs_specs())
    assert "<tool_call>" in instr and "<tool_response>" in instr  # protocol explained
    assert "write_file(path, content)" in instr  # required args, no '?'
    assert "list_directory(path?)" in instr  # optional arg marked '?'
    assert "read_file" in instr and "run_command" in instr


def test_tool_response_format():
    assert pt.tool_response("read_file", "hello") == '<tool_response name="read_file">\nhello\n</tool_response>'


def test_parse_minimax_xml_tool_call():
    # MiniMax-M2 emits its own XML format (not the Hermes <tool_call> JSON we instruct) — we must
    # still recognize it, or the call is shown as text and "nothing happens".
    text = (
        "Sure, let me check.\n"
        "<minimax:tool_call>\n"
        '<invoke name="read_file">\n'
        '<parameter name="path">src/app.py</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )
    assert pt.parse_tool_calls(text) == [{"name": "read_file", "arguments": {"path": "src/app.py"}}]
    assert pt.strip_tool_calls(text) == "Sure, let me check."


def test_parse_minimax_multiple_invokes_and_value_coercion():
    text = (
        "<minimax:tool_call>\n"
        '<invoke name="search_web"><parameter name="query_tag">["tech", "events"]</parameter></invoke>\n'
        '<invoke name="write_file"><parameter name="path">a.py</parameter>'
        '<parameter name="content">x = 1</parameter></invoke>\n'
        "</minimax:tool_call>"
    )
    assert pt.parse_tool_calls(text) == [
        {"name": "search_web", "arguments": {"query_tag": ["tech", "events"]}},  # JSON array parsed
        {"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1"}},  # bare string kept
    ]


def test_minimax_xml_does_not_misread_prose_and_survives_dropped_wrapper():
    # prose that merely mentions invoking a tool is not a call
    assert pt.parse_tool_calls("I'll invoke the search function for you.") == []
    # the <tool_call> JSON form still wins when both could appear; XML is only the fallback
    both = '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'
    assert pt.parse_tool_calls(both) == [{"name": "read_file", "arguments": {"path": "a"}}]
    # a bare <invoke> block whose wrapper was dropped still parses (some servers drop it)
    bare = '<invoke name="list_directory"><parameter name="path">.</parameter></invoke>'
    assert pt.parse_tool_calls(bare) == [{"name": "list_directory", "arguments": {"path": "."}}]
