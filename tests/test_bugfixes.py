"""Regression tests for the whole-codebase bug-review fixes (2026-06-16)."""

import asyncio
import sys

import pytest

from omnicode import bootstrap
from omnicode.chat import capabilities, client
from omnicode.chat.blocks import linkify_urls


# --- HIGH: a >64 KB line must not kill log/install streaming -------------

def test_run_streamed_survives_overlong_line():
    # one 200 KB line (no newline) used to raise ValueError from readline() and abort the
    # install/download; now it's skipped and later output still streams.
    lines: list[str] = []
    code = "import sys; sys.stdout.write('X' * 200_000 + '\\n'); print('AFTER')"
    rc = asyncio.run(bootstrap.run_streamed([sys.executable, "-c", code], lines.append))
    assert rc == 0
    assert any("AFTER" in ln for ln in lines)  # output after the overlong line still arrives


# --- MED: gpt-oss tool-call recovery -------------------------------------

def test_recover_loose_tool_calls_finds_call_after_prose_mention():
    # the model narrates ("I'll use web_search …") before the real call — must still recover it
    out = client.recover_loose_tool_calls(
        'I will use web_search to find it.\nweb_search{"query": "real"}', ["web_search"])
    assert out == [{"name": "web_search", "arguments": {"query": "real"}}]


def test_recover_loose_tool_calls_ignores_pure_prose():
    assert client.recover_loose_tool_calls("just call read_file with a path like /tmp", ["read_file"]) == []


def test_recover_json_tool_calls_finds_drifted_object():
    # the "the tool call became text" drift: a model that abandons its tagged format and emits a
    # bare JSON object in a loose wrapper. Keyed on known tool names so it's safe.
    names = ["list_directory", "read_file"]
    text = 'Let me explore. [calling tool: {"name": "list_directory", "arguments": {"path": "src"}}]'
    assert client.recover_json_tool_calls(text, names) == [
        {"name": "list_directory", "arguments": {"path": "src"}}]
    # an empty arguments object is still a valid call (e.g. list_directory with its default)
    assert client.recover_json_tool_calls('{"name": "list_directory", "arguments": {}}', names) == [
        {"name": "list_directory", "arguments": {}}]


def test_recover_json_tool_calls_is_conservative():
    names = ["list_directory"]
    assert client.recover_json_tool_calls('{"name": "rm_rf", "arguments": {}}', names) == []   # unknown tool
    assert client.recover_json_tool_calls('{"name": "list_directory"}', names) == []            # no arguments
    assert client.recover_json_tool_calls('see the config {"path": "x"}', names) == []          # no name key
    assert client.recover_json_tool_calls("no json here at all", names) == []
    assert client.recover_json_tool_calls('{"name": "list_directory", "arguments": {}}', []) == []  # no tools


def test_tool_call_echo_keeps_native_markup_but_cleans_harmony():
    # _tool_call_echo was part of the old hand-rolled loop's text-recovery logic.
    # With deepagents, tool call echo is handled by LangChain's AIMessage/ToolMessage
    # reconstruction. The extract_tool_calls layer (engine/extract.py) still recovers
    # text-protocol calls, so we verify that recovery still works.
    from omnicode.engine.extract import extract_tool_calls

    # MiniMax XML tool calls are recovered from text
    xml = ('<minimax:tool_call>\n<invoke name="read_file">'
           '<parameter name="path">a.py</parameter></invoke>\n</minimax:tool_call>')
    ext = extract_tool_calls({"content": xml, "tool_calls": []}, None, ["read_file"])
    assert ext.calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]

    # Hermes <tool_call> tags are recovered too
    hermes = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    ext2 = extract_tool_calls({"content": hermes, "tool_calls": []}, None, ["read_file"])
    assert ext2.calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]


def test_loads_lenient_handles_brace_inside_string_value():
    # a `}` inside a string value + trailing junk used to truncate the object → {}
    assert client._loads_lenient('{"content": "if (x) { y }"} trailing') == {"content": "if (x) { y }"}
    assert client._loads_lenient('{"path": "a}b"} junk') == {"path": "a}b"}
    assert client._loads_lenient('{"a": 1}') == {"a": 1}
    assert client._loads_lenient("no json here") == {}


# --- LOW: linkify keeps balanced parens, trims trailing junk -------------

def test_linkify_keeps_balanced_parens():
    out = linkify_urls("see https://en.wikipedia.org/wiki/Python_(programming_language) ok")
    # destination is angle-bracketed so the Markdown parser doesn't cut it at the first ')'
    assert "(<https://en.wikipedia.org/wiki/Python_(programming_language)>)" in out


def test_linkify_trims_trailing_paren_and_period():
    out = linkify_urls("(see https://example.com).")
    assert "[https://example.com](https://example.com)" in out
    assert out.endswith(").")  # the trailing ). stays outside the link


def test_linkify_leaves_existing_markdown_links_untouched():
    src = "[x](https://example.com)"
    assert linkify_urls(src) == src


# --- LOW: estimate_prompt_tokens tolerates a bare-string content part ----

def test_estimate_prompt_tokens_handles_string_part():
    assert capabilities.estimate_prompt_tokens([{"role": "user", "content": ["hello world"]}]) > 0


# --- LOW: ServerConfig rejects negative numeric fields -------------------

def test_serverconfig_rejects_negative_numeric_fields():
    from pydantic import ValidationError

    from omnicode.config.models import ServerConfig
    for field in ("max_kv_size", "num_draft_tokens", "decode_concurrency",
                  "prompt_concurrency", "prefill_step_size", "kv_group_size"):
        with pytest.raises(ValidationError):
            ServerConfig(**{field: -1})
    assert ServerConfig(quantized_kv_start=0).quantized_kv_start == 0  # 0 is a valid start index


# --- 2nd pass: store.load salvages valid entries instead of wiping everything ---

def test_config_load_salvages_valid_servers(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from omnicode.config import store
    from omnicode.config.models import ConfigFile, ServerConfig

    store.save(ConfigFile(servers=[ServerConfig(name="Good", model="/m")]))
    p = store.config_path()
    data = json.loads(p.read_text())
    data["servers"].append({"id": "bad", "name": "Bad", "model": "/x", "max_kv_size": -5})  # out of range
    p.write_text(json.dumps(data))

    loaded = store.load()
    names = [s.name for s in loaded.servers]
    assert "Good" in names and "Bad" not in names  # one bad field no longer wipes the whole file


def test_chat_load_salvages_valid_chats(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from omnicode.chat import store as cstore
    from omnicode.chat.models import Chat, ChatStoreFile

    cstore.save(ChatStoreFile(chats=[Chat(title="Keep")]))
    p = cstore.chats_path()
    data = json.loads(p.read_text())
    data["chats"].append({"id": "bad", "title": "Bad", "messages": [{"role": "INVALID"}]})
    p.write_text(json.dumps(data))

    loaded = cstore.load()
    titles = [c.title for c in loaded.chats]
    assert "Keep" in titles and "Bad" not in titles


# --- 2nd pass: refuse switching chats while a main generation is in flight ---

def test_chat_switch_refused_during_main_generation():
    from omnicode.chat.models import Chat, ChatStoreFile
    from omnicode.screens.chat import ChatScreen

    cur, other = Chat(title="current"), Chat(title="other")
    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = cur
    cs.data = ChatStoreFile(chats=[cur, other])
    cs._gen_main = True
    opened: list = []
    notes: list = []
    cs._open_chat = lambda c: opened.append(c)
    cs.notify = lambda *a, **k: notes.append(a)
    cs._reselect_current_chat = lambda: None

    class _Ev:
        class item:
            chat_id = other.id

    cs._chat_selected(_Ev())
    assert opened == [] and notes  # refused mid-generation (didn't switch, did notify)

    cs._gen_main = False
    cs._chat_selected(_Ev())
    assert opened == [other]  # switches when idle


# --- 3rd pass: absolute host paths inside fs_root must not double (2026-07-24) ---

def test_strip_fs_root_maps_absolute_paths_to_virtual():
    """Models emit absolute paths (the system note names the working dir absolutely),
    but LocalShellBackend(virtual_mode=True) interprets every path relative to root —
    '/work/proj/index.html' would land at '{root}/work/proj/index.html'. The middleware
    must strip the root prefix so the file lands where the model meant it."""
    from omnicode.core.deepagents.middleware import _strip_fs_root

    root = "/work/proj"
    assert _strip_fs_root(root, "/work/proj/index.html") == "/index.html"
    assert _strip_fs_root(root, "/work/proj") == "/"
    assert _strip_fs_root(root, "/work/proj/sub/dir/f.py") == "/sub/dir/f.py"
    assert _strip_fs_root(root, "index.html") == "index.html"       # relative untouched
    assert _strip_fs_root(root, "/etc/passwd") is None              # outside root → reject
    assert _strip_fs_root(root, "/work/project/x") is None          # prefix-collision → reject
    # the observed typo case: fused "Desktop/Test" → "DesktopTest" is outside root
    assert _strip_fs_root("/Users/n/Desktop/Test", "/Users/n/DesktopTest/i.html") is None


def test_file_tools_middleware_normalizes_absolute_path(tmp_path):
    """End-to-end: write_file with an absolute host path inside the root writes to the
    intended location through a real LocalShellBackend — no doubled directory tree."""
    import os

    from deepagents.backends.local_shell import LocalShellBackend
    from omnicode.core.deepagents.middleware import (
        _LISFileToolsMiddleware,
        _ToolEventCapture,
    )

    backend = LocalShellBackend(root_dir=str(tmp_path), virtual_mode=True)
    mw = _LISFileToolsMiddleware(permission=None, capture=_ToolEventCapture(), fs_root=str(tmp_path))

    class _Req:
        def __init__(self, tool_call):
            self.tool_call = tool_call

        def override(self, tool_call=None):
            return _Req(tool_call or self.tool_call)

    async def handler(request):
        args = request.tool_call["args"]
        return backend.write(args["file_path"], args["content"])

    abs_path = os.path.join(str(tmp_path), "index.html")
    req = _Req({"name": "write_file", "args": {"path": abs_path, "content": "hi"}, "id": "1"})
    asyncio.run(mw.awrap_tool_call(req, handler))

    assert (tmp_path / "index.html").read_text() == "hi"
    # No doubled tree: nothing may exist under {root}/{root-without-leading-slash}
    assert not (tmp_path / str(tmp_path).lstrip("/")).exists()


def test_file_tools_middleware_rejects_absolute_path_outside_root(tmp_path):
    """An absolute path OUTSIDE the root (e.g. a model typo fusing 'Desktop/Test' into
    'DesktopTest') must NOT be passed to the virtual_mode backend — it would join it
    under root and create a garbage directory tree. The middleware returns an error
    ToolMessage instead, telling the model to use relative paths."""
    import os

    from deepagents.backends.local_shell import LocalShellBackend
    from omnicode.core.deepagents.middleware import (
        _LISFileToolsMiddleware,
        _ToolEventCapture,
    )

    backend = LocalShellBackend(root_dir=str(tmp_path), virtual_mode=True)
    mw = _LISFileToolsMiddleware(permission=None, capture=_ToolEventCapture(), fs_root=str(tmp_path))

    class _Req:
        def __init__(self, tool_call):
            self.tool_call = tool_call

        def override(self, tool_call=None):
            return _Req(tool_call or self.tool_call)

    handled: list[str] = []

    async def handler(request):
        args = request.tool_call["args"]
        handled.append(args["file_path"])
        return backend.write(args["file_path"], args["content"])

    # typo: root is {tmp}, path is {tmp} with the last separator removed (outside root)
    bad = str(tmp_path).rsplit("/", 1)[0] + str(tmp_path).rsplit("/", 1)[1] + "/index.html"
    req = _Req({"name": "write_file", "args": {"path": bad, "content": "x"}, "id": "1"})
    msg = asyncio.run(mw.awrap_tool_call(req, handler))

    assert handled == [], "handler must not run for an outside-root absolute path"
    assert "outside the working directory" in msg.content
    assert not (tmp_path / bad.lstrip("/")).exists()


# --- Playwright MCP fixes: null-filled optional args + typo'd file:// URLs (2026-07-24) ---

class _FakeMcpResult:
    def __init__(self, text="ok", is_error=False):
        from types import SimpleNamespace

        self.content = [SimpleNamespace(type="text", text=text)]
        self.isError = is_error


class _FakeMcpSession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return _FakeMcpResult()


def _router():
    return {
        "mcp__playwright__browser_take_screenshot": ("playwright", "browser_take_screenshot"),
        "mcp__playwright__browser_navigate": ("playwright", "browser_navigate"),
    }


def test_call_mcp_strips_null_arguments():
    """The reported failure: playwright's browser_take_screenshot errored with
    'Invalid input: expected string, received null' at element/target/filename.
    Optional MCP params are plain "string" in the server schema, so an explicit JSON
    null is a schema violation — call_mcp must omit null-valued args entirely."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_take_screenshot",
        {"element": None, "filename": None, "target": None, "fullPage": None, "type": "png"},
    ))
    assert out == "ok"
    _, forwarded = session.calls[0]
    assert forwarded == {"type": "png"}


def test_call_mcp_aliases_ref_to_target():
    """The current @playwright/mcp renamed the element-reference arg `ref` → `target`
    (browser_click/type/hover require `target`). Models trained on the older API emit
    `ref` and get "expected string, received undefined at target" — alias it."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate",  # any routed name; the args are what matter
        {"element": "Click me button", "ref": "e3", "url": None},
    ))
    assert out == "ok"
    _, forwarded = session.calls[0]
    assert forwarded == {"element": "Click me button", "target": "e3"}


def test_call_mcp_unwraps_quoted_target():
    """A model copying the ref out of a snapshot with literal quotes sends target=\"'e3'\",
    and playwright parses it as a CSS selector: \"Unexpected token while parsing css
    selector '.\" Unwrap the quotes; leave real selectors containing quotes alone."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", {"target": "'e3'"},
    ))
    assert session.calls[0][1] == {"target": "e3"}

    session2 = _FakeMcpSession()
    selector = 'input[name="q"]'
    asyncio.run(mcp.call_mcp(
        {"playwright": session2}, _router(),
        "mcp__playwright__browser_navigate", {"target": selector},
    ))
    assert session2.calls[0][1] == {"target": selector}


def test_call_mcp_repairs_mangled_target():
    """Reported failure: the model pasted the snapshot line format into `target`
    ('"link" [text=Click me]'), and playwright parsed it as CSS: "Error while parsing
    selector". Repair: a pasted snapshot line yields the bare ref; an invented
    [text=…] attribute becomes Playwright's text engine selector. Real CSS selectors
    and plain refs pass through untouched."""
    from omnicode.core.tools import mcp

    # Pasted snapshot line → bare ref
    session = _FakeMcpSession()
    asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate",
        {"target": '- link "Click me" [ref=e6]'},
    ))
    assert session.calls[0][1] == {"target": "e6"}

    # Invented attribute selector → text engine (the exact reported error string)
    session = _FakeMcpSession()
    asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate",
        {"target": '"link" [text=Click me]'},
    ))
    assert session.calls[0][1] == {"target": "text=Click me"}

    # Real CSS selector with attributes is left alone
    session = _FakeMcpSession()
    selector = 'a[href="/about"]'
    asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", {"target": selector},
    ))
    assert session.calls[0][1] == {"target": selector}

    # Plain ref untouched
    session = _FakeMcpSession()
    asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", {"target": "e12"},
    ))
    assert session.calls[0][1] == {"target": "e12"}


def test_args_schema_treats_spec_required_fields_as_optional():
    """@playwright/mcp marks browser_take_screenshot's `type`/`scale` as required in the
    schema yet defaults them server-side, so a model calling it with {} is legitimate.
    The pydantic args schema must not reject what the server would accept — every field
    is Optional and unset fields stay None (then get stripped before the MCP call)."""
    from omnicode.core.deepagents.tools import _spec_to_args_schema

    spec = {
        "type": "function",
        "function": {
            "name": "shot",
            "description": "take a screenshot",
            "parameters": {
                "type": "object",
                "required": ["type", "scale"],
                "properties": {
                    "type": {"type": "string"},
                    "scale": {"type": "string"},
                    "filename": {"type": "string"},
                },
            },
        },
    }
    model = _spec_to_args_schema(spec)()  # must NOT raise ValidationError
    assert model.type is None and model.scale is None and model.filename is None


def test_playwright_descriptions_are_hardened():
    """The model never sees `page` in browser_evaluate (its function arg is an arrow
    function, not Playwright code) and the canonical element arg is `target`, not the
    old `ref`. The descriptions we hand the model must say so."""
    from omnicode.core.tools.mcp import _harden_playwright_description

    desc = _harden_playwright_description(
        "browser_evaluate", {"properties": {"function": {}, "target": {}}}, "base")
    assert "NO Playwright page object" in desc

    desc = _harden_playwright_description(
        "browser_click", {"properties": {"element": {}, "target": {}}}, "base")
    assert "`target`" in desc

    desc = _harden_playwright_description(
        "browser_navigate", {"properties": {"url": {}}}, "base")
    assert desc == "base"  # no target-style param → untouched


    """The reported failure: playwright's browser_take_screenshot errored with
    'Invalid input: expected string, received null' at element/target/filename.
    Optional MCP params are plain "string" in the server schema, so an explicit JSON
    null is a schema violation — call_mcp must omit null-valued args entirely."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_take_screenshot",
        {"element": None, "filename": None, "target": None, "fullPage": None, "type": "png"},
    ))
    assert out == "ok"
    _, forwarded = session.calls[0]
    assert forwarded == {"type": "png"}


def test_call_mcp_rejects_nonexistent_file_url(tmp_path):
    """The reported failure: the model navigated to
    file:///Users/nick/DesktopTest/index.html — fusing 'Desktop/Test' into
    'DesktopTest' — and Playwright happily loaded a broken page. A file:// URL to a
    nonexistent path must bounce back with a self-correcting error (pointing at
    open_in_browser) instead of reaching the browser."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    bogus = f"file://{tmp_path}/DesktopTest/index.html"  # tmp_path/DesktopTest doesn't exist
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", {"url": bogus},
    ))
    assert "does not exist" in out
    assert "open_in_browser" in out
    assert session.calls == []  # the navigation never reached the server


def test_call_mcp_allows_existing_file_url(tmp_path):
    from omnicode.core.tools import mcp

    page = tmp_path / "index.html"
    page.write_text("<h1>hi</h1>")
    session = _FakeMcpSession()
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", {"url": f"file://{page}"},
    ))
    assert out == "ok"
    assert session.calls[0][0] == "browser_navigate"


def test_executor_tool_arun_strips_none_from_unset_optional_fields():
    """Root cause of the null-args bug one layer up: _spec_to_args_schema declares
    every OPTIONAL MCP param as Optional[...] default None, so pydantic FILLS all
    unset fields with None and _arun must drop them — otherwise a playwright tool
    receives {element: null, filename: null, target: null, fullPage: null, …}."""
    from omnicode.core.agent import ToolSet
    from omnicode.core.deepagents.tools import toolset_to_langchain_tools

    spec = {
        "type": "function",
        "function": {
            "name": "shot",
            "description": "take a screenshot",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "element": {"type": "string"},
                    "filename": {"type": "string"},
                    "target": {"type": "string"},
                    "fullPage": {"type": "boolean"},
                },
            },
        },
    }
    seen: list[dict] = []

    async def execute(name, args):
        from omnicode.core.agent import ToolOutcome

        seen.append(args)
        return ToolOutcome("ok")

    tools = toolset_to_langchain_tools(ToolSet(specs=[spec], execute=execute, mutating=frozenset()))
    asyncio.run(tools[0]._arun(type="png"))  # model only sets 'type'
    assert seen == [{"type": "png"}]


# --- Agent loop audit fixes (2026-07-24) ---------------------------------

def test_native_fail_key_skips_futile_native_retry():
    """Once a server has rejected the native ``tools`` param (4xx), the model should
    skip the futile native attempt on every subsequent call — no wasted HTTP round-trip
    per turn. This is tracked in ``_native_fail_keys`` (class-level, like _native_ok_keys)."""
    from omnicode.core.deepagents.model import EngineChatModel

    class _StubEngine:
        base_url = "http://test-fail:1234"
        model = "test-model"

        def __init__(self):
            self.calls = []  # list of (messages, tools)

        async def chat(self, messages, tools=None):
            self.calls.append((messages, tools))
            if tools is not None:
                raise Exception("400 Bad Request: tools param not supported")
            # prompted path: return content
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    engine = _StubEngine()
    model = EngineChatModel(engine=engine, tool_specs=[{"type": "function", "function": {"name": "t"}}],
                            tool_names=["t"], model_name="test-model")
    # First call: native fails (4xx), falls back to prompted
    asyncio.run(model._call_with_fallback([{"role": "user", "content": "hi"}]))
    assert engine.calls[0][1] is not None   # tried native
    assert engine.calls[1][1] is None       # fell back to prompted

    # Second call: must skip native entirely (no wasted round-trip)
    engine.calls.clear()
    asyncio.run(model._call_with_fallback([{"role": "user", "content": "hi again"}]))
    assert len(engine.calls) == 1           # only ONE call (prompted), no futile native attempt
    assert engine.calls[0][1] is None       # sent prompted from the start

    # Cleanup: remove the key so other tests aren't affected
    EngineChatModel._native_fail_keys.discard(model._native_key())


def test_call_mcp_times_out_on_hung_tool():
    """A hung MCP tool call must time out after ``_CALL_TIMEOUT`` seconds instead of
    blocking the entire agent turn indefinitely."""
    from omnicode.core.tools import mcp

    class _HungSession:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(999)  # never returns

    # Temporarily lower the timeout so the test is fast
    original = mcp._CALL_TIMEOUT
    mcp._CALL_TIMEOUT = 0.1
    try:
        out = asyncio.run(mcp.call_mcp(
            {"srv": _HungSession()}, {"mcp__srv__slow": ("srv", "slow")},
            "mcp__srv__slow", {"q": "x"},
        ))
    finally:
        mcp._CALL_TIMEOUT = original
    assert "timed out" in out


def test_call_mcp_handles_malformed_result_defensively():
    """A malformed MCP result (non-iterable content, missing .text) must not crash the
    agent loop — it returns an error string instead of raising."""
    from omnicode.core.tools import mcp

    class _BadResult:
        content = "not-iterable"  # strings are iterable but yield chars; this exercises the try
        isError = False

    class _BadSession:
        async def call_tool(self, name, arguments):
            return _BadResult()

    # Should not raise — the result parsing is inside a try/except
    out = asyncio.run(mcp.call_mcp(
        {"srv": _BadSession()}, {"mcp__srv__t": ("srv", "t")},
        "mcp__srv__t", {},
    ))
    # content="not-iterable" iterates as chars; the text join produces garbage but doesn't crash.
    # If isError is False and there's content, we just get the text back.
    assert isinstance(out, str)


def test_call_mcp_handles_chunk_without_text_attr():
    """A text chunk with no ``.text`` attribute must not raise AttributeError."""
    from omnicode.core.tools import mcp
    from types import SimpleNamespace

    class _WeirdChunkResult:
        content = [SimpleNamespace(type="text")]  # has type=text but no .text
        isError = False

    class _WeirdSession:
        async def call_tool(self, name, arguments):
            return _WeirdChunkResult()

    out = asyncio.run(mcp.call_mcp(
        {"srv": _WeirdSession()}, {"mcp__srv__t": ("srv", "t")},
        "mcp__srv__t", {},
    ))
    assert isinstance(out, str)


def test_call_mcp_rejects_non_dict_arguments():
    """A malformed arguments value (string/list instead of dict) must return an error
    string instead of crashing with AttributeError on .items()."""
    from omnicode.core.tools import mcp

    session = _FakeMcpSession()
    out = asyncio.run(mcp.call_mcp(
        {"playwright": session}, _router(),
        "mcp__playwright__browser_navigate", "not a dict",
    ))
    assert "must be an object" in out
    assert session.calls == []


def test_parse_env_handles_quoted_values():
    """``_parse_env`` must use shlex.split so quoted values with spaces survive."""
    from omnicode.core.tools.mcp import _parse_env

    env = _parse_env('API_KEY="secret with spaces" FOO=bar')
    assert env == {"API_KEY": "secret with spaces", "FOO": "bar"}

    # Single quotes too
    env2 = _parse_env("MSG='hello world'")
    assert env2 == {"MSG": "hello world"}

    # None when empty
    assert _parse_env("") is None
    assert _parse_env(None) is None


def test_build_messages_does_not_double_inject_tool_instructions():
    """``_call_with_fallback`` must NOT re-inject tool instructions on the prompted
    retry — they're already in the messages (prepended by ``_build_messages``).
    The old code called ``_inject_prompted_tools`` which prepended them AGAIN."""
    from omnicode.core.deepagents.model import EngineChatModel

    class _CaptureEngine:
        base_url = "http://test-inject:1234"
        model = "test-model"

        def __init__(self):
            self.received_messages = []

        async def chat(self, messages, tools=None):
            self.received_messages.append(messages)
            if tools is not None:
                raise Exception("400: unsupported")
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    engine = _CaptureEngine()
    model = EngineChatModel(
        engine=engine,
        tool_specs=[{"type": "function", "function": {"name": "t", "description": "test"}}],
        tool_names=["t"],
        model_name="test-model",
    )
    msgs = model._build_messages([{"role": "user", "content": "hi"}])
    asyncio.run(model._call_with_fallback(msgs))

    # The prompted retry (second call) received the SAME messages — not a re-injected copy
    prompted_msgs = engine.received_messages[1]
    system_text = prompted_msgs[0].get("content", "") if prompted_msgs else ""
    # Tool instructions should appear exactly ONCE in the system message
    assert system_text.count("# Tools") == 1, "tool instructions must not be double-injected"

    # Cleanup
    EngineChatModel._native_fail_keys.discard(model._native_key())


def test_max_iters_controls_recursion_limit():
    """``run_turn`` must wire ``max_iters`` to the LangGraph ``recursion_limit`` so the
    declared iteration cap actually constrains the loop (was hard-coded to 100)."""
    import inspect
    from omnicode.core.deepagents.adapter import run_turn

    sig = inspect.signature(run_turn)
    assert "max_iters" in sig.parameters, "run_turn must accept max_iters"
    assert sig.parameters["max_iters"].default == 24, "default should match RunPolicy.default"


# --- Prompt + MCP second-pass fixes (2026-07-24) -----------------------

def test_tool_instructions_include_param_types():
    """The tool schema rendered for the model must include per-parameter type annotations
    (``path: str``, not bare ``path``) so the model can emit correctly-shaped arguments
    without guessing — critical for MCP servers with rich schemas."""
    from omnicode.engine.prompted import tool_instructions

    specs = [{
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate to a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "the target URL"},
                    "wait": {"type": "boolean", "description": "wait for load", "enum": [True, False]},
                },
                "required": ["url"],
            },
        },
    }]
    instr = tool_instructions(specs)
    # Required arg: type annotation, no '?'
    assert "url: str" in instr
    # Optional arg: type annotation + '?'
    assert "wait?: bool" in instr
    # Per-parameter description is present
    assert "the target URL" in instr
    # Enum values are surfaced
    assert "one of [True, False]" in instr


def test_coerce_handles_list_arguments():
    """A model that emits arguments as a positional list (``[\"a\", 1]``) instead of a dict
    must not cause the call to be silently dropped — wrap it so the executor can see it."""
    from omnicode.engine.prompted import _coerce

    call = _coerce({"name": "multi_arg", "arguments": ["a", 1]})
    assert call is not None
    assert call["name"] == "multi_arg"
    assert call["arguments"] == {"items": ["a", 1]}


def test_call_mcp_truncates_large_result():
    """A large MCP result (e.g. browser_snapshot's accessibility tree) must be capped
    before being fed back to the model — otherwise it blows the context window."""
    from omnicode.core.tools import mcp
    from types import SimpleNamespace

    big_text = "X" * (mcp._MAX_RESULT_CHARS + 5000)

    class _BigResult:
        content = [SimpleNamespace(type="text", text=big_text)]
        isError = False

    class _BigSession:
        async def call_tool(self, name, arguments):
            return _BigResult()

    out = asyncio.run(mcp.call_mcp(
        {"srv": _BigSession()}, {"mcp__srv__big": ("srv", "big")},
        "mcp__srv__big", {},
    ))
    assert "truncated" in out
    assert len(out) < len(big_text)  # actually smaller than the raw result


def test_missing_file_url_uses_isfile_not_exists(tmp_path):
    """A file:// URL to a directory must be rejected — browsing a directory renders a
    listing or errors in headless mode. The guard uses isfile, not exists."""
    from omnicode.core.tools.mcp import _missing_file_url

    # A directory exists but is not a file → should be flagged as missing
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    assert _missing_file_url({"url": f"file://{subdir}/"}) is not None

    # An actual file → None (OK to browse)
    page = tmp_path / "index.html"
    page.write_text("<h1>hi</h1>")
    assert _missing_file_url({"url": f"file://{page}"}) is None


def test_description_hardening_preserves_note_under_truncation():
    """When the tool's description + appended note exceeds 1024 chars, the note must
    NOT be silently sliced off — the note's usage guidance is the whole point."""
    from omnicode.core.tools.mcp import _harden_playwright_description, _TARGET_NOTE

    # A description that, with the note, would exceed 1024 chars
    long_desc = "A" * 1000
    hardened = _harden_playwright_description(
        "browser_click", {"properties": {"target": {}}}, long_desc)
    # The note must survive
    assert _TARGET_NOTE.strip()[:50] in hardened

