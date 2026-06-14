from mlx_launcher.chat import capabilities as cap
from mlx_launcher.chat import store
from mlx_launcher.chat.blocks import split_blocks
from mlx_launcher.chat.client import ThinkSplitter, build_openai_messages
from mlx_launcher.chat.models import Attachment, Chat, ChatMessage, Project


def test_web_search_spec():
    from mlx_launcher.chat.tools import web_search_spec

    spec = web_search_spec()
    assert spec["type"] == "function" and spec["function"]["name"] == "web_search"
    assert "query" in spec["function"]["parameters"]["properties"]


def test_mcp_slug_and_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mlx_launcher.chat import mcp_client
    from mlx_launcher.chat.models import McpServer

    assert mcp_client.slug("My Tools!") == "My_Tools"
    data = store.load()
    srv = McpServer(name="x", command="echo", args="hi")
    store.upsert_mcp(data, srv)
    store.save(data)
    assert len(store.load().mcp_servers) == 1
    store.delete_mcp(data, srv.id)
    store.save(data)
    assert store.load().mcp_servers == []


def test_split_blocks():
    blocks = split_blocks('intro text\n\n```json\n{"a": 1}\n```\n\noutro')
    assert [b[0] for b in blocks] == ["prose", "code", "prose"]
    code = next(b for b in blocks if b[0] == "code")
    assert code[1] == "json" and '"a"' in code[2]
    assert split_blocks("just prose")[0] == ("prose", "just prose")


def test_think_splitter_across_chunks():
    sp = ThinkSplitter()
    out = []
    for chunk in ["Hello <th", "ink>thinking ", "here</thi", "nk> world"]:
        out += sp.feed(chunk)
    out += sp.flush()
    reason = "".join(t for k, t in out if k == "reason")
    content = "".join(t for k, t in out if k == "content")
    assert reason == "thinking here"
    assert content == "Hello  world"


def test_harmony_parser_routes_channels_across_chunks():
    from mlx_launcher.chat.client import HarmonyParser

    # the exact gpt-oss shape from the screenshot, split so control tokens straddle
    full = (
        '<|channel|>analysis<|message|>The user says "hello". Respond politely.'
        "<|end|><|start|>assistant<|channel|>final<|message|>Hello! How can I help you today?<|return|>"
    )
    p = HarmonyParser()
    out = []
    for i in range(0, len(full), 7):  # 7-char chunks force mid-token splits
        out += p.feed(full[i:i + 7])
    out += p.flush()
    reason = "".join(t for k, t in out if k == "reason")
    content = "".join(t for k, t in out if k == "content")
    assert reason == 'The user says "hello". Respond politely.'
    assert content == "Hello! How can I help you today?"
    assert "<|" not in content and "<|" not in reason  # all control tokens stripped


def test_harmony_drops_echoed_role_turns():
    from mlx_launcher.chat.client import parse_harmony

    # a server that streams the FULL transcript verbatim, not just the assistant turn
    content, reason = parse_harmony(
        "<|start|>system<|message|>You are helpful.<|end|>"
        "<|start|>user<|message|>hi there<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>greeting<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hi!<|return|>"
    )
    assert content == "Hi!"  # only the final channel becomes the answer
    assert reason == "greeting"
    assert "You are helpful" not in content and "hi there" not in content


def test_harmony_passthrough_for_normal_text():
    from mlx_launcher.chat.client import HarmonyParser, parse_harmony

    p = HarmonyParser()
    out = p.feed("Just a normal answer with a < and a |.") + p.flush()
    assert "".join(t for k, t in out if k == "content") == "Just a normal answer with a < and a |."
    assert not [t for k, t in out if k == "reason"]
    # one-shot helper, no markup
    assert parse_harmony("plain reply") == ("plain reply", "")


def test_parse_harmony_oneshot_splits():
    from mlx_launcher.chat.client import parse_harmony

    content, reason = parse_harmony(
        "<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>the answer<|return|>"
    )
    assert content == "the answer"
    assert reason == "thinking"


def test_parse_harmony_tool_calls_recovers_commentary_call():
    # the exact gpt-oss shape: mlx_lm returns this verbatim (no native tool_calls),
    # so web search silently did nothing. We must recover the call from the text.
    from mlx_launcher.chat.client import parse_harmony, parse_harmony_tool_calls

    raw = (
        "<|channel|>analysis<|message|>I should look this up.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.web_search "
        '<|constrain|>json<|message|>{"query": "uefa fixtures next week"}<|call|>'
    )
    calls = parse_harmony_tool_calls(raw)
    assert calls == [{"name": "web_search", "arguments": {"query": "uefa fixtures next week"}}]
    # and the visible answer is empty (only analysis + the call) — which is exactly
    # why the UI showed "(no answer)" before the fix
    content, reason = parse_harmony(raw)
    assert content == "" and "look this up" in reason

    # two calls in one turn, and a no-constrain variant
    multi = (
        '<|channel|>commentary to=functions.a<|message|>{"x": 1}<|call|>'
        '<|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"y": 2}<|call|>'
    )
    assert parse_harmony_tool_calls(multi) == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    # ordinary text and normal final answers have no calls
    assert parse_harmony_tool_calls("just a normal reply") == []
    assert parse_harmony_tool_calls(
        "<|channel|>final<|message|>here is your answer<|return|>"
    ) == []


def test_prepend_system_never_emits_two_system_messages():
    from mlx_launcher.chat.client import build_openai_messages, prepend_system
    from mlx_launcher.chat.models import Chat, ChatMessage, Project

    # with an existing system message (skill/project) → merge, stay at one
    msgs = build_openai_messages(
        Chat(messages=[ChatMessage(role="user", text="hi")]),
        Project(name="p", instructions="rules"),
        "SKILL",
    )
    prepend_system(msgs, "FS NOTE")
    assert [m["role"] for m in msgs].count("system") == 1
    assert msgs[0]["role"] == "system"
    assert "FS NOTE" in msgs[0]["content"] and "rules" in msgs[0]["content"] and "SKILL" in msgs[0]["content"]

    # with no existing system message → insert one at the front
    msgs2 = build_openai_messages(Chat(messages=[ChatMessage(role="user", text="hi")]))
    prepend_system(msgs2, "FS NOTE")
    assert msgs2[0] == {"role": "system", "content": "FS NOTE"}
    assert [m["role"] for m in msgs2].count("system") == 1


def test_safe_content_survives_markup_breaking_text():
    import pytest
    from textual.content import Content
    from mlx_launcher.widgets.safe_content import plain, title_sub

    nasty = 'imgs = ["https://x?w=600&h=400&fit=crop","y"]'  # crashes Textual markup
    with pytest.raises(Exception):
        Content.from_markup(nasty)  # confirms the naive (markup) path would crash
    # our helpers must NOT raise and must preserve the text literally
    assert isinstance(plain(nasty), Content) and nasty in plain(nasty).plain
    c = title_sub(f"title {nasty}", f"sub {nasty}")
    assert isinstance(c, Content) and "title " in c.plain and "sub " in c.plain


def test_bridge_chat_aborts_promptly_on_stop():
    import asyncio
    from mlx_launcher.screens.chat import ChatScreen

    class FakeBridge:
        async def chat(self, messages, tools=None, *, read_timeout=300.0):
            await asyncio.sleep(30)  # a long, blocking "build a project" response
            return {"choices": []}

    class FakeClient:
        bridge = FakeBridge()

    cs = ChatScreen.__new__(ChatScreen)  # no widgets needed for this unit
    cs._cancel = False

    async def go():
        task = asyncio.ensure_future(cs._bridge_chat(FakeClient(), [], None))
        await asyncio.sleep(0.2)
        cs._cancel = True  # user hits Stop
        return await asyncio.wait_for(task, timeout=2.0)  # must return ~immediately

    assert asyncio.run(go()) is None  # cancelled → None (loop breaks → button resets)


def test_chat_requests_a_real_token_budget(monkeypatch):
    # mlx_lm.server caps at 512 tokens by default, truncating a reasoning model
    # before it answers — the chat must send its own budget on every request.
    import asyncio

    import mlx_launcher.acp.bridge as bridge_mod
    from mlx_launcher.acp.bridge import MlxBridge
    from mlx_launcher.chat.client import DEFAULT_MAX_TOKENS, ChatClient

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            captured.update(json)
            return FakeResp()

    monkeypatch.setattr(bridge_mod.httpx, "AsyncClient", FakeClient)
    asyncio.run(MlxBridge("http://x/v1", "m", max_tokens=2048).chat([{"role": "user", "content": "hi"}]))
    assert captured["max_tokens"] == 2048

    # ChatClient defaults to a generous budget (not the server's 512)
    assert ChatClient("http://x/v1", "m").bridge.max_tokens == DEFAULT_MAX_TOKENS == 16384

    # no budget set → key omitted (server default applies), never sent as 0/None
    captured.clear()
    asyncio.run(MlxBridge("http://x/v1", "m").chat([{"role": "user", "content": "hi"}]))
    assert "max_tokens" not in captured


def test_port_blockers_stops_every_server_on_the_target_port(monkeypatch):
    # All profiles default to :8080, so switching models must free the port by
    # stopping whatever holds it — not just the chat's previously-linked server.
    from mlx_launcher.chat.models import Chat
    from mlx_launcher.screens.chat import ChatScreen

    class Cfg:
        def __init__(self, id, port=8080, host="127.0.0.1"):
            self.id, self.port, self.host, self.name = id, port, host, f"srv-{id}"

    class Mgr:
        def __init__(self, cfg, running=True):
            self.cfg, self._running = cfg, running

        @property
        def is_running(self):
            return self._running

    target = Cfg("new", port=8080)
    mgrs = {
        "A": Mgr(Cfg("A", port=9999)),            # the chat's OLD model on another port → STOP anyway
        "D": Mgr(Cfg("D", port=8080)),            # a different profile on the TARGET port → STOP
        "B": Mgr(Cfg("B", port=8080), running=False),  # not running → ignore
        "C": Mgr(Cfg("C", port=8081)),            # unrelated, different port → ignore
        "new": Mgr(target),                       # the model we're switching TO → never stop
    }

    class FakeApp:
        class config:
            servers = [Cfg("A", port=9999), target]
        def get_manager(self, cid):
            return mgrs.get(cid)
        def running_managers(self):
            return [m for m in mgrs.values() if m.is_running]

    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = Chat(server_id="A")  # this chat's previous server (on a different port)
    fake_app = FakeApp()
    monkeypatch.setattr(ChatScreen, "app", property(lambda self: fake_app))

    blocked = {m.cfg.id for m in cs._port_blockers(target)}
    # A = the chat's old model (freed regardless of port); D = occupant of the target
    # port. NOT B (stopped), C (unrelated port), or the target itself.
    assert blocked == {"A", "D"}


def test_thinking_indicator_animates_then_yields_to_content():
    import asyncio

    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll

    from mlx_launcher.widgets.thinking import ThinkingIndicator

    seen: list[str] = []
    orig = ThinkingIndicator.update

    def rec(self, content="", *a, **k):
        seen.append(getattr(content, "plain", str(content)))
        return orig(self, content, *a, **k)

    class T(App):
        CSS = ".thinking-indicator { color: $accent; }"

        def compose(self) -> ComposeResult:
            with VerticalScroll():
                yield ThinkingIndicator(id="ind", classes="msg-body thinking-indicator")

    async def go():
        app = T()
        async with app.run_test() as pilot:
            ThinkingIndicator.update = rec
            ind = app.query_one("#ind", ThinkingIndicator)
            assert ind._timer is not None  # animating on mount
            start = ind._frame
            await pilot.pause(0.5)
            assert ind._frame > start  # the spinner advanced
            assert any("Thinking" in s for s in seen)
            assert all("[" not in s for s in seen)  # markup-safe frames (no crash)
            ind.stop()  # first answer token → caller stops the spinner
            assert ind._timer is None
            frozen = ind._frame
            await pilot.pause(0.3)
            assert ind._frame == frozen  # no longer animating
            ind.update("real answer")  # content sticks; spinner won't overwrite it
            await pilot.pause(0.2)
            assert seen[-1] == "real answer"

    try:
        asyncio.run(go())
    finally:
        ThinkingIndicator.update = orig


def test_perm_prompt_summaries():
    from mlx_launcher.screens.chat import _perm_prompt

    s, d = _perm_prompt("write_file", {"path": "a.py", "content": "xyz"})
    assert "a.py" in s and "xyz" in d
    s, d = _perm_prompt("run_command", {"command": "ls -la"})
    assert "command" in s.lower() and "ls -la" in d
    s, _ = _perm_prompt("delete_path", {"path": "x/y"})
    assert "x/y" in s


def test_http_error_surfaces_server_body():
    import httpx
    from mlx_launcher.acp.bridge import _http_error

    r1 = httpx.Response(500, json={"detail": "chat template error: tools unsupported"})
    assert "500" in _http_error(r1) and "tools unsupported" in _http_error(r1)
    r2 = httpx.Response(503, text="Service Unavailable\nmodel still loading")
    assert "503" in _http_error(r2) and "model still loading" in _http_error(r2)
    assert _http_error(httpx.Response(500)) == "server returned HTTP 500"


def test_context_window_from_config_name_and_unknown(tmp_path):
    from mlx_launcher.chat import capabilities

    (tmp_path / "config.json").write_text('{"max_position_embeddings": 32768}')
    assert capabilities.context_window(str(tmp_path)) == 32768  # local config wins
    assert capabilities.context_window("mlx-community/Qwen2.5-7B-128k") == 128 * 1024  # name hint
    assert capabilities.context_window("gpt-oss-120b") is None  # 120b is params, not ctx
    assert capabilities.context_window("some-random-model") is None  # unknown → hidden
    assert capabilities.context_window("model-8k4bit") is None  # 'k' followed by a digit ≠ ctx hint
    assert capabilities.context_window("foo-1000000k") is None  # absurd value rejected


def test_estimate_prompt_tokens():
    from mlx_launcher.chat import capabilities

    msgs = [
        {"role": "system", "content": "x" * 40},
        {"role": "user", "content": [{"type": "text", "text": "y" * 40}, {"type": "image_url", "image_url": {}}]},
    ]
    t = capabilities.estimate_prompt_tokens(msgs)
    assert t >= 800  # the image dominates
    assert capabilities.approx_tokens("12345678") == 2


def test_capabilities_heuristics():
    assert cap.supports_vision("mlx-community/Qwen2.5-VL-7B-Instruct")
    assert not cap.supports_vision("Qwen2.5-7B-Instruct")
    assert cap.supports_reasoning("DeepSeek-R1-Distill-Qwen-7B")
    assert not cap.supports_reasoning("Llama-3.1-8B-Instruct")
    assert cap.classify("/x/pic.PNG") == "image"
    assert cap.classify("/x/notes.md") == "text"


def test_build_messages_inlines_text_and_system(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("remember this")
    chat = Chat(model="Qwen2.5-7B")
    chat.messages.append(ChatMessage(role="user", text="summarize", attachments=[Attachment(path=str(f), name="notes.txt", kind="text")]))
    msgs = build_openai_messages(chat, Project(name="P", instructions="Be terse."))
    assert msgs[0] == {"role": "system", "content": "Be terse."}
    assert "remember this" in msgs[1]["content"]  # text attachment inlined


def test_build_messages_image_becomes_multimodal(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)  # not a real png, but has .png ext
    chat = Chat(model="Qwen2.5-VL")
    chat.messages.append(ChatMessage(role="user", text="describe", attachments=[Attachment(path=str(img), name="a.png", kind="image")]))
    content = build_openai_messages(chat)[0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_chat_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data = store.load()
    proj = Project(name="Work")
    chat = Chat(title="hi", model="m", project_id=proj.id)
    store.upsert_project(data, proj)
    store.upsert_chat(data, chat)
    store.save(data)

    again = store.load()
    assert [p.name for p in again.projects] == ["Work"]
    assert len(store.chats_in(again, proj.id)) == 1
    assert len(store.chats_in(again, None)) == 1

    store.delete_project(again, proj.id)
    assert store.get_chat(again, chat.id).project_id is None  # detached, not deleted
