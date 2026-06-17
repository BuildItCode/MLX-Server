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


def test_recover_stripped_harmony_splits_leaked_channel_names():
    # some gpt-oss servers strip the <|...|> tokens but leak the channel/role NAMES
    # inline ('analysis…assistantfinal…'); recover the clean answer + reasoning.
    from mlx_launcher.chat.client import parse_harmony, recover_stripped_harmony

    leak = ('analysisThe user says "hello". A simple greeting. I should respond politely. '
            'Probably ask how I can help.assistantfinalHello! How can I assist you today?')
    content, reason = recover_stripped_harmony(leak)
    assert content == "Hello! How can I assist you today?"
    assert reason.startswith("The user says") and "assistant" not in reason
    assert recover_stripped_harmony("assistantfinalHi!") == ("Hi!", "")  # no-reasoning variant
    # normal prose / literal-token form must NOT be touched (no false positives)
    assert recover_stripped_harmony("Hello! How can I help?") is None
    assert recover_stripped_harmony("analysis of the dataset shows clusters") is None  # space, not glued
    assert recover_stripped_harmony("<|channel|>analysis<|message|>x<|return|>") is None
    # parse_harmony routes both the literal and the stripped form to (content, reason)
    assert parse_harmony(leak)[0] == "Hello! How can I assist you today?"


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


def test_recover_loose_tool_calls_handles_stripped_gpt_oss():
    # gpt-oss on mlx_lm can strip the Harmony delimiters, leaving the call as bare text:
    #   "...Use web_search function.{ "query": "...", "max_results": 10 }"
    # — recover it when we know the tool name. Conservative: known names + parseable JSON.
    from mlx_launcher.chat.client import recover_loose_tool_calls

    leaked = ('The user asks ... Use web_search function.'
              '{ "query": "UEFA matches this week", "max_results": 10 }')
    assert recover_loose_tool_calls(leaked, ["web_search"]) == [
        {"name": "web_search", "arguments": {"query": "UEFA matches this week", "max_results": 10}}
    ]
    # unknown tool names are never recovered
    assert recover_loose_tool_calls(leaked, ["other_tool"]) == []
    # prose that merely MENTIONS a tool (no JSON right after) is not a call
    assert recover_loose_tool_calls("you could use web_search for this", ["web_search"]) == []
    # the JSON must be near the name, not paragraphs away
    far = "web_search is great. " + "x " * 50 + '{"query": "y"}'
    assert recover_loose_tool_calls(far, ["web_search"]) == []


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


def test_chatmessage_persists_tool_fields():
    from mlx_launcher.chat.models import ChatMessage

    tool = ChatMessage(role="tool", tool_name="read_file", text="data")
    assert ChatMessage.model_validate(tool.model_dump()).tool_name == "read_file"
    call = ChatMessage(role="assistant", text="", tool_calls=[{"name": "x", "arguments": {"k": 1}}])
    round_tripped = ChatMessage.model_validate(call.model_dump())
    assert round_tripped.tool_calls[0] == {"name": "x", "arguments": {"k": 1}}


def test_tool_steps_round_trip_into_history():
    # The agentic loop persists its tool exchange; build_openai_messages must replay it as the
    # universal text protocol so a follow-up turn keeps the work context.
    from mlx_launcher.chat.client import build_openai_messages
    from mlx_launcher.chat.models import Chat, ChatMessage

    chat = Chat(messages=[
        ChatMessage(role="user", text="add a flag"),
        ChatMessage(role="assistant", text="",  # pure tool-call turn (no prose)
                    tool_calls=[{"name": "read_file", "arguments": {"path": "a.py"}},
                                {"name": "read_file", "arguments": {"path": "b.py"}}]),
        ChatMessage(role="tool", tool_name="read_file", text="AAA"),
        ChatMessage(role="tool", tool_name="read_file", text="BBB"),
        ChatMessage(role="assistant", text="Done."),
    ])
    msgs = build_openai_messages(chat)

    # the tool-call turn carries the calls as <tool_call> tags (no prose ⇒ just the tags)
    call_turn = next(m for m in msgs if m["role"] == "assistant" and "<tool_call>" in m["content"])
    assert call_turn["content"].startswith("<tool_call>")
    assert '"name": "read_file"' in call_turn["content"] and '"path": "a.py"' in call_turn["content"]
    # the two tool results render as <tool_response> and coalesce into ONE user turn (so the
    # original "add a flag" ask + the merged results = 2 user turns, not 3)
    user_turns = [m for m in msgs if m["role"] == "user"]
    assert len(user_turns) == 2
    results = next(m["content"] for m in user_turns if "<tool_response" in m["content"])
    assert "AAA" in results and "BBB" in results


def test_continue_keeps_tool_context_and_coalesces():
    # The reported bug: typing "continue" made the model re-read everything. With the exchange
    # persisted, the prior read is still in context, and the dangling tool result + "continue"
    # merge into one user turn (strict alternating templates 500 on two consecutive user turns).
    from mlx_launcher.chat.client import build_openai_messages
    from mlx_launcher.chat.models import Chat, ChatMessage

    chat = Chat(messages=[
        ChatMessage(role="user", text="go"),
        ChatMessage(role="assistant", text="", tool_calls=[{"name": "read_file", "arguments": {"path": "x.py"}}]),
        ChatMessage(role="tool", tool_name="read_file", text="SECRET_CONTENTS"),
        ChatMessage(role="user", text="continue"),
    ])
    msgs = build_openai_messages(chat)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]  # result + continue merged
    blob = "\n".join(m["content"] for m in msgs)
    assert "SECRET_CONTENTS" in blob  # the file it already read is still in context
    assert msgs[-1]["content"].endswith("continue")


def test_scaled_max_tokens(monkeypatch):
    from mlx_launcher.chat import client as cl

    # 128k model: an explicit KV cap → ~1/4 of it; no cap → ~1/6 of the model max
    monkeypatch.setattr(cl.capabilities, "context_window", lambda m: 131072)
    assert cl.scaled_max_tokens("m", context_cap=32768) == 8192          # 1/4 of the cap
    assert cl.scaled_max_tokens("m", context_cap=None) == 131072 // 6     # 1/6 of model max
    # a cap is bounded by the model max, and the ceiling bounds a huge cap
    assert cl.scaled_max_tokens("m", context_cap=10**9) == 32768         # min(131072//4, 65536)
    # tiny context → the floor keeps a reasoning model from being starved
    monkeypatch.setattr(cl.capabilities, "context_window", lambda m: 12000)
    assert cl.scaled_max_tokens("m", context_cap=8192) == 4096           # max(8192//4, 4096)
    # unknown context window → the fixed fallback
    monkeypatch.setattr(cl.capabilities, "context_window", lambda m: None)
    assert cl.scaled_max_tokens("m") == cl.DEFAULT_MAX_TOKENS


def test_tool_phrase_is_natural_language():
    from mlx_launcher.screens.chat import _tool_phrase

    assert _tool_phrase("read_file", {"path": "src/app.py"}) == "Reading src/app.py"
    assert _tool_phrase("write_file", {"path": "a.py"}) == "Writing a.py"
    assert _tool_phrase("edit_file", {"path": "a.py"}) == "Editing a.py"
    assert _tool_phrase("list_directory", {}) == "Listing ."
    assert _tool_phrase("delete_path", {"path": "x"}) == "Deleting x"
    assert _tool_phrase("run_command", {"command": "pytest -q"}).startswith("Running")
    assert _tool_phrase("web_search", {"query": "vllm"}).startswith("Searching the web")
    assert _tool_phrase("read_file", {}) == "Reading a file"          # missing arg → graceful
    # MCP / unknown tool → humanized identifier, never raw snake_case
    assert _tool_phrase("github_create_issue", {}) == "Github create issue"


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


def test_sampling_sent_per_request_for_every_engine(monkeypatch):
    # Sampling is sent in the chat body (works on any OpenAI-compatible engine), not as mlx-lm-only
    # launch flags — so temperature etc. apply regardless of engine.
    import asyncio

    import mlx_launcher.acp.bridge as bridge_mod
    from mlx_launcher.acp.bridge import MlxBridge
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.screens.chat import ChatScreen

    # a profile's fields → OpenAI request params, only what the user set
    cfg = ServerConfig(model="/m", temp=0.7, top_p=0.9, top_k=40)  # min_p left unset
    assert ChatScreen._sampling_of(cfg) == {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    assert ChatScreen._sampling_of(ServerConfig(model="/m")) == {}  # nothing set → nothing sent
    assert ChatScreen._sampling_of(None) == {}

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
    asyncio.run(MlxBridge("http://x/v1", "m", sampling={"temperature": 0.7, "top_p": 0.9}).chat(
        [{"role": "user", "content": "hi"}]))
    assert captured["temperature"] == 0.7 and captured["top_p"] == 0.9

    captured.clear()  # no sampling → key omitted entirely
    asyncio.run(MlxBridge("http://x/v1", "m").chat([{"role": "user", "content": "hi"}]))
    assert "temperature" not in captured


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


def test_effective_context_meters_against_configured_setting(monkeypatch):
    # the chat context bar should show used/SETTING (the profile's --max-kv-size),
    # capped by the model's true max — not always used/model-max.
    from mlx_launcher.chat.models import Chat
    from mlx_launcher.screens.chat import ChatScreen

    class Cfg:
        def __init__(self, max_kv_size, engine="mlx-vlm"):  # engine that accepts --max-kv-size
            self.id, self.max_kv_size, self.engine = "s", max_kv_size, engine

    def effective(model, max_kv_size, has_server=True, engine="mlx-vlm"):
        cs = ChatScreen.__new__(ChatScreen)
        cs.chat = Chat(model=model, server_id="s" if has_server else None)
        servers = [Cfg(max_kv_size, engine)] if has_server else []

        class FakeApp:
            class config:
                pass
        FakeApp.config.servers = servers
        monkeypatch.setattr(ChatScreen, "app", property(lambda self: FakeApp()))
        return cs._effective_context()

    assert effective("foo-128k", 8192) == 8192        # setting below model max → the setting
    assert effective("foo-128k", 200000) == 131072    # setting above model max → capped by model
    assert effective("foo-128k", None) == 131072      # no setting → model max (old behavior)
    assert effective("mystery-model", 4096) == 4096   # unknown model → the setting alone
    assert effective("mystery-model", None, has_server=False) is None  # nothing → bar hidden
    # mlx-lm has no --max-kv-size flag → a stale setting is ignored, meter against model max
    assert effective("foo-128k", 8192, engine="mlx-lm") == 131072


def test_editor_gates_fields_by_engine():
    # the simplified editor: visible KV/option fields are DISABLED (not removed) when the
    # engine can't use them; the rest live in manual groups that hide per engine.
    import asyncio

    from textual.app import App as TApp
    from textual.widgets import Select

    from mlx_launcher.screens.editor import EditorScreen, _MANUAL_GROUP_ENGINES

    manual_shown = {  # grp-sampling now shows for every engine (sent per request, not a launch flag)
        "mlx-lm": {"grp-sampling", "grp-shared-adv", "grp-mlxlm-adv"},
        "mlx-vlm": {"grp-sampling", "grp-shared-adv", "grp-kv-extra", "grp-kv-mlxvlm"},
        "vllm-mlx": {"grp-sampling", "grp-kv-extra", "grp-vllm"},
        "llama-cpp": {"grp-sampling", "grp-llamacpp"},
    }
    disabled = {  # which visible KV/option fields are greyed out per engine
        "mlx-lm": {"kv_bits": True, "max_kv_size": True, "turboquant": True, "draft_model": False},
        "mlx-vlm": {"kv_bits": False, "max_kv_size": False, "turboquant": False, "draft_model": False},
        "vllm-mlx": {"kv_bits": False, "max_kv_size": False, "turboquant": True, "draft_model": True},
        "llama-cpp": {"kv_bits": True, "max_kv_size": True, "turboquant": True, "draft_model": True},
    }

    async def go():
        class Host(TApp):
            def on_mount(self):
                self.push_screen(EditorScreen())

        async with Host().run_test() as pilot:
            await pilot.pause(0.1)
            ed = pilot.app.screen
            for engine in manual_shown:
                ed.query_one("#engine", Select).value = engine
                await pilot.pause(0.05)
                shown = {g for g in _MANUAL_GROUP_ENGINES if ed.query_one(f"#{g}").display}
                assert shown == manual_shown[engine], f"{engine}: {shown}"
                for fid, want in disabled[engine].items():
                    got = ed.query_one(f"#{fid}").disabled
                    assert got is want, f"{engine}: #{fid}.disabled={got} != {want}"
                assert ed.query_one("#model").display  # core field always present
                assert ed.query_one("#max_tokens")     # max-tokens moved into manual, still there

    asyncio.run(go())


def test_coding_mode_injects_senior_engineer_system_prompt():
    from mlx_launcher.chat.client import CODING_MODE_INSTRUCTIONS, build_openai_messages
    from mlx_launcher.chat.models import Chat, ChatMessage

    on = build_openai_messages(Chat(coding=True, messages=[ChatMessage(role="user", text="hi")]))
    assert on[0]["role"] == "system"
    assert "senior software engineer" in on[0]["content"]
    assert "VALIDATE" in on[0]["content"] and "tsc --noEmit" in CODING_MODE_INSTRUCTIONS
    # off → no system message at all
    off = build_openai_messages(Chat(messages=[ChatMessage(role="user", text="hi")]))
    assert off[0]["role"] == "user"
    # coding + plan both apply, plan kept LAST (most salient framing)
    both = build_openai_messages(Chat(coding=True, mode="plan", messages=[ChatMessage(role="user", text="hi")]))
    sys = both[0]["content"]
    assert sys.index("senior software engineer") < sys.index("PLAN MODE")


def test_toggle_chip_click_posts_changed_and_reflects_state():
    import asyncio

    from textual import on
    from textual.app import App as TApp
    from textual.containers import Horizontal

    from mlx_launcher.widgets.toggle_chip import ToggleChip

    seen = []

    class T(TApp):
        def compose(self):
            with Horizontal():
                yield ToggleChip("web", "web", id="c")

        @on(ToggleChip.Changed)
        def _h(self, e):
            seen.append((e.key, e.value))

    async def go():
        async with T().run_test() as pilot:
            chip = pilot.app.query_one("#c", ToggleChip)
            assert not chip.value and not chip.has_class("-on")
            await pilot.click("#c")
            assert chip.value and chip.has_class("-on")        # lit when on
            await pilot.click("#c")
            assert not chip.value and not chip.has_class("-on")
            chip.set_value(True)                               # programmatic: no post
            assert chip.has_class("-on")
            chip.set_enabled(False)                            # lock → off + greyed
            assert chip.has_class("-disabled") and not chip.value
            await pilot.click("#c")                            # locked click is a no-op
            assert not chip.value

    asyncio.run(go())
    assert seen == [("web", True), ("web", False)]  # only the two real clicks posted


def test_chip_changed_dispatch_updates_chat_flags():
    from mlx_launcher.chat.models import Chat
    from mlx_launcher.screens.chat import ChatScreen
    from mlx_launcher.widgets.toggle_chip import ToggleChip

    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = Chat()
    cs._update_topbar = lambda: None
    cs._persist = lambda: None
    cs.notify = lambda *a, **k: None
    for key, field in [("web", "web_search"), ("tools", "tools"),
                       ("coding", "coding"), ("reasoning", "reasoning")]:
        cs._chip_changed(ToggleChip.Changed(key, True))
        assert getattr(cs.chat, field) is True
        cs._chip_changed(ToggleChip.Changed(key, False))
        assert getattr(cs.chat, field) is False


def test_connectors_modal_toggles_enabled_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.app import App as TApp

    from mlx_launcher.chat import store
    from mlx_launcher.chat.models import McpServer
    from mlx_launcher.screens.chat import ConnectorsModal
    from mlx_launcher.widgets.toggle_chip import ToggleChip

    data = store.load()
    srv = McpServer(name="My Tools", command="echo", enabled=True)
    store.upsert_mcp(data, srv)
    store.save(data)

    async def go():
        async with TApp().run_test() as pilot:
            await pilot.app.push_screen(ConnectorsModal(data))
            await pilot.pause(0.1)
            modal = pilot.app.screen
            chip = modal.query_one(".connector-chip", ToggleChip)
            assert chip.value is True
            await pilot.click(".connector-chip")
            await pilot.pause(0.05)
            assert srv.enabled is False                       # in-memory flipped
            assert store.load().mcp_servers[0].enabled is False  # and persisted to disk

    asyncio.run(go())


def test_subagent_store_roundtrip_and_delete_detaches(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mlx_launcher.chat import store
    from mlx_launcher.chat.models import Chat, Subagent

    data = store.load()
    sub = Subagent(name="Researcher", server_id="srv1", web_search=True)
    store.upsert_subagent(data, sub)
    chat = Chat(subagent_ids=[sub.id])
    store.upsert_chat(data, chat)
    store.save(data)

    again = store.load()
    assert [s.name for s in again.subagents] == ["Researcher"]
    assert again.chats[0].subagent_ids == [sub.id]
    # delete detaches from every chat
    store.delete_subagent(again, sub.id)
    assert again.subagents == [] and again.chats[0].subagent_ids == []


def test_is_fatal_generation_error_classifies():
    from mlx_launcher.screens.chat import _is_fatal_generation_error

    reshape = RuntimeError("server returned HTTP 500: Generation failed: [reshape] Cannot reshape "
                           "array of size 32768 into shape (2,16,1,512).")
    assert _is_fatal_generation_error(reshape)
    assert _is_fatal_generation_error(RuntimeError("Metal: [METAL] out of memory"))
    # a tools/template rejection is NOT fatal — prompted-mode fallback should still try
    assert not _is_fatal_generation_error(RuntimeError("chat template error: tools unsupported"))
    assert not _is_fatal_generation_error(RuntimeError("connection refused"))


def test_subagent_system_demands_search_and_no_fabrication():
    # The web-search insistence (don't fabricate sources) now lives in the subagent's system note;
    # the tool list itself is prepended by the shared AgentRunner.
    from mlx_launcher.chat.models import Subagent
    from mlx_launcher.screens.chat import ChatScreen

    cs = ChatScreen.__new__(ChatScreen)
    d = cs._subagent_system(Subagent(name="R", web_search=True))
    assert "MUST call web_search" in d and "fabricate" in d.lower()  # mandated when web is on
    # a bare subagent without web search gets no mandate (and no other context here)
    assert "MUST call web_search" not in cs._subagent_system(Subagent(name="R", web_search=False))


def test_slash_command_menu(monkeypatch, tmp_path):
    # Typing "/" opens a command menu that filters as you type; Enter runs the highlighted
    # command; a space (a real message that merely starts with "/") closes it.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.widgets import OptionList

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.chat import _SLASH_COMMANDS, ChatScreen, PromptArea

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            prompt = scr.query_one("#prompt", PromptArea)
            sug = scr.query_one("#slash-suggest", OptionList)

            def menu_for(text):  # deterministic: set the text, recompute the menu
                prompt.load_text(text)
                scr._update_slash_menu()
                return [c for c, _ in scr._slash_items], sug.display

            cmds, shown = menu_for("/")
            assert shown and set(cmds) == _SLASH_COMMANDS   # all commands on a bare "/"
            assert menu_for("/c")[0] == ["/compact"]        # filters by prefix
            assert menu_for("/p")[0] == ["/plan"]
            assert menu_for("/x")[1] is False               # no match → hidden
            assert menu_for("/plan ")[1] is False           # a space → it's a message, not a command
            assert menu_for("hello")[1] is False
            assert menu_for("/plan the rollout")[1] is False

            # key-driven: Enter while the menu is open runs the highlighted command (not "send")
            menu_for("/p")  # highlights /plan
            prompt.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert scr.chat.mode == "plan"            # the command ran
            assert sug.display is False               # menu closed
            assert prompt.text == ""                  # prompt cleared, nothing sent

    asyncio.run(go())


def test_assistant_message_has_a_copy_control(tmp_path, monkeypatch):
    # the user wants to copy a reply's text — every assistant message gets a "⧉ Copy"
    # control that copies its text; user messages don't.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.chat.models import ChatMessage
    from mlx_launcher.screens.chat import ChatScreen  # noqa: F401  (screen pushed below)

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            scr.chat.messages.append(ChatMessage(role="user", text="hi"))
            scr.chat.messages.append(ChatMessage(role="assistant", text="the reply text",
                                                 tps=10, n_tokens=5, elapsed=1.0))
            scr._render_transcript()
            await pilot.pause(0.1)
            copies = list(scr.query(".msg-copy"))
            assert len(copies) == 1, "one copy control (assistant only, not the user msg)"
            assert getattr(copies[-1], "_copy_text", "") == "the reply text"
            captured = {}
            monkeypatch.setattr(app, "copy_text", lambda t: captured.update(t=t) or True)
            scr.on_click(type("E", (), {"widget": copies[-1]})())
            assert captured.get("t") == "the reply text"  # click copied the message text

    asyncio.run(go())


def test_send_button_reflects_focused_pane_not_the_other():
    # the bug: while the subagent (side) is generating, focusing the main pane must show
    # "Send" (main is idle) — NOT stay stuck on "Stop" just because the other pane runs.
    from mlx_launcher.screens.chat import ChatScreen

    class FakeBtn:
        label = "Send"
        variant = "primary"
        border_title = ""

        def set_class(self, *a, **k):
            pass

    btn = FakeBtn()
    cs = ChatScreen.__new__(ChatScreen)
    cs._active_pane = "main"
    cs._side_open = True
    cs._side_sub = None
    cs._gen = {"main": False, "side": True}        # side (subagent) is answering
    cs._cancel_flags = {"main": False, "side": False}
    cs.query_one = lambda *a, **k: btn

    cs._set_active_pane("main")  # focus main (idle) → Send, even though side is running
    assert cs._active_pane == "main" and btn.label == "Send" and btn.variant == "primary"
    cs._set_active_pane("side")  # focus the running side pane → Stop
    assert cs._active_pane == "side" and btn.label == "■ Stop" and btn.variant == "error"
    # and a send to main is allowed while side runs (only the SAME pane being busy blocks)
    cs._active_pane = "main"
    assert cs._gen.get("main", False) is False


def test_subagents_modal_chat_button_returns_id(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.widgets import Button

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.chat import store
    from mlx_launcher.chat.models import Subagent
    from mlx_launcher.screens.chat import SubagentsModal

    data = store.load()
    store.upsert_subagent(data, Subagent(name="Researcher", server_id="srv1"))
    store.save(data)

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            result = {}
            await app.push_screen(SubagentsModal(data), lambda v: result.__setitem__("v", v))
            await pilot.pause(0.1)
            modal = app.screen
            btn = modal.query_one(".sa-chat", Button)
            sid = store.load().subagents[0].id
            assert getattr(btn, "_sa_id", None) == sid
            await pilot.click(".sa-chat")
            await pilot.pause(0.1)
            assert result.get("v") == sid

    asyncio.run(go())


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


# --- audit fixes: tool-call recovery false positives --------------------

def test_recover_loose_tool_calls_ignores_explanatory_prose():
    # the false-positive bug: a model EXPLAINING a tool (its name, then an example JSON
    # further along the sentence) must NOT be executed as a real call — only a
    # punctuation/"function" bridge straight to the `{` counts as a stripped call.
    from mlx_launcher.chat.client import recover_loose_tool_calls

    prose = 'You could call read_file with a path like {"path": "a.txt"} to read it.'
    assert recover_loose_tool_calls(prose, ["read_file", "delete_path"]) == []
    # destructive tools mentioned in prose are likewise safe
    assert recover_loose_tool_calls('use delete_path on the old {"path": "x"} folder',
                                    ["delete_path"]) == []
    # but the genuine stripped artifacts ("name.{json}" / "name function.{json}") still fire
    assert recover_loose_tool_calls('web_search.{"query": "x"}', ["web_search"]) == [
        {"name": "web_search", "arguments": {"query": "x"}}]
    assert recover_loose_tool_calls('call web_search function.{"query": "y"}', ["web_search"]) == [
        {"name": "web_search", "arguments": {"query": "y"}}]


def test_recover_stripped_harmony_keeps_answers_starting_with_analysis():
    # the false-positive bug: an answer that merely STARTS with "analysis" and later
    # contains a word like "finalist"/"finalize" must not be split on a bare "final"
    # (which cut the answer mid-word and hid its front in the reasoning panel).
    from mlx_launcher.chat.client import parse_harmony, recover_stripped_harmony

    assert recover_stripped_harmony("analysis. The finalists were chosen carefully.") is None
    assert recover_stripped_harmony("analysisLet me finalize the report now.") is None
    content, reason = parse_harmony("analysis. The finalists were chosen carefully.")
    assert content == "analysis. The finalists were chosen carefully." and reason == ""
    # the REAL stripped form (explicit assistant…final marker) is still recovered
    assert recover_stripped_harmony("analysisThinking hard.assistantfinalHere it is.") == (
        "Here it is.", "Thinking hard.")


# --- audit fixes: streaming Stop, MCP robustness, teardown, data integrity --

def test_iter_sse_lines_cancels_a_stalled_stream():
    import asyncio

    from mlx_launcher.acp.bridge import _CANCELLED, _iter_sse_lines

    class StalledResp:
        """aiter_lines yields one line then blocks forever (server prefilling)."""

        def aiter_lines(self):
            async def gen():
                yield "data: hello"
                await asyncio.Event().wait()  # never resolves
                yield "unreachable"
            return gen()

    async def go():
        stop = {"v": False}
        out = []

        async def consume():
            async for raw in _iter_sse_lines(StalledResp(), lambda: stop["v"], poll=0.02):
                out.append(raw)

        async def presser():
            while "data: hello" not in out:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)  # let the generator enter the blocking read of line 2
            stop["v"] = True           # Stop pressed while the stream is silent
        await asyncio.wait_for(asyncio.gather(consume(), presser()), timeout=2.0)
        return out

    out = asyncio.run(go())
    assert out[0] == "data: hello" and out[-1] is _CANCELLED  # interrupted, never reached line 2


def test_open_sessions_disambiguates_truncated_tool_name_collisions(monkeypatch):
    import asyncio
    import types
    from contextlib import AsyncExitStack

    from mlx_launcher.chat import mcp_client

    class _ACM:
        def __init__(self, val):
            self._val = val

        async def __aenter__(self):
            return self._val

        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def initialize(self):
            pass

        async def list_tools(self):
            long = "x" * 70  # two names whose 64-char-truncated fq collides
            tool = lambda n: types.SimpleNamespace(name=n, description="d", inputSchema=None)
            return types.SimpleNamespace(tools=[tool(long + "AAA"), tool(long + "BBB")])

    monkeypatch.setattr("mcp.ClientSession", FakeSession)
    monkeypatch.setattr("mcp.StdioServerParameters", lambda **k: None)
    monkeypatch.setattr("mcp.client.stdio.stdio_client", lambda params: _ACM((None, None)))

    servers = [types.SimpleNamespace(name="srv", enabled=True, transport="stdio",
                                     command="echo", args="", env="")]

    async def go():
        async with AsyncExitStack() as stack:
            return await mcp_client.open_sessions(stack, servers)

    _sessions, specs, router = asyncio.run(go())
    names = [s["function"]["name"] for s in specs]
    assert len(names) == 2 and len(set(names)) == 2          # collision disambiguated
    assert all(len(n) <= 64 for n in names) and len(router) == 2


def test_call_mcp_returns_error_string_for_unknown_tool():
    import asyncio

    from mlx_launcher.chat import mcp_client

    out = asyncio.run(mcp_client.call_mcp({}, {}, "mcp__x__missing", {}))
    assert "unknown MCP tool" in out  # an error string, not a raised KeyError


# --- reasoning-effort control -------------------------------------------

def test_reasoning_template_kwargs_maps_per_model_family():
    from mlx_launcher.chat import capabilities as cap

    # gpt-oss → graded reasoning_effort; 'off' clamps to 'low' (it can't fully disable)
    assert cap.reasoning_template_kwargs("openai/gpt-oss-20b", "high") == {"reasoning_effort": "high"}
    assert cap.reasoning_template_kwargs("gpt-oss-120b", "off") == {"reasoning_effort": "low"}
    # Qwen3 → enable_thinking bool (on for any level, off disables thinking)
    assert cap.reasoning_template_kwargs("Qwen3-8B", "medium") == {"enable_thinking": True}
    assert cap.reasoning_template_kwargs("Qwen3-8B", "off") == {"enable_thinking": False}
    # 'auto' (None) sends nothing
    assert cap.reasoning_template_kwargs("gpt-oss-20b", None) == {}
    # an EXPLICIT effort is sent even for a model the heuristic doesn't flag — harmless if the
    # template ignores it, and it lets unrecognized reasoners (e.g. Step) respond to the control
    assert cap.reasoning_template_kwargs("llama-3-8b-instruct", "high") == {"reasoning_effort": "high"}
    assert cap.reasoning_template_kwargs("step-3.7", "medium") == {"reasoning_effort": "medium"}
    # other reasoning families get a best-effort effort hint
    assert cap.reasoning_template_kwargs("deepseek-r1-distill-qwen", "medium") == {"reasoning_effort": "medium"}
    # gpt-oss is now recognized as a reasoning model (so the chip shows for it)
    assert cap.supports_reasoning("openai/gpt-oss-20b") is True


def test_bridge_sends_chat_template_kwargs(monkeypatch):
    import asyncio

    import mlx_launcher.acp.bridge as bridge_mod
    from mlx_launcher.acp.bridge import MlxBridge

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
    br = MlxBridge("http://x/v1", "m", chat_template_kwargs={"reasoning_effort": "high"})
    asyncio.run(br.chat([{"role": "user", "content": "hi"}]))
    assert captured["chat_template_kwargs"] == {"reasoning_effort": "high"}

    # none set → key omitted (the server's template default applies)
    captured.clear()
    asyncio.run(MlxBridge("http://x/v1", "m").chat([{"role": "user", "content": "hi"}]))
    assert "chat_template_kwargs" not in captured


def test_effort_chip_cycles_and_reflects_state():
    from mlx_launcher.chat.models import Chat
    from mlx_launcher.screens.chat import ChatScreen

    class FakeChip:
        def __init__(self):
            self.label = ""
            self.classes = set()

        def update(self, txt):
            self.label = str(txt)

        def set_class(self, add, name):
            (self.classes.add if add else self.classes.discard)(name)

    chip = FakeChip()
    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = Chat(title="t", model="openai/gpt-oss-20b")  # a reasoning model
    cs.query_one = lambda *a, **k: chip
    cs.notify = lambda *a, **k: None
    cs._persist = lambda: None

    assert cs.chat.reasoning_effort is None                       # auto by default
    cs._cycle_effort(); assert cs.chat.reasoning_effort == "off"
    cs._cycle_effort(); assert cs.chat.reasoning_effort == "low"
    cs._cycle_effort(); cs._cycle_effort(); assert cs.chat.reasoning_effort == "high"
    cs._cycle_effort(); assert cs.chat.reasoning_effort is None    # wraps back to auto
    assert chip.label == "effort: auto" and "hidden" not in chip.classes  # shown for gpt-oss


def test_copy_to_clipboard_uses_native_clipboard(monkeypatch):
    # Textual's base copy_to_clipboard only emits OSC 52 (ignored by macOS Terminal.app);
    # ours ALSO pipes to a native CLI so selection-copy (Ctrl/Cmd+C) and the ⧉ controls land.
    import mlx_launcher.app as appmod
    from textual.app import App as BaseApp

    from mlx_launcher.app import MlxLauncherApp

    monkeypatch.setattr(BaseApp, "copy_to_clipboard", lambda self, t: None)  # skip real OSC-52
    monkeypatch.setattr(appmod.sys, "platform", "darwin")
    monkeypatch.setattr(appmod.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(appmod.subprocess, "run", lambda cmd, **k: calls.append((cmd, k.get("input"))))

    app = MlxLauncherApp.__new__(MlxLauncherApp)
    app.copy_to_clipboard("hello selection")
    assert calls and calls[0][0][0].endswith("pbcopy") and calls[0][1] == b"hello selection"


def test_select_part_of_a_reply_and_copy_it(tmp_path, monkeypatch):
    # the user-facing ask: drag to select PART of a model reply, then Ctrl/Cmd+C copies it.
    # Needs the prose rendered as a selectable Markdown WIDGET (a Static+Rich-Markdown is NOT
    # selectable) + the native-clipboard copy.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.events import MouseDown, MouseMove, MouseUp

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.chat.models import ChatMessage
    from mlx_launcher.screens.chat import ChatScreen

    captured = []

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            app.copy_to_clipboard = lambda t: captured.append(t)  # capture what Ctrl/Cmd+C copies
            scr.chat.messages.append(ChatMessage(role="assistant", text="The quick brown fox jumps over the lazy dog."))
            scr._render_transcript()
            await pilot.pause(0.2)
            body = scr.query("#transcript Markdown.msg-body").first()
            await pilot._post_mouse_events([MouseDown], body, offset=(2, 0), button=1)
            for x in (6, 10, 14, 18):
                await pilot._post_mouse_events([MouseMove], body, offset=(x, 0), button=1)
            await pilot._post_mouse_events([MouseUp], body, offset=(18, 0), button=1)
            await pilot.pause(0.05)
            sel = scr.get_selected_text()
            scr.action_copy_text()  # what Ctrl+C / Cmd+C triggers
            assert sel and sel.strip(), "drag produced no selection (prose isn't selectable)"
            assert captured and captured[0] == sel, "copy did not get the selection"

    asyncio.run(go())


def test_linkify_urls_wraps_only_bare_urls():
    from mlx_launcher.chat.blocks import linkify_urls

    assert linkify_urls("see https://x.com/p here") == "see [https://x.com/p](https://x.com/p) here"
    assert linkify_urls("[docs](https://x.com)") == "[docs](https://x.com)"               # md link kept
    assert linkify_urls("[https://x.com](https://x.com)") == "[https://x.com](https://x.com)"  # already linked
    assert linkify_urls("run `curl https://x.com`") == "run `curl https://x.com`"         # inline code kept
    assert linkify_urls("go to https://x.com.") == "go to [https://x.com](https://x.com)."  # trailing dot excluded
    assert linkify_urls("<https://x.com>") == "<https://x.com>"                             # autolink kept
    assert linkify_urls("no links here") == "no links here"


def test_on_click_opens_a_rendered_link(monkeypatch):
    # clicking text that carries a link style (a markdown link or a linkified bare URL)
    # opens it in the browser via app.open_url.
    from mlx_launcher.screens.chat import ChatScreen

    opened = {}

    class FakeApp:
        def open_url(self, url):
            opened["url"] = url

    cs = ChatScreen.__new__(ChatScreen)
    monkeypatch.setattr(ChatScreen, "app", property(lambda self: FakeApp()))

    class Style:
        link = "https://example.com/x"

    class Ev:
        widget = None
        style = Style()

    cs.on_click(Ev())
    assert opened["url"] == "https://example.com/x"


def test_load_knowledge_reads_files_and_folders(tmp_path):
    from mlx_launcher.chat import knowledge

    (tmp_path / "a.md").write_text("Alpha doc")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\x00bytes")          # binary → skipped
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("Gamma in folder")
    (sub / "ignore.bin").write_bytes(b"\x00\x01")                    # non-text in folder → skipped

    out = knowledge.load_knowledge([str(tmp_path / "a.md"), str(tmp_path / "pic.png"), str(sub)])
    assert out.startswith("# Knowledge base")
    assert "Alpha doc" in out and "Gamma in folder" in out
    assert 'name="a.md"' in out and 'name="c.md"' in out
    assert "bytes" not in out and "ignore.bin" not in out           # binaries excluded
    assert knowledge.load_knowledge([]) == ""                       # nothing → empty
    assert knowledge.load_knowledge([str(tmp_path / "missing.md")]) == ""  # absent path → empty


def test_load_knowledge_extracts_pdf_text(tmp_path):
    from mlx_launcher.chat import knowledge

    def make_pdf(text: str) -> bytes:  # a minimal single-page PDF that shows `text`
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
        ]
        stream = b"BT /F1 24 Tf 72 720 Td (" + text.encode() + b") Tj ET"
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        pdf, offs = b"%PDF-1.4\n", []
        for i, body in enumerate(objs, 1):
            offs.append(len(pdf))
            pdf += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
        x = len(pdf)
        pdf += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
        for o in offs:
            pdf += ("%010d 00000 n \n" % o).encode()
        pdf += (b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
                + str(x).encode() + b"\n%%EOF")
        return pdf

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(make_pdf("Quarterly revenue grew 12 percent"))
    out = knowledge.load_knowledge([str(pdf)])
    assert "Quarterly revenue grew 12 percent" in out and 'name="report.pdf"' in out


def test_subagent_system_injects_knowledge(tmp_path):
    from mlx_launcher.chat.models import Subagent
    from mlx_launcher.screens.chat import ChatScreen

    doc = tmp_path / "handbook.md"
    doc.write_text("Company policy: always be kind.")
    cs = ChatScreen.__new__(ChatScreen)

    sub = Subagent(name="HR", system_prompt="You are HR.", knowledge_paths=[str(doc)])
    system = cs._subagent_system(sub)
    assert "You are HR." in system
    assert "Company policy: always be kind." in system and "Knowledge base" in system
    # no knowledge attached → no knowledge block
    assert "Knowledge base" not in cs._subagent_system(Subagent(name="x", system_prompt="hi"))


def test_effort_chip_renders_and_hides_for_non_reasoning_models(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # hermetic: never touch the real store
    import asyncio

    from textual.widgets import Static

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.chat import ChatScreen

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            chip = scr.query_one("#chip-effort", Static)

            # a reasoning model → chip is visible; 'auto' is the unlit default
            scr.chat.model = "openai/gpt-oss-20b"
            scr._sync_effort_chip()
            await pilot.pause()
            assert not chip.has_class("hidden") and not chip.has_class("-on")
            scr._cycle_effort()  # auto → off
            await pilot.pause()
            assert scr.chat.reasoning_effort == "off" and chip.has_class("-on")  # set → lit

            # a model the heuristic doesn't flag as a reasoner → chip stays available (clickable),
            # since the user may want reasoning on a model we don't recognize by name (e.g. Step)
            scr.chat.model = "llama-3-8b-instruct"
            scr._sync_effort_chip()
            await pilot.pause()
            assert not chip.has_class("hidden")

    asyncio.run(go())


# --- slash commands / compaction / browser tool ---------------------------

def test_handle_slash_command_recognizes_known_commands():
    # mode commands (/build /plan /auto), /compact, /help are intercepted; anything else (a path,
    # a sentence that merely starts with "/") is NOT consumed and gets sent as a normal message.
    from mlx_launcher.screens.chat import ChatScreen

    cs = ChatScreen.__new__(ChatScreen)
    cleared = []

    class FakePrompt:
        def load_text(self, t):
            cleared.append(t)

    cs.query_one = lambda *a, **k: FakePrompt()
    cs.notify = lambda *a, **k: None
    modes, compacted = [], []
    cs._set_mode = lambda m: modes.append(m)
    cs._start_compaction = lambda **k: compacted.append(k.get("auto"))

    assert cs._handle_slash_command("/plan") is True
    assert cs._handle_slash_command("/build") is True
    assert cs._handle_slash_command("  /AUTO ") is True            # trim + case-insensitive
    assert modes == ["plan", "build", "auto"]
    assert cs._handle_slash_command("/compact") is True and compacted == [False]
    assert cs._handle_slash_command("/help") is True
    # not commands → not consumed (sent as a normal message)
    assert cs._handle_slash_command("/plan the rollout next week") is False
    assert cs._handle_slash_command("/usr/local/bin/thing") is False
    assert cs._handle_slash_command("hello there") is False
    assert cleared == ["", "", "", "", ""]  # cleared only for the five real commands


def test_set_and_cycle_mode_updates_state_and_chip():
    from mlx_launcher.chat.models import Chat
    from mlx_launcher.screens.chat import ChatScreen

    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = Chat(model="m")
    labels = []

    class FakeChip:
        def update(self, text):
            labels.append(text)

        def set_class(self, *a, **k):
            pass

    cs.query_one = lambda *a, **k: FakeChip()
    cs.notify = lambda *a, **k: None
    cs._update_topbar = lambda: None
    cs._persist = lambda: None

    assert cs.chat.mode == "build"                 # default
    cs._cycle_mode()
    assert cs.chat.mode == "plan"                   # build → plan
    cs._cycle_mode()
    assert cs.chat.mode == "auto"                   # plan → auto
    cs._cycle_mode()
    assert cs.chat.mode == "build"                  # auto → build (wraps)
    cs._set_mode("auto")
    assert cs.chat.mode == "auto"
    cs._set_mode("bogus")                           # invalid → ignored
    assert cs.chat.mode == "auto"
    assert labels[-1] == "mode: auto"               # chip reflects the current mode


def test_maybe_autocompact_only_when_idle_and_over_threshold():
    from mlx_launcher.chat.models import Chat, ChatMessage
    from mlx_launcher.screens.chat import ChatScreen

    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = Chat(model="m", messages=[
        ChatMessage(role="user", text="a"), ChatMessage(role="assistant", text="b"),
        ChatMessage(role="user", text="c"),
    ])
    cs._gen = {"main": False}
    cs._compacting = False
    cs.notify = lambda *a, **k: None
    started = []
    cs._start_compaction = lambda **k: started.append(k.get("auto"))

    cs._context_usage = lambda: (9600, 10000)  # 96% of a healthy window → auto-compact
    cs._maybe_autocompact()
    assert started == [True]

    started.clear()
    cs._context_usage = lambda: (5000, 10000)  # under 95% → nothing
    cs._maybe_autocompact()
    assert started == []

    cs._context_usage = lambda: (9600, 10000)
    cs._gen = {"main": True}                    # mid-reply → never
    cs._maybe_autocompact()
    assert started == []

    cs._gen = {"main": False}
    cs._context_usage = lambda: (1950, 1990)    # window too small to fit a summary → skip
    cs._maybe_autocompact()
    assert started == []

    cs._context_usage = lambda: (9600, 10000)
    cs.chat.messages = [ChatMessage(role="user", text="x" * 100000)]  # one giant turn → don't thrash
    cs._maybe_autocompact()
    assert started == []


def test_slash_plan_command_sets_plan_mode_in_app(tmp_path, monkeypatch):
    # end-to-end: typing /plan into the prompt and sending sets plan mode + reflects on the mode
    # chip, and the command text is consumed (not sent as a message).
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.widgets import Static

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.chat import ChatScreen, PromptArea

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            n0 = len(scr.chat.messages)
            scr.query_one("#prompt", PromptArea).load_text("/plan")
            scr.action_send()
            await pilot.pause(0.1)
            assert scr.chat.mode == "plan"
            assert scr.query_one("#chip-mode", Static).has_class("-on")  # lit for a non-build mode
            assert scr.query_one("#prompt", PromptArea).text == ""   # consumed
            assert len(scr.chat.messages) == n0                       # nothing was sent
            # /auto then drops to no-permission mode
            scr.query_one("#prompt", PromptArea).load_text("/auto")
            scr.action_send()
            await pilot.pause(0.1)
            assert scr.chat.mode == "auto"

    asyncio.run(go())


def test_chat_mode_migrates_legacy_plan_mode_flag():
    # chats saved before the 3-way mode used a plan_mode bool → it maps to mode="plan"
    from mlx_launcher.chat.models import Chat

    assert Chat().mode == "build"                       # default
    assert Chat(plan_mode=True).mode == "plan"          # legacy true → plan
    assert Chat(plan_mode=False).mode == "build"        # legacy false → build
    assert Chat(plan_mode=True, mode="auto").mode == "auto"  # an explicit mode wins
    assert not hasattr(Chat(plan_mode=True), "plan_mode")    # the old field is gone


