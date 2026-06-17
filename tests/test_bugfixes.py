"""Regression tests for the whole-codebase bug-review fixes (2026-06-16)."""

import asyncio
import sys

import pytest

from mlx_launcher import bootstrap
from mlx_launcher.chat import capabilities, client
from mlx_launcher.chat.blocks import linkify_urls


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
    from mlx_launcher.core.agent import _tool_call_echo  # moved into the unified loop

    calls = [{"name": "read_file", "arguments": {"path": "a.py"}}]
    # MiniMax's native XML is echoed verbatim so the model stays in its dialect (re-rendering it
    # as Hermes is what made it drift)
    xml = ('Let me look.\n<minimax:tool_call>\n<invoke name="read_file">'
           '<parameter name="path">a.py</parameter></invoke>\n</minimax:tool_call>')
    assert _tool_call_echo(xml, "", calls) == xml
    # anything else is rebuilt as a clean prose + <tool_call> turn (nudges a drifted model back,
    # and avoids gpt-oss Harmony's empty final channel / nested tokens)
    echo = _tool_call_echo("", "thinking", calls)
    assert "<tool_call>" in echo and "read_file" in echo


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

    from mlx_launcher.config.models import ServerConfig
    for field in ("max_kv_size", "num_draft_tokens", "decode_concurrency",
                  "prompt_concurrency", "prefill_step_size", "kv_group_size"):
        with pytest.raises(ValidationError):
            ServerConfig(**{field: -1})
    assert ServerConfig(quantized_kv_start=0).quantized_kv_start == 0  # 0 is a valid start index


# --- 2nd pass: store.load salvages valid entries instead of wiping everything ---

def test_config_load_salvages_valid_servers(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mlx_launcher.config import store
    from mlx_launcher.config.models import ConfigFile, ServerConfig

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
    from mlx_launcher.chat import store as cstore
    from mlx_launcher.chat.models import Chat, ChatStoreFile

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
    from mlx_launcher.chat.models import Chat, ChatStoreFile
    from mlx_launcher.screens.chat import ChatScreen

    cur, other = Chat(title="current"), Chat(title="other")
    cs = ChatScreen.__new__(ChatScreen)
    cs.chat = cur
    cs.data = ChatStoreFile(chats=[cur, other])
    cs._gen = {"main": True}
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

    cs._gen = {"main": False}
    cs._chat_selected(_Ev())
    assert opened == [other]  # switches when idle
