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
