import os

from mlx_launcher.widgets.path_input import path_hint, resolve_path, sanitize_drag


def test_sanitize_strips_quotes():
    assert sanitize_drag("  '/a/b c'  ") == "/a/b c"
    assert sanitize_drag('"/a/b"') == "/a/b"


def test_sanitize_unescapes_backslashes_when_unquoted():
    assert sanitize_drag("/Users/me/My\\ Model") == "/Users/me/My Model"


def test_sanitize_is_idempotent():
    once = sanitize_drag("/Users/me/My\\ Model")
    assert sanitize_drag(once) == once


def test_resolve_expands_user():
    assert resolve_path("~").startswith("/")


def test_hint_detects_kinds(tmp_path):
    assert "folder exists" in path_hint(str(tmp_path))
    assert "HuggingFace" in path_hint("mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert "not found" in path_hint("/no/such/path/xyz123")
    assert path_hint("") == ""

    f = tmp_path / "weights.bin"
    f.write_text("x")
    assert "file exists" in path_hint(str(f))
