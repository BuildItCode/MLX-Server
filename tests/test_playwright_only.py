"""Enforcement tests: every web/browser action must go through the Playwright MCP
server — never through open_in_browser, a non-playwright MCP server, or a direct fetch."""

import asyncio

import pytest

from omnicode.core.session import Session
from omnicode.core.tools import mcp
from omnicode.engine import prompted


# --- prompt-level rule ------------------------------------------------------


def test_tool_instructions_include_playwright_only_rule():
    instr = prompted.tool_instructions([])
    assert "WEB/BROWSER RULE" in instr
    assert "mcp__playwright__browser_" in instr
    assert "open_in_browser" in instr  # called out as local-files-only
    assert "web_search" in instr       # called out as snippets-only


# --- MCP spec filter: non-playwright servers never expose browser tools ------


@pytest.mark.parametrize("name", [
    "browser_navigate", "browser_click", "browser_snapshot", "navigate", "click",
    "scroll", "screenshot", "go_back", "evaluate", "browser_tabs_list",
])
def test_is_browser_tool_matches_page_actions(name):
    assert mcp._is_browser_tool(name)


@pytest.mark.parametrize("name", [
    "read_file", "search_issues", "list_repos", "create_issue", "get_weather",
    "query_database", "send_message",
])
def test_is_browser_tool_ignores_non_browser_tools(name):
    assert not mcp._is_browser_tool(name)


# --- runtime guard: open_in_browser rejects http(s) URLs ---------------------


def test_open_in_browser_rejects_http_urls(tmp_path):
    outcome = asyncio.run(
        Session._open_in_browser(str(tmp_path), {"url": "https://example.com"}, None)
    )
    assert not outcome.ok
    assert "playwright" in outcome.text.lower()


def test_open_in_browser_still_allows_local_files(tmp_path):
    (tmp_path / "page.html").write_text("<p>hi</p>")
    outcome = asyncio.run(
        Session._open_in_browser(str(tmp_path), {"path": "page.html"}, None)
    )
    assert outcome.ok
    assert "file://" in outcome.text
