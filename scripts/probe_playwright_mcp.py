"""End-to-end probe of the @playwright/mcp server through omnicode's own code path.

Opens a session exactly like the app does (open_sessions from core.tools.mcp),
discovers every tool the server exposes, then exercises each browser tool with
representative arguments via call_mcp (which applies _clean_arguments).

It also simulates the deepagents ExecutorTool layer: builds the pydantic args
schema from the tool spec, fills every unset optional field with None exactly
like pydantic validation does, and verifies the None-stripping still produces
a successful call.

Usage: .venv/bin/python scripts/probe_playwright_mcp.py
Exit code 0 = every exercised tool passed, 1 = failures (details on stdout).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omnicode.core.tools.mcp import call_mcp, open_sessions  # noqa: E402

PAGE = (ROOT / "tests" / "fixtures" / "mcp_probe_page.html").as_uri()

SERVER = SimpleNamespace(
    name="playwright",
    transport="stdio",
    command="npx",
    args="-y @playwright/mcp@latest --browser chromium --allow-unrestricted-file-access",
    env="",
    url="",
    enabled=True,
)

FAILURES: list[tuple[str, str]] = []
PASSES: list[str] = []
SKIPPED: list[str] = []


def _fq(tool: str) -> str:
    return f"mcp__playwright__{tool}"


def _record(tool: str, out: str) -> str:
    bad = out.startswith("ERROR") or "### Error" in out or out.startswith("Error:")
    if bad:
        FAILURES.append((tool, out[:400]))
    else:
        PASSES.append(tool)
    return out


async def call(sessions: dict, router: dict, tool: str, args: dict, label: str | None = None) -> str:
    out = await call_mcp(sessions, router, _fq(tool), args)
    _record(label or tool, out)
    return out


def _refs(snapshot: str) -> dict[str, str]:
    """Map element text -> aria ref from a browser_snapshot YAML dump."""
    refs = {}
    for m in re.finditer(r'-\s+(\w+)[^[]*?"([^"]+)"\s*\[ref=(\w+)\]', snapshot):
        refs[m.group(2)] = m.group(3)
    for m in re.finditer(r'(\w+)\s*"([^"]*)"\s*\[ref=(\w+)\]', snapshot):
        refs.setdefault(m.group(2), m.group(3))
    return refs


async def probe_executor_layer(specs: list[dict], sessions: dict, router: dict) -> None:
    """Simulate the deepagents ExecutorTool: pydantic fills unset optionals with None."""
    from omnicode.core.agent import ToolSet, ToolOutcome
    from omnicode.core.deepagents.tools import toolset_to_langchain_tools

    spec = next(s for s in specs if s["function"]["name"] == _fq("browser_take_screenshot"))

    async def execute(name: str, args: dict) -> ToolOutcome:
        text = await call_mcp(sessions, router, name, args)
        return ToolOutcome(text=text)

    toolset = ToolSet(specs=[spec], mutating=frozenset(), execute=execute)
    tool = toolset_to_langchain_tools(toolset)[0]
    # Mimic pydantic: validate an EMPTY call (screenshot needs no args; the server
    # defaults type/scale) so every field becomes None, then _arun must strip them.
    # Before the args-schema fix an empty call raised "type: Field required" —
    # playwright advertises type/scale as required yet defaults them server-side.
    model = tool.args_schema()  # args_schema is the model CLASS; () instantiates
    filled = {k: getattr(model, k, None) for k in model.model_fields}
    out = await tool._arun(**filled)
    _record("browser_take_screenshot[via ExecutorTool empty call]", out)


async def main() -> int:
    stack = AsyncExitStack()
    async with stack:
        sessions, specs, router = await open_sessions(
            stack, [SERVER], on_error=lambda n, r: print(f"connect error {n}: {r}")
        )
        tools = sorted(name for name in (r[1] for r in router.values()))
        print(f"connected; tools discovered ({len(tools)}): {', '.join(tools)}\n")

        # --- navigate ------------------------------------------------------------
        out = await call(sessions, router, "browser_navigate", {"url": PAGE})

        # --- snapshot + ref extraction ------------------------------------------
        snap = await call(sessions, router, "browser_snapshot", {})
        refs = _refs(snap)
        btn_ref = next((r for t, r in refs.items() if "Click me" in t), None)
        input_ref = next((r for t, r in refs.items() if "type here" in t), None)
        select_ref = next((r for t, r in refs.items() if "Alpha" in t or "option" in t.lower()), None)
        link_ref = next((r for t, r in refs.items() if "Jump link" in t), None)
        print(f"refs found: button={btn_ref} input={input_ref} select={select_ref} link={link_ref}")

        # --- click (exact ref the model would copy) ------------------------------
        if btn_ref:
            await call(sessions, router, "browser_click",
                       {"element": "Click me button", "target": btn_ref})
            # Reproduce the user's css-selector parse error: a ref wrapped in
            # literal quotes, as a model might emit when copying "'[ref=e1]'".
            out = await call_mcp(sessions, router, _fq("browser_click"),
                                 {"element": "Click me button", "ref": f"'{btn_ref}'"})
            _record("browser_click[quoted ref]", out)
        else:
            SKIPPED.append("browser_click (no ref)")

        # --- type -----------------------------------------------------------------
        if input_ref:
            await call(sessions, router, "browser_type",
                       {"element": "type here input", "target": input_ref, "text": "hello probe"})
        else:
            SKIPPED.append("browser_type (no ref)")

        # --- evaluate: both documented signatures --------------------------------
        await call(sessions, router, "browser_evaluate",
                   {"function": "() => document.title"})
        if btn_ref:
            await call(sessions, router, "browser_evaluate",
                       {"function": "(element) => element.textContent",
                        "element": "Click me button", "target": btn_ref})
        # Reproduce the user's error: the model assumes a `page` argument. The server
        # MUST reject it (there is no page object) — the fix for this failure mode is
        # the hardened description note, so an error here is the CORRECT behavior.
        out = await call_mcp(sessions, router, _fq("browser_evaluate"),
                             {"function": "(page) => page.locator('h1').textContent()"})
        if "Cannot read properties of undefined" in out and ("ERROR" in out or "### Error" in out):
            PASSES.append("browser_evaluate[page-arg correctly rejected]")
        else:
            FAILURES.append(("browser_evaluate[page-arg assumption]",
                             f"expected a page-is-undefined error, got: {out[:300]}"))

        # --- the rest of the toolbox ---------------------------------------------
        await call(sessions, router, "browser_take_screenshot", {})
        await call(sessions, router, "browser_console_messages", {})
        await call(sessions, router, "browser_network_requests", {})
        await call(sessions, router, "browser_press_key", {"key": "Tab"})
        await call(sessions, router, "browser_wait_for", {"text": "Playwright MCP probe", "time": 2})
        await call(sessions, router, "browser_resize", {"width": 900, "height": 700})
        await call(sessions, router, "browser_navigate_back", {})
        await call(sessions, router, "browser_navigate", {"url": PAGE})
        if btn_ref:
            snap2 = await call_mcp(sessions, router, _fq("browser_snapshot"), {})
            refs2 = _refs(snap2)
            btn_ref = refs2.get("Click me", btn_ref)
            link_ref = refs2.get("Jump link", link_ref)
            await call(sessions, router, "browser_hover",
                       {"element": "Click me button", "target": btn_ref})
        if select_ref:
            await call(sessions, router, "browser_select_option",
                       {"element": "select", "target": select_ref, "values": ["b"]})
        if link_ref:
            await call(sessions, router, "browser_click",
                       {"element": "Jump link", "target": link_ref}, label="browser_click[link]")
        await call(sessions, router, "browser_tabs", {"action": "list"})
        await call(sessions, router, "browser_tabs", {"action": "new"}, label="browser_tabs[new]")
        await call(sessions, router, "browser_tabs", {"action": "close"}, label="browser_tabs[close]")

        if "browser_run_code_unsafe" in tools:
            await call(sessions, router, "browser_run_code_unsafe",
                       {"code": "async (page) => await page.title()"})

        # --- deepagents ExecutorTool layer with pydantic null-fill ----------------
        try:
            await probe_executor_layer(specs, sessions, router)
        except Exception as exc:  # noqa: BLE001 — probe bug, not an MCP failure
            FAILURES.append(("ExecutorTool probe (crash)", repr(exc)))

        await call_mcp(sessions, router, _fq("browser_close"), {})

    print(f"\n=== PASS ({len(PASSES)}) ===")
    for t in PASSES:
        print(f"  ok  {t}")
    if SKIPPED:
        print(f"\n=== SKIPPED ({len(SKIPPED)}) ===")
        for t in SKIPPED:
            print(f"  --  {t}")
    if FAILURES:
        print(f"\n=== FAIL ({len(FAILURES)}) ===")
        for t, out in FAILURES:
            print(f"  XX  {t}\n      {out}")
        return 1
    print("\nall exercised playwright tools passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
