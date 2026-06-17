"""Built-in tools the chat models can call. Currently: web search (DuckDuckGo)."""

from __future__ import annotations

import asyncio

WEB_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web with DuckDuckGo and return the top results (title, URL, "
            "snippet). Use this for current events or facts you are unsure about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "How many results (default 6)"},
            },
            "required": ["query"],
        },
    },
}


def web_search_spec() -> dict:
    return WEB_SEARCH_SPEC


_SEARCH_TIMEOUT = 15.0  # bound the network call so a slow/hung DDG request can't stall the turn


async def run_web_search(query: str, max_results: int = 6) -> str:
    max_results = max(1, min(int(max_results or 6), 10))

    def _search() -> list:
        from ddgs import DDGS

        with DDGS(timeout=_SEARCH_TIMEOUT) as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        # The inner DDGS timeout bounds each HTTP call; wait_for is a backstop in case the
        # library hangs anyway (the abandoned worker thread then just exits on its own).
        rows = await asyncio.wait_for(asyncio.to_thread(_search), timeout=_SEARCH_TIMEOUT + 5)
    except asyncio.TimeoutError:
        return "web_search error: timed out"
    except Exception as exc:  # noqa: BLE001
        return f"web_search error: {exc}"
    if not rows:
        return "No results found."
    out = []
    for r in rows[:max_results]:
        out.append(f"- {r.get('title', '')}\n  {r.get('href', '')}\n  {r.get('body', '')}")
    return "\n".join(out)
