"""HTTP bridge to a running mlx_lm.server's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable, Optional

import httpx


def _http_error(resp: httpx.Response) -> str:
    """A concise, useful message from an error response — surface the server's own
    body (mlx servers put the real reason, e.g. a chat-template error, there)."""
    detail = ""
    try:
        data = resp.json()
        detail = data.get("detail") or data.get("error") or ""
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail)
    except Exception:  # noqa: BLE001 — not JSON
        detail = (resp.text or "").strip()
    detail = str(detail).strip().replace("\n", " ")[:400]
    base = f"server returned HTTP {resp.status_code}"
    return f"{base}: {detail}" if detail else base


async def fetch_models(base_url: str, api_key: str = "not-needed", *, timeout: float = 5.0) -> list[str]:
    """Return the model ids served at `base_url` (`GET /v1/models`)."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


_CANCELLED = object()  # sentinel: cancellation observed mid-stream


async def _iter_sse_lines(resp: httpx.Response, cancel: Optional[Callable[[], bool]], poll: float = 0.2):
    """Yield SSE lines from `resp`, waking every `poll` seconds to check `cancel()`
    even while the server is silent (e.g. prefilling a long prompt). Without this the
    read blocks until the next byte arrives, so Stop can't interrupt a stalled stream.
    Yields the `_CANCELLED` sentinel as soon as cancellation is observed."""
    if cancel is None:
        async for raw in resp.aiter_lines():
            yield raw
        return
    line_iter = resp.aiter_lines().__aiter__()
    while True:
        if cancel():
            yield _CANCELLED
            return
        nxt = asyncio.ensure_future(line_iter.__anext__())
        while True:
            done, _pending = await asyncio.wait({nxt}, timeout=poll)
            if done:
                break
            if cancel():  # silent stream + Stop pressed → abort the in-flight read
                nxt.cancel()
                yield _CANCELLED
                return
        try:
            raw = nxt.result()
        except StopAsyncIteration:
            return
        yield raw


class MlxBridge:
    """Talks chat completions to the MLX server (streaming and non-streaming)."""

    def __init__(
        self, base_url: str, model: str, api_key: str = "not-needed",
        max_tokens: Optional[int] = None, chat_template_kwargs: Optional[dict] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # mlx_lm.server defaults to only 512 generated tokens, which truncates a
        # reasoning model mid-thought (empty answer). Send a real budget per request.
        self.max_tokens = max_tokens
        # extra kwargs forwarded to the server's apply_chat_template (e.g. gpt-oss
        # reasoning_effort, Qwen3 enable_thinking). Omitted from the payload when empty.
        self.chat_template_kwargs = chat_template_kwargs or None

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield ('content', text) and ('reason', text) chunks as they stream,
        then a final ('finish', reason) with the raw OpenAI finish_reason (or
        'cancelled')."""
        url = f"{self.base_url}/chat/completions"
        payload: dict = {"model": self.model, "messages": messages, "stream": True}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        finish = "stop"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=self._headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()  # body isn't read yet in streaming mode
                    raise RuntimeError(_http_error(resp))
                async for raw in _iter_sse_lines(resp, cancel):
                    if raw is _CANCELLED:
                        finish = "cancelled"
                        break
                    line = raw.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield ("content", delta["content"])
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield ("reason", reasoning)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        yield ("finish", finish)

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        *,
        read_timeout: float = 600.0,
    ) -> dict:
        """One non-streaming completion (used by the agentic tool loop). Returns the
        parsed response JSON. The window is generous because this call blocks for the
        whole (possibly long, up to max_tokens) response; Stop stays responsive because
        the caller polls cancellation separately (see ChatScreen._bridge_chat)."""
        url = f"{self.base_url}/chat/completions"
        payload: dict = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers)
            if resp.status_code >= 400:
                raise RuntimeError(_http_error(resp))
            return resp.json()
