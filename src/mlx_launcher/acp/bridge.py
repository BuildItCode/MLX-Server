"""HTTP bridge to a running mlx_lm.server's OpenAI-compatible API."""

from __future__ import annotations

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


class MlxBridge:
    """Talks chat completions to the MLX server (streaming and non-streaming)."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

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
        payload = {"model": self.model, "messages": messages, "stream": True}
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        finish = "stop"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=self._headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()  # body isn't read yet in streaming mode
                    raise RuntimeError(_http_error(resp))
                async for raw in resp.aiter_lines():
                    if cancel is not None and cancel():
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
        read_timeout: float = 300.0,
    ) -> dict:
        """One non-streaming completion (used by the agentic tool loop). Returns the
        parsed response JSON."""
        url = f"{self.base_url}/chat/completions"
        payload: dict = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers)
            if resp.status_code >= 400:
                raise RuntimeError(_http_error(resp))
            return resp.json()
