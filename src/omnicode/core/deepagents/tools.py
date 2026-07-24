"""Converts omnicode's existing tool specs (OpenAI function dicts) and executors into
LangChain tools so ``create_deep_agent`` can bind them.

The existing tools (``core/tools/fs.py``, ``core/tools/web.py``, ``core/tools/mcp.py``)
are already spec'd as OpenAI function dicts with an async executor closure. Rather than
re-implementing them, we wrap each one in a custom ``BaseTool`` subclass that:
  1. Describes itself via the original OpenAI spec (name, description, parameters).
  2. Delegates execution to the existing async executor (so all path confinement, web
     search, and MCP routing logic is unchanged).
  3. Passes the full tool-call args through without pydantic stripping extras (critical
     for MCP tools with loose schemas and for the prompted-protocol fallback).

We use a custom ``BaseTool`` rather than ``StructuredTool`` because LangChain's
``_to_args_and_kwargs`` short-circuits to ``(), {}`` when the args_schema has no defined
fields — which would silently drop args for tools whose spec has ``parameters: {}``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import ToolOutcome, ToolSet


def _json_type_to_python(jtype: str | list) -> type:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    if isinstance(jtype, list):
        for t in jtype:
            if t in mapping:
                return mapping[t]
        return str
    return mapping.get(jtype, str)


def _spec_to_args_schema(spec: dict) -> type[BaseModel]:
    """Build a pydantic args schema from an OpenAI function-tool spec's parameters.

    ``extra="allow"`` so args the spec doesn't enumerate (loose MCP schemas) survive
    validation. Always has at least one synthetic field so LangChain's
    ``_to_args_and_kwargs`` doesn't short-circuit to ``()`` (which would drop args).
    """
    fn = spec.get("function") or {}
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    fields: dict = {}
    for name, schema in props.items():
        py_type = _json_type_to_python(schema.get("type", "string"))
        desc = schema.get("description", "")
        # Every field is Optional regardless of the spec's `required` list. This schema
        # is only used for LangChain-side validation/binding — the LLM still sees the
        # original spec verbatim, and the MCP server enforces required params itself
        # with a clear error. Pydantic-level enforcement actively hurts: @playwright/mcp
        # advertises browser_take_screenshot's `type`/`scale` as required yet has
        # server-side defaults, so a model calling it with {} would get a pydantic
        # ValidationError ("type: Field required") for a call the server would accept.
        default = schema.get("default")
        fields[name] = (Optional[py_type], Field(default=default, description=desc)) if desc else (Optional[py_type], default)
    # Ensure the model has at least one field — LangChain strips args when the schema is
    # empty. A synthetic optional field with extra="allow" is harmless.
    if not fields:
        fields["query"] = (Optional[str], Field(default=None, description="Tool argument"))
    return create_model(fn.get("name", "Tool") + "Args", __config__=ConfigDict(extra="allow"), **fields)


class ExecutorTool(BaseTool):
    """A LangChain tool that delegates to a omnicode tool executor closure.

    Unlike ``StructuredTool``, this preserves ALL tool-call args (including extras not in
    the schema) by overriding ``_to_args_and_kwargs`` to pass the full args dict through.
    The ``args_schema`` is still set (for ``convert_to_openai_tool`` and model binding),
    but validation doesn't strip unknown fields.
    """

    _executor: Any = None  # the async (name, args) -> ToolOutcome closure
    _spec_args: list[str] = None  # the arg names from the spec

    def _run(self, **kwargs: Any) -> Any:
        import asyncio

        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        # Drop ALL None values, not just the synthetic "query" field: pydantic fills
        # every unset OPTIONAL schema field with None (playwright's browser_take_screenshot
        # has ~10 — element, filename, target, fullPage…), and forwarding them as JSON
        # null breaks MCP servers whose schema types the param as e.g. "string" (not
        # ["string", "null"]) → "expected string, received null". In JSON Schema an
        # optional argument means OMITTED, never an explicit null. All real args from the
        # model's tool call survive here via extra="allow".
        args = {k: v for k, v in kwargs.items() if v is not None}
        if self._executor is None:
            return f"Unknown tool: {self.name}"
        from ..agent import ToolOutcome

        outcome: ToolOutcome = await self._executor(self.name, args)
        return outcome.text


def toolset_to_langchain_tools(toolset: "ToolSet") -> list[BaseTool]:
    """Convert a omnicode :class:`~omnicode.core.agent.ToolSet` into LangChain tools.

    Each tool spec becomes an :class:`ExecutorTool` whose async ``_arun`` delegates to
    ``toolset.execute``. This preserves the existing executor closure (web/fs/MCP routing,
    path confinement, permission checks) — deepagents just sees native tools.
    """
    tools: list[BaseTool] = []
    for spec in toolset.specs:
        fn = spec.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        description = fn.get("description", "")
        args_schema = _spec_to_args_schema(spec)

        tool = ExecutorTool(
            name=name,
            description=description,
            args_schema=args_schema,
        )
        # Attach the executor closure (can't be a pydantic field on BaseTool)
        object.__setattr__(tool, "_executor", toolset.execute)
        tools.append(tool)
    return tools


def mutating_tool_names(toolset: "ToolSet") -> frozenset[str]:
    """The names of tools that mutate state (for the permission middleware)."""
    return toolset.mutating
