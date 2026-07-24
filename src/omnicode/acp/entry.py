"""Console entrypoint for `mlx-acp-agent` — the ACP agent Xcode 27 launches.

This is a self-contained stdio process: it speaks JSON-RPC over stdin/stdout, so
**nothing** may be printed to stdout except protocol traffic. All diagnostics go
to stderr or a log file."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import acp

from ..config import store
from .agent import MlxAcpAgent


def _resolve(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    """Resolve (base_url, model) from flags > env > saved profile (--config-id)."""
    base_url = args.base_url or os.environ.get("MLX_ACP_BASE_URL")
    model = args.model or os.environ.get("MLX_ACP_MODEL")
    if (not base_url or not model) and args.config_id:
        cfg = store.find_server_by_id(args.config_id)
        if cfg is not None:
            base_url = base_url or cfg.base_url()
            model = model or (cfg.model or "default")
    return base_url, model


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlx-acp-agent",
        description="ACP agent bridging an editor (Xcode 27) to a running mlx_lm.server.",
    )
    parser.add_argument("--base-url", help="OpenAI base URL, e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--model", help="Model name as served by mlx_lm.server")
    parser.add_argument("--api-key", default="not-needed", help="Sent as a Bearer token (mlx ignores it)")
    parser.add_argument("--config-id", help="Resolve --base-url/--model from a saved profile id")
    parser.add_argument(
        "--tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable agentic file/terminal tools when the editor grants access (default: on)",
    )
    parser.add_argument("--log-file", help="Write diagnostics here instead of stderr")
    args = parser.parse_args(argv)

    handlers: list[logging.Handler]
    if args.log_file:
        handlers = [logging.FileHandler(args.log_file)]
    else:
        handlers = [logging.StreamHandler(sys.stderr)]
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_url, model = _resolve(args)
    if not base_url or not model:
        print(
            "mlx-acp-agent: need --base-url and --model (or --config-id pointing at a saved profile).",
            file=sys.stderr,
        )
        return 2

    logging.getLogger("mlx-acp-agent").info(
        "starting; base_url=%s model=%s tools=%s", base_url, model, args.tools
    )
    agent = MlxAcpAgent(base_url=base_url, model=model, api_key=args.api_key, use_tools=args.tools)
    try:
        asyncio.run(acp.run_agent(agent))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
