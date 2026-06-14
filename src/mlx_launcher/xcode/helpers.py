"""Build the copy-paste configuration Xcode 27 needs to talk to a running server,
two ways: an OpenAI-compatible "Locally Hosted" provider, and an ACP agent.

All functions here are pure string/struct builders (no I/O), so they're trivially
testable."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

from ..config.models import ServerConfig

XCODE_DOCS_URL = "https://developer.apple.com/documentation/xcode/setting-up-coding-intelligence"
PLACEHOLDER_API_KEY = "not-needed"


@dataclass
class ProviderInfo:
    """Settings for Xcode → Settings → Intelligence → add a Locally Hosted provider."""

    base_url: str
    api_key: str
    model: str
    port: int


def openai_provider(cfg: ServerConfig, model_id: Optional[str] = None) -> ProviderInfo:
    return ProviderInfo(
        base_url=cfg.base_url(),
        api_key=PLACEHOLDER_API_KEY,
        model=model_id or cfg.model or "default",
        port=cfg.port,
    )


def acp_command() -> list[str]:
    """How to launch the ACP agent. Prefer the installed console script (its path is
    stable whether run from a venv or a global pipx install); fall back to running the
    module via the current interpreter when running from an uninstalled checkout."""
    exe = shutil.which("mlx-acp-agent")
    if exe:
        return [exe]
    return [sys.executable, "-m", "mlx_launcher.acp.entry"]


@dataclass
class AcpRegistration:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)

    @property
    def json_block(self) -> str:
        return json.dumps(
            {"command": self.command, "args": self.args, "env": self.env}, indent=2
        )

    @property
    def shell_preview(self) -> str:
        import shlex

        return " ".join(shlex.quote(p) for p in [self.command, *self.args])


def acp_registration(cfg: ServerConfig, *, by_config_id: bool = False) -> AcpRegistration:
    """The command/args Xcode should launch for the ACP agent.

    `by_config_id=True` wires the agent to a saved profile id (survives host/port
    edits); otherwise it pins the explicit base-url/model."""
    cmd = acp_command()
    if by_config_id:
        target = ["--config-id", cfg.id]
    else:
        target = ["--base-url", cfg.base_url(), "--model", cfg.model or "default"]
    return AcpRegistration(command=cmd[0], args=[*cmd[1:], *target], env={})
