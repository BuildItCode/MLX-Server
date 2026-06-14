"""Data models for saved server profiles and app settings."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Engine = Literal["mlx-lm", "mlx-vlm"]


def _new_id() -> str:
    return uuid.uuid4().hex


class ServerConfig(BaseModel):
    """One saved `mlx_lm.server` profile. Optional fields are omitted from the
    command line when unset, so the server's own defaults apply."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    name: str = "Untitled server"

    # Which runtime serves the model. mlx-lm for text LLMs (`mlx_lm.server`),
    # mlx-vlm for vision-language models (`mlx_vlm.server`). The two binaries
    # accept different flag sets, so this gates how the command line is built.
    engine: Engine = "mlx-lm"

    # Core
    model: str = ""  # local path or HuggingFace repo id
    adapter_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8080

    # Sampling defaults
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None

    # Prompt cache (closest mlx_lm.server has to "cache size")
    prompt_cache_size: Optional[int] = None
    prompt_cache_bytes: Optional[str] = None  # e.g. "2GB"

    # Quantized KV cache — shrinks context memory. mlx-vlm ONLY: mlx_lm.server
    # (0.31.3) has no such flags, so these are emitted only for the mlx-vlm engine.
    kv_bits: Optional[str] = None            # bits for KV quantization, e.g. "8", "4", "3.5"
    kv_quant_scheme: Optional[str] = None    # "uniform" | "turboquant"
    kv_group_size: Optional[int] = None      # group size for uniform quantization (e.g. 64)
    max_kv_size: Optional[int] = None        # cap the KV cache (context) at N tokens
    quantized_kv_start: Optional[int] = None  # token index at which to start quantizing

    # Misc server flags
    trust_remote_code: bool = False
    log_level: LogLevel = "INFO"
    allowed_origins: Optional[str] = None
    draft_model: Optional[str] = None
    num_draft_tokens: Optional[int] = None
    chat_template: Optional[str] = None
    use_default_chat_template: bool = False
    chat_template_args: Optional[str] = None  # JSON string
    decode_concurrency: Optional[int] = None
    prompt_concurrency: Optional[int] = None
    prefill_step_size: Optional[int] = None
    pipeline: bool = False

    # App-specific
    mlx_server_path: Optional[str] = None  # override the resolved binary
    custom_params: str = ""  # free-form extra flags (e.g. --kv-bits 4)

    def server_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", "*") else self.host
        return f"http://{host}:{self.port}"

    def base_url(self) -> str:
        return f"{self.server_url()}/v1"

    def health_url(self) -> str:
        return f"{self.server_url()}/health"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    last_used_id: Optional[str] = None
    theme: str = "mlx-dark"
    mlx_server_path: Optional[str] = None  # global fallback when not on PATH


class ConfigFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    servers: list[ServerConfig] = Field(default_factory=list)
    settings: AppSettings = Field(default_factory=AppSettings)
