"""Data models for saved server profiles and app settings."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Engine = Literal["mlx-lm", "mlx-vlm", "vllm-mlx", "llama-cpp"]


def _new_id() -> str:
    return uuid.uuid4().hex


class ServerConfig(BaseModel):
    """One saved `mlx_lm.server` profile. Optional fields are omitted from the
    command line when unset, so the server's own defaults apply."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    name: str = "Untitled server"

    # Which runtime serves the model. mlx-lm for text LLMs (`mlx_lm.server`),
    # mlx-vlm for vision-language models (`mlx_vlm.server`), vllm-mlx for a
    # vLLM-style MLX server (`vllm-mlx serve …`, continuous batching + prefix
    # cache + KV-quant for text *and* vision). Each binary takes a different
    # flag set, so this gates how the command line is built.
    engine: Engine = "mlx-lm"

    # Core
    model: str = ""  # local path or HuggingFace repo id
    adapter_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)

    # Sampling defaults (bounded so a bad value is caught at edit/load time instead of
    # surfacing as a cryptic engine crash on launch)
    temp: Optional[float] = Field(None, ge=0.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)

    # Prompt cache (closest mlx_lm.server has to "cache size")
    prompt_cache_size: Optional[int] = Field(None, ge=0)
    prompt_cache_bytes: Optional[str] = None  # e.g. "2GB"

    # Quantized KV cache — shrinks context memory. mlx-vlm ONLY: mlx_lm.server
    # (0.31.3) has no such flags, so these are emitted only for the mlx-vlm engine.
    kv_bits: Optional[str] = None            # bits for KV quantization, e.g. "8", "4", "3.5"
    kv_quant_scheme: Optional[str] = None    # "uniform" | "turboquant"
    kv_group_size: Optional[int] = None      # group size for uniform quantization (e.g. 64)
    max_kv_size: Optional[int] = None        # cap the KV cache (context) at N tokens
    quantized_kv_start: Optional[int] = None  # token index at which to start quantizing

    # vllm-mlx ONLY (`vllm-mlx serve …`). KV quantization there reuses kv_bits
    # (4 or 8) / kv_group_size / max_kv_size above.
    continuous_batching: bool = True         # --continuous-batching (vLLM's batched scheduler)
    tool_call_parser: Optional[str] = None   # --tool-call-parser (default "auto"); enables native tools
    reasoning_parser: Optional[str] = None   # --reasoning-parser (e.g. gpt_oss/harmony/qwen3)

    # llama.cpp ONLY (`llama-server`). GGUF models with native flags. (continuous_batching
    # above is reused — llama.cpp defaults it on, so we only emit --no-cont-batching to disable.)
    ctx: Optional[int] = Field(None, ge=1)            # -c: prompt context size (blank = from the model)
    n_gpu_layers: Optional[int] = Field(None, ge=0)   # -ngl: layers offloaded to GPU
    n_threads: Optional[int] = Field(None, ge=1)      # -t: CPU threads
    flash_attn: bool = False                          # --flash-attn on
    cache_type_k: Optional[str] = None                # --cache-type-k (f16, q8_0, q4_0, …)
    cache_type_v: Optional[str] = None                # --cache-type-v
    parallel: Optional[int] = Field(None, ge=1)       # --parallel: concurrent server slots
    jinja: bool = False                               # --jinja: enable the Jinja chat-template parser

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
