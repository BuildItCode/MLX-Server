"""Turn a ServerConfig into the engine's server command line.

Pure functions, no I/O — heavily unit-tested. A flag is emitted only when its
field is set (so the server's own defaults apply otherwise); `custom_params` is
shell-split and appended verbatim.

`mlx_lm.server` and `mlx_vlm.server` take *different* flag sets — passing an
mlx-lm-only flag (e.g. `--prompt-cache-size`, `--pipeline`) to mlx-vlm makes
its argparse abort. So the flag tables are gated per engine: mlx-vlm gets its own
native quantized-KV-cache flags (`--kv-bits`, `--max-kv-size`, …); anything still
without a field (e.g. `--enable-thinking`) rides through the Custom params box."""

from __future__ import annotations

import shlex

from ...models.config import ServerConfig

# Flags both engines accept (config field, CLI flag) — verified against the
# real `mlx_lm.server` and `mlx_vlm.server` --help.
_SHARED_VALUE_FLAGS: list[tuple[str, str]] = [
    ("model", "--model"),
    ("adapter_path", "--adapter-path"),
    ("host", "--host"),
    ("port", "--port"),
    ("max_tokens", "--max-tokens"),
    ("prefill_step_size", "--prefill-step-size"),
    ("draft_model", "--draft-model"),
    ("log_level", "--log-level"),
]

# Value-bearing flags accepted only by `mlx_lm.server`. Sampling (temp/top_p/top_k/min_p) is NOT
# here — it's sent per request in the chat body (see ChatScreen._sampling_of), so it applies to
# every engine instead of only mlx-lm's launch defaults.
_MLX_LM_VALUE_FLAGS: list[tuple[str, str]] = [
    ("prompt_cache_size", "--prompt-cache-size"),
    ("prompt_cache_bytes", "--prompt-cache-bytes"),
    ("allowed_origins", "--allowed-origins"),
    ("num_draft_tokens", "--num-draft-tokens"),
    ("chat_template", "--chat-template"),
    ("chat_template_args", "--chat-template-args"),
    ("decode_concurrency", "--decode-concurrency"),
    ("prompt_concurrency", "--prompt-concurrency"),
]

# Value-bearing flags accepted only by `mlx_vlm.server` — quantized KV cache
# (context). mlx_lm.server has none of these. Verified against `--help`.
_MLX_VLM_VALUE_FLAGS: list[tuple[str, str]] = [
    ("kv_bits", "--kv-bits"),
    ("kv_quant_scheme", "--kv-quant-scheme"),
    ("kv_group_size", "--kv-group-size"),
    ("max_kv_size", "--max-kv-size"),
    ("quantized_kv_start", "--quantized-kv-start"),
]

# (config field, store-true flag), gated per engine.
_SHARED_BOOL_FLAGS: list[tuple[str, str]] = [
    ("trust_remote_code", "--trust-remote-code"),
]
_MLX_LM_BOOL_FLAGS: list[tuple[str, str]] = [
    ("use_default_chat_template", "--use-default-chat-template"),
    ("pipeline", "--pipeline"),
]


def _tables(engine: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(value-flag table, bool-flag table) for an engine."""
    if engine == "mlx-vlm":
        # mlx-vlm shares model/adapter/host/port/draft-model/log-level + trust
        # remote code, plus its own native quantized-KV-cache flags.
        return _SHARED_VALUE_FLAGS + _MLX_VLM_VALUE_FLAGS, _SHARED_BOOL_FLAGS
    return (
        _SHARED_VALUE_FLAGS + _MLX_LM_VALUE_FLAGS,
        _SHARED_BOOL_FLAGS + _MLX_LM_BOOL_FLAGS,
    )


def _vllm_mlx_args(cfg: ServerConfig) -> list[str]:
    """`vllm-mlx serve <model> …` — a subcommand CLI with the model as a positional
    arg and its own flag names (unlike the `--model`-style mlx servers)."""
    args: list[str] = ["serve"]
    if cfg.model:
        args.append(cfg.model)
    if cfg.host:
        args += ["--host", cfg.host]
    args += ["--port", str(cfg.port)]
    if cfg.max_tokens:
        args += ["--max-tokens", str(cfg.max_tokens)]
    if cfg.max_kv_size:
        args += ["--max-kv-size", str(cfg.max_kv_size)]
    # quantized KV cache (context): kv_bits selects the 4- or 8-bit width
    if cfg.kv_bits and str(cfg.kv_bits).strip():
        bits = str(cfg.kv_bits).strip().split(".")[0]  # 4 or 8 (vllm-mlx is integer-only)
        args += ["--kv-cache-quantization", "--kv-cache-quantization-bits", bits]
        if cfg.kv_group_size:
            args += ["--kv-cache-quantization-group-size", str(cfg.kv_group_size)]
    if cfg.continuous_batching:
        args.append("--continuous-batching")
    # native tool calling — vllm-mlx ships gpt-oss/harmony parsers; "auto" picks one
    parser = (cfg.tool_call_parser or "auto").strip()
    if parser:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    if cfg.reasoning_parser and cfg.reasoning_parser.strip():
        args += ["--reasoning-parser", cfg.reasoning_parser.strip()]
    if cfg.trust_remote_code:
        args.append("--trust-remote-code")
    custom = cfg.custom_params.strip()
    if custom:
        args += shlex.split(custom)
    return args


def _looks_like_hf_repo(model: str) -> bool:
    """Heuristic (no I/O): a HuggingFace repo id like `org/repo[:quant]` rather than a
    local GGUF path — has a '/', isn't a filesystem path, and doesn't end in .gguf. Used
    to choose `-hf` (download) vs `-m` (local file) for llama-server."""
    return ("/" in model and not model.endswith(".gguf")
            and not model.startswith(("/", "~", ".")))


def _llama_cpp_args(cfg: ServerConfig) -> list[str]:
    """`llama-server -m <model.gguf> …` — llama.cpp's OpenAI-compatible server. GGUF
    models, native short flags (verified against `llama-server --help`)."""
    args: list[str] = []
    model = cfg.model.strip()
    if model:
        args += (["-hf", model] if _looks_like_hf_repo(model) else ["-m", model])
    if cfg.host:
        args += ["--host", cfg.host]
    args += ["--port", str(cfg.port)]
    if cfg.ctx:
        args += ["-c", str(cfg.ctx)]
    if cfg.n_gpu_layers is not None:
        args += ["-ngl", str(cfg.n_gpu_layers)]
    if cfg.n_threads:
        args += ["-t", str(cfg.n_threads)]
    if cfg.max_tokens:
        args += ["--n-predict", str(cfg.max_tokens)]
    # sampling (temp/top_p/top_k/min_p) is sent per request in the chat body, not as launch
    # flags — see ChatScreen._sampling_of — so it works the same across every engine.
    if cfg.cache_type_k:
        args += ["--cache-type-k", cfg.cache_type_k]
    if cfg.cache_type_v:
        args += ["--cache-type-v", cfg.cache_type_v]
    if cfg.parallel:
        args += ["--parallel", str(cfg.parallel)]
    if cfg.flash_attn:
        args += ["--flash-attn", "on"]  # help shows [on|off|auto] — not a bare flag
    if not cfg.continuous_batching:
        args.append("--no-cont-batching")  # llama.cpp enables continuous batching by default
    if cfg.jinja:
        args.append("--jinja")
    if cfg.chat_template:
        args += ["--chat-template", cfg.chat_template]
    custom = cfg.custom_params.strip()
    if custom:
        args += shlex.split(custom)
    return args


def build_args(cfg: ServerConfig) -> list[str]:
    """The flags only (without the binary path)."""
    if cfg.engine == "vllm-mlx":
        return _vllm_mlx_args(cfg)
    if cfg.engine == "llama-cpp":
        return _llama_cpp_args(cfg)
    value_flags, bool_flags = _tables(cfg.engine)
    args: list[str] = []
    for field, flag in value_flags:
        val = getattr(cfg, field)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        args += [flag, str(val)]
    for field, flag in bool_flags:
        if getattr(cfg, field):
            args.append(flag)
    custom = cfg.custom_params.strip()
    if custom:
        args += shlex.split(custom)
    return args


def build_argv(cfg: ServerConfig, mlx_path: str) -> list[str]:
    """Full argv: [binary, *flags]."""
    return [mlx_path, *build_args(cfg)]


_DEFAULT_BINARY = {"mlx-vlm": "mlx_vlm.server", "vllm-mlx": "vllm-mlx", "llama-cpp": "llama-server"}


def preview_command(cfg: ServerConfig, mlx_path: str | None = None) -> str:
    """A copy-pasteable, shell-quoted preview of the launch command."""
    binary = mlx_path or _DEFAULT_BINARY.get(cfg.engine, "mlx_lm.server")
    return " ".join(shlex.quote(p) for p in build_argv(cfg, binary))
