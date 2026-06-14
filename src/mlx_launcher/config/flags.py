"""Turn a ServerConfig into the engine's server command line.

Pure functions, no I/O — heavily unit-tested. A flag is emitted only when its
field is set (so the server's own defaults apply otherwise); `custom_params` is
shell-split and appended verbatim.

`mlx_lm.server` and `mlx_vlm.server` take *different* flag sets — passing an
mlx-lm-only flag (e.g. `--temp`, `--max-tokens`, `--pipeline`) to mlx-vlm makes
its argparse abort. So the flag tables are gated per engine; mlx-vlm-specific
tuning (`--kv-bits`, `--max-kv-size`, `--enable-thinking`, …) rides through the
Custom params box, where those flags are native."""

from __future__ import annotations

import shlex

from .models import ServerConfig

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

# Value-bearing flags accepted only by `mlx_lm.server`.
_MLX_LM_VALUE_FLAGS: list[tuple[str, str]] = [
    ("temp", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("prompt_cache_size", "--prompt-cache-size"),
    ("prompt_cache_bytes", "--prompt-cache-bytes"),
    ("allowed_origins", "--allowed-origins"),
    ("num_draft_tokens", "--num-draft-tokens"),
    ("chat_template", "--chat-template"),
    ("chat_template_args", "--chat-template-args"),
    ("decode_concurrency", "--decode-concurrency"),
    ("prompt_concurrency", "--prompt-concurrency"),
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
        # remote code. KV-cache / thinking flags go via custom_params.
        return _SHARED_VALUE_FLAGS, _SHARED_BOOL_FLAGS
    return (
        _SHARED_VALUE_FLAGS + _MLX_LM_VALUE_FLAGS,
        _SHARED_BOOL_FLAGS + _MLX_LM_BOOL_FLAGS,
    )


def build_args(cfg: ServerConfig) -> list[str]:
    """The flags only (without the binary path)."""
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


def preview_command(cfg: ServerConfig, mlx_path: str | None = None) -> str:
    """A copy-pasteable, shell-quoted preview of the launch command."""
    binary = mlx_path or ("mlx_vlm.server" if cfg.engine == "mlx-vlm" else "mlx_lm.server")
    return " ".join(shlex.quote(p) for p in build_argv(cfg, binary))
