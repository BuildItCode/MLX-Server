from mlx_launcher.config.flags import build_args, build_argv, preview_command
from mlx_launcher.config.models import ServerConfig


def test_unset_optional_flags_are_omitted():
    args = build_args(ServerConfig(model="/m"))
    assert "--model" in args
    assert "--host" in args and "--port" in args  # have defaults
    assert "--temp" not in args
    assert "--adapter-path" not in args
    assert "--prompt-cache-bytes" not in args


def test_set_values_are_emitted():
    c = ServerConfig(model="/m", temp=0.7, max_tokens=2048, prompt_cache_bytes="2GB")
    args = build_args(c)
    assert args[args.index("--temp") + 1] == "0.7"
    assert args[args.index("--max-tokens") + 1] == "2048"
    assert args[args.index("--prompt-cache-bytes") + 1] == "2GB"


def test_booleans_gated():
    c = ServerConfig(model="/m", trust_remote_code=True, pipeline=False)
    args = build_args(c)
    assert "--trust-remote-code" in args
    assert "--pipeline" not in args


def test_custom_params_shlex_split_and_appended():
    c = ServerConfig(model="/m", custom_params="--kv-bits 4 --max-kv-size 8192")
    args = build_args(c)
    # quantized KV cache / context size ride through custom params verbatim
    assert args[-4:] == ["--kv-bits", "4", "--max-kv-size", "8192"]


def test_build_argv_prepends_binary_and_preview_quotes():
    c = ServerConfig(model="/m with space", port=9000)
    argv = build_argv(c, "/usr/bin/mlx_lm.server")
    assert argv[0] == "/usr/bin/mlx_lm.server"
    assert "--port" in argv and "9000" in argv
    assert "'/m with space'" in preview_command(c)


def test_vlm_engine_omits_mlx_lm_only_flags():
    # mlx_vlm.server rejects --temp/--top-p/--pipeline/--prompt-cache-*; gated out.
    # But --max-tokens IS valid for mlx-vlm (verified against the real --help).
    c = ServerConfig(
        engine="mlx-vlm", model="/m", temp=0.7, top_p=0.9, max_tokens=2048,
        pipeline=True, prompt_cache_bytes="2GB",
    )
    args = build_args(c)
    assert "--model" in args and "--host" in args and "--port" in args
    assert args[args.index("--max-tokens") + 1] == "2048"  # shared flag, kept
    assert "--temp" not in args
    assert "--top-p" not in args
    assert "--pipeline" not in args
    assert "--prompt-cache-bytes" not in args


def test_vlm_engine_keeps_shared_flags_and_custom_params():
    c = ServerConfig(
        engine="mlx-vlm",
        model="/m",
        draft_model="/d",
        trust_remote_code=True,
        custom_params="--kv-bits 4 --max-kv-size 8192 --enable-thinking",
    )
    args = build_args(c)
    assert args[args.index("--draft-model") + 1] == "/d"
    assert "--trust-remote-code" in args
    # vlm-native KV-cache / thinking tuning rides through custom params verbatim
    assert args[-5:] == ["--kv-bits", "4", "--max-kv-size", "8192", "--enable-thinking"]


def test_kv_cache_quantization_is_mlx_vlm_only():
    # mlx_vlm.server natively supports quantized KV cache (context); mlx_lm.server
    # has no such flags, so the same fields must NOT be emitted for mlx-lm (passing
    # --kv-bits to mlx_lm.server would make its argparse abort).
    fields = dict(
        kv_bits="3.5", kv_quant_scheme="turboquant", kv_group_size=64,
        max_kv_size=8192, quantized_kv_start=0,
    )
    vlm = build_args(ServerConfig(engine="mlx-vlm", model="/m", **fields))
    assert vlm[vlm.index("--kv-bits") + 1] == "3.5"  # preserved exactly, incl. 3.5
    assert vlm[vlm.index("--kv-quant-scheme") + 1] == "turboquant"
    assert vlm[vlm.index("--max-kv-size") + 1] == "8192"
    assert "--kv-group-size" in vlm and "--quantized-kv-start" in vlm

    lm = build_args(ServerConfig(engine="mlx-lm", model="/m", **fields))
    for flag in ("--kv-bits", "--kv-quant-scheme", "--kv-group-size", "--max-kv-size", "--quantized-kv-start"):
        assert flag not in lm


def test_preview_uses_engine_binary():
    assert preview_command(ServerConfig(model="/m")).startswith("mlx_lm.server")
    assert preview_command(ServerConfig(engine="mlx-vlm", model="/m")).startswith("mlx_vlm.server")
