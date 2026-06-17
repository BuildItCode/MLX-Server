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
    assert "--temp" not in args  # sampling is sent per request now, not as a launch flag
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


def test_vllm_mlx_uses_subcommand_positional_model_and_own_flags():
    # vllm-mlx is `vllm-mlx serve <model> …` — a subcommand with the model as a
    # POSITIONAL arg (not --model) and its own quantization/parser flag names.
    c = ServerConfig(
        engine="vllm-mlx", model="/models/gpt-oss", port=8080, max_tokens=16384,
        max_kv_size=32768, kv_bits="4", kv_group_size=64, reasoning_parser="gpt_oss",
        tool_call_parser="auto", continuous_batching=True, trust_remote_code=True,
    )
    args = build_args(c)
    assert args[0] == "serve" and args[1] == "/models/gpt-oss"  # subcommand + positional model
    assert "--model" not in args                                # NOT the mlx --model style
    assert args[args.index("--port") + 1] == "8080"
    assert args[args.index("--max-tokens") + 1] == "16384"
    assert args[args.index("--max-kv-size") + 1] == "32768"
    # quantized context: enable flag + integer bit width + group size
    assert "--kv-cache-quantization" in args
    assert args[args.index("--kv-cache-quantization-bits") + 1] == "4"
    assert args[args.index("--kv-cache-quantization-group-size") + 1] == "64"
    assert "--continuous-batching" in args
    # native tools on by default (vllm-mlx has gpt-oss/harmony parsers)
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "auto"
    assert args[args.index("--reasoning-parser") + 1] == "gpt_oss"
    assert "--trust-remote-code" in args
    # "4.0" normalizes to integer "4" for vllm-mlx's {4,8} choice
    assert build_args(ServerConfig(engine="vllm-mlx", model="/m", kv_bits="4.0"))[
        build_args(ServerConfig(engine="vllm-mlx", model="/m", kv_bits="4.0")).index("--kv-cache-quantization-bits") + 1
    ] == "4"


def test_preview_uses_engine_binary():
    assert preview_command(ServerConfig(model="/m")).startswith("mlx_lm.server")
    assert preview_command(ServerConfig(engine="mlx-vlm", model="/m")).startswith("mlx_vlm.server")
    assert preview_command(ServerConfig(engine="vllm-mlx", model="/m")).startswith("vllm-mlx serve /m")
    assert preview_command(ServerConfig(engine="llama-cpp", model="/m.gguf")).startswith("llama-server -m /m.gguf")


def test_llama_cpp_builds_native_flags():
    c = ServerConfig(
        engine="llama-cpp", model="/models/m.gguf", port=8081, ctx=8192, n_gpu_layers=99,
        n_threads=8, max_tokens=4096, temp=0.7, cache_type_k="q8_0", cache_type_v="q8_0",
        parallel=2, flash_attn=True, jinja=True, continuous_batching=False,
    )
    a = build_args(c)
    assert a[a.index("-m") + 1] == "/models/m.gguf"
    assert a[a.index("--port") + 1] == "8081"
    assert a[a.index("-c") + 1] == "8192"
    assert a[a.index("-ngl") + 1] == "99"
    assert a[a.index("-t") + 1] == "8"
    assert a[a.index("--n-predict") + 1] == "4096"        # max_tokens → --n-predict
    assert a[a.index("--cache-type-k") + 1] == "q8_0"
    assert a[a.index("--flash-attn") + 1] == "on"          # a VALUE, not a bare flag
    assert "--no-cont-batching" in a                       # continuous_batching=False (default on)
    assert "--jinja" in a
    assert "--temp" not in a                               # sampling is sent per request, not at launch


def test_llama_cpp_hf_repo_vs_local_path():
    assert build_args(ServerConfig(engine="llama-cpp", model="org/Repo-GGUF:Q4_K_M"))[:2] == \
        ["-hf", "org/Repo-GGUF:Q4_K_M"]                     # HF repo → download
    assert build_args(ServerConfig(engine="llama-cpp", model="/models/m.gguf"))[:2] == \
        ["-m", "/models/m.gguf"]                            # absolute path → local file
    assert build_args(ServerConfig(engine="llama-cpp", model="./local.gguf"))[:2] == \
        ["-m", "./local.gguf"]                             # relative .gguf → local file


def test_llama_cpp_omits_mlx_only_flags():
    a = build_args(ServerConfig(engine="llama-cpp", model="/m.gguf", kv_bits="4",
                                pipeline=True, prompt_cache_size=10))
    assert "--kv-bits" not in a and "--pipeline" not in a and "--prompt-cache-size" not in a
    # continuous_batching defaults on → no flag emitted
    assert "--no-cont-batching" not in build_args(ServerConfig(engine="llama-cpp", model="/m.gguf"))
