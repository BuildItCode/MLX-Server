"""Create/edit a server profile — a deliberately small form.

Everyday use needs only the essentials (engine, name, model, host/port) plus the
knobs people actually tune: KV-cache quantization, a draft model, and free-form
custom args. Fields the selected engine can't use are *disabled* in place, so the
form is one screen with no tabs. Everything else (sampling, prompt cache, parsers,
…) lives in a collapsed "Manual overrides" section and falls back to the server's
own defaults when left blank — so a profile auto-adjusts to the engine + KV choices
unless you deliberately open it up and override."""

from __future__ import annotations

import json
import os
import shlex
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Select,
    Switch,
)
from textual.widgets.option_list import Option

from .. import hf
from ..config import flags, store
from ..config.models import ServerConfig
from ..server import discovery
from ..widgets.path_input import DropPathInput, PathInput, path_hint, resolve_path, sanitize_drag

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_ENGINE_HINTS = {
    "mlx-lm": "Text LLMs via mlx_lm.server.",
    "mlx-vlm": "Vision-language (and text) models via mlx_vlm.server — supports a quantized KV cache.",
    "vllm-mlx": "vLLM-style MLX server (vllm-mlx serve): continuous batching, prefix cache, native tools, 4/8-bit KV cache.",
    "llama-cpp": "GGUF via llama.cpp (llama-server) — a .gguf file, its model folder (LM Studio dir), or an HF repo (org/repo:quant). A sibling vision projector (mmproj) loads automatically.",
}

# Per-engine guidance for the KV-cache row.
_KV_HINTS = {
    "mlx-lm": "mlx-lm has no KV-cache quantization — switch to mlx-vlm or vllm-mlx to shrink context memory.",
    "mlx-vlm": "KV bits: 8, 4, or 3.5. Turboquant available. Max KV size caps the context (tokens).",
    "vllm-mlx": "KV bits: 4 or 8. Max KV size caps the context (tokens). Turboquant is mlx-vlm only.",
    "llama-cpp": "llama.cpp uses --cache-type-k/v (f16, q8_0, q4_0, …) below, not KV bits. Context via the -c field.",
}

# Manual-section groups → the engines that use them; the rest are hidden so a
# profile only exposes flags the chosen server actually accepts. (Emission is
# gated in flags.py too; this just cuts UI clutter.)
_MANUAL_GROUP_ENGINES: dict[str, set[str]] = {
    "grp-sampling": {"mlx-lm"},                  # server-level sampling defaults
    "grp-shared-adv": {"mlx-lm", "mlx-vlm"},     # adapter / prefill / log level
    "grp-mlxlm-adv": {"mlx-lm"},                 # prompt cache, templates, concurrency
    "grp-kv-extra": {"mlx-vlm", "vllm-mlx"},     # KV group size
    "grp-kv-mlxvlm": {"mlx-vlm"},                # mlx-vlm-only KV start index
    "grp-vllm": {"vllm-mlx"},                    # continuous batching, parsers
    "grp-llamacpp": {"llama-cpp"},               # GPU layers, threads, ctx, KV cache types
}

# Format → engine for the "Search HF" flow: a downloaded model is paired with the engine
# that runs its format (GGUF → llama-cpp; MLX → an MLX engine).
_MLX_ENGINES = {"mlx-lm", "mlx-vlm", "vllm-mlx"}


class EditorScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, server: Optional[ServerConfig] = None) -> None:
        super().__init__()
        self.server = server
        # model-field autocomplete of already-downloaded HF models (lazy: scanned on first focus)
        self._cached: Optional[list[hf.LocalModel]] = None
        self._loading_cached = False
        self._suggest_items: list[hf.LocalModel] = []
        self._suppress_suggest = False  # set while we fill the field ourselves, so it doesn't reopen

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            # --- essentials --------------------------------------------------
            yield Label("Engine")
            yield Select(
                [
                    ("mlx-lm  ·  text LLMs", "mlx-lm"),
                    ("mlx-vlm  ·  vision-language models", "mlx-vlm"),
                    ("vllm-mlx  ·  vLLM-style (text + vision, KV-quant)", "vllm-mlx"),
                    ("llama-cpp  ·  GGUF models (llama.cpp)", "llama-cpp"),
                ],
                value="mlx-lm",
                allow_blank=False,
                id="engine",
            )
            yield Label("", id="engine-hint", classes="hint")
            yield Label("Name")
            yield DropPathInput(id="name", placeholder="My model server")
            yield Label("Model — drag a folder, paste a HuggingFace repo id, or search HF")
            with Horizontal(id="model-row"):
                yield PathInput(id="model", placeholder="/path/to/model  or  mlx-community/Qwen2.5-7B-4bit")
                yield Button("Search HF", id="hf-search")
            # dropdown of already-downloaded HF models — appears when the field is focused
            yield OptionList(id="model-suggest")
            yield Label("", id="model-hint", classes="hint")
            with Horizontal(classes="row"):
                with Vertical(classes="col"):
                    yield Label("Host")
                    yield DropPathInput(id="host", value="127.0.0.1")
                with Vertical(classes="col"):
                    yield Label("Port")
                    yield DropPathInput(id="port", value="8080")

            # --- KV cache & options (engine-gated, disabled in place) --------
            yield Label("KV cache & options", classes="section")
            yield Label("", id="kv-hint", classes="hint")
            with Horizontal(classes="row"):
                with Vertical(classes="col"):
                    yield Label("KV bits")
                    yield Input(id="kv_bits", placeholder="off · 8, 4, (3.5 mlx-vlm)")
                with Vertical(classes="col"):
                    yield Label("Max KV size (tokens)")
                    yield Input(id="max_kv_size", placeholder="off · e.g. 8192")
            with Horizontal(classes="switch-row"):
                yield Switch(id="turboquant")
                yield Label("turboquant  ·  mlx-vlm KV scheme")
            yield Label("Draft model — speculative decoding (optional)")
            yield Input(id="draft_model", placeholder="/path/to/smaller-model  or  repo id")
            yield Label("Custom args — appended to the command verbatim")
            yield Input(id="custom_params", placeholder="--enable-thinking")
            yield Label("", id="cmd-preview", classes="preview")

            # --- manual overrides (collapsed; auto/defaults until opened) -----
            with Collapsible(title="Manual overrides — advanced (auto by default)",
                             collapsed=True, id="manual"):
                yield Label("Each falls back to the server's own default when left blank — "
                            "only set what you need.", classes="hint")
                with Horizontal(classes="row"):
                    with Vertical(classes="col"):
                        yield Label("Max tokens")
                        yield Input(id="max_tokens", placeholder="server default")
                    with Vertical(classes="col"):
                        yield Label("Server binary path (override)")
                        yield Input(id="mlx_server_path", placeholder="blank = use PATH")
                with Horizontal(classes="switch-row"):
                    yield Switch(id="trust_remote_code")
                    yield Label("trust remote code")

                with Vertical(id="grp-sampling"):
                    yield Label("Sampling — mlx-lm", classes="hint")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Temperature")
                            yield Input(id="temp", placeholder="0.0")
                        with Vertical(classes="col"):
                            yield Label("top-p")
                            yield Input(id="top_p", placeholder="1.0")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("top-k")
                            yield Input(id="top_k", placeholder="0")
                        with Vertical(classes="col"):
                            yield Label("min-p")
                            yield Input(id="min_p", placeholder="0.0")

                with Vertical(id="grp-shared-adv"):
                    yield Label("Adapter / prefill / logging", classes="hint")
                    yield Label("Adapter path (LoRA)")
                    yield Input(id="adapter_path")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("prefill step size")
                            yield Input(id="prefill_step_size", placeholder="2048")
                        with Vertical(classes="col"):
                            yield Label("Log level")
                            yield Select([(lvl, lvl) for lvl in _LOG_LEVELS], value="INFO",
                                         allow_blank=False, id="log_level")

                with Vertical(id="grp-mlxlm-adv"):
                    yield Label("mlx-lm extras", classes="hint")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Prompt cache size")
                            yield Input(id="prompt_cache_size", placeholder="10")
                        with Vertical(classes="col"):
                            yield Label("Prompt cache bytes")
                            yield Input(id="prompt_cache_bytes", placeholder="e.g. 2GB")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Num draft tokens")
                            yield Input(id="num_draft_tokens", placeholder="3")
                        with Vertical(classes="col"):
                            yield Label("Allowed origins (CORS)")
                            yield Input(id="allowed_origins", placeholder="*")
                    yield Label("Chat template")
                    yield Input(id="chat_template")
                    yield Label("Chat template args (JSON)")
                    yield Input(id="chat_template_args", placeholder='{"enable_thinking": false}')
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("decode concurrency")
                            yield Input(id="decode_concurrency")
                        with Vertical(classes="col"):
                            yield Label("prompt concurrency")
                            yield Input(id="prompt_concurrency")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="use_default_chat_template")
                        yield Label("use default chat template")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="pipeline")
                        yield Label("pipeline (multi-device)")

                with Vertical(id="grp-kv-extra"):
                    yield Label("KV cache extras", classes="hint")
                    yield Label("KV group size")
                    yield Input(id="kv_group_size", placeholder="64")

                with Vertical(id="grp-kv-mlxvlm"):
                    yield Label("Quantized KV start — token index (mlx-vlm)")
                    yield Input(id="quantized_kv_start", placeholder="e.g. 0")

                with Vertical(id="grp-vllm"):
                    yield Label("vllm-mlx options", classes="hint")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="continuous_batching", value=True)
                        yield Label("continuous batching")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Tool-call parser")
                            yield Input(id="tool_call_parser", placeholder="auto")
                        with Vertical(classes="col"):
                            yield Label("Reasoning parser")
                            yield Input(id="reasoning_parser", placeholder="gpt_oss | harmony | qwen3")

                with Vertical(id="grp-llamacpp"):
                    yield Label("llama.cpp options", classes="hint")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("GPU layers (-ngl)")
                            yield Input(id="n_gpu_layers", placeholder="all · e.g. 99")
                        with Vertical(classes="col"):
                            yield Label("Threads (-t)")
                            yield Input(id="n_threads", placeholder="auto")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Context size (-c)")
                            yield Input(id="ctx", placeholder="from model")
                        with Vertical(classes="col"):
                            yield Label("Parallel slots")
                            yield Input(id="parallel", placeholder="1")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("KV cache type K")
                            yield Input(id="cache_type_k", placeholder="f16 · q8_0 · q4_0")
                        with Vertical(classes="col"):
                            yield Label("KV cache type V")
                            yield Input(id="cache_type_v", placeholder="f16 · q8_0 · q4_0")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="flash_attn")
                        yield Label("flash attention")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="jinja")
                        yield Label("Jinja chat template (--jinja)")
        with Horizontal(id="buttons"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Save & Launch", id="save_launch", variant="success")
            yield Button("Cancel", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        if self.server is not None:
            self._populate(self.server)
        self._refresh_engine_hint()
        self._apply_engine_gating()
        self._update_preview()
        # the suggestion list is mouse-clickable but never takes keyboard focus — the model
        # field keeps focus and drives it (arrows/enter), so there's no focus-juggling.
        self.query_one("#model-suggest", OptionList).can_focus = False
        self.query_one("#name", Input).focus()

    def _refresh_engine_hint(self) -> None:
        engine = self.query_one("#engine", Select).value
        self.query_one("#engine-hint", Label).update(_ENGINE_HINTS.get(str(engine), ""))

    def _apply_engine_gating(self) -> None:
        """Disable the KV/option fields the engine can't use, hide manual groups
        that don't apply, and update the KV hint."""
        engine = str(self.query_one("#engine", Select).value)
        # visible KV/options — disabled in place (still shown, so it's clear they exist)
        self.query_one("#kv_bits", Input).disabled = engine in ("mlx-lm", "llama-cpp")
        self.query_one("#max_kv_size", Input).disabled = engine in ("mlx-lm", "llama-cpp")
        self.query_one("#turboquant", Switch).disabled = engine != "mlx-vlm"
        self.query_one("#draft_model", Input).disabled = engine in ("vllm-mlx", "llama-cpp")
        self.query_one("#kv-hint", Label).update(_KV_HINTS.get(engine, ""))
        # manual groups — hidden when irrelevant to the engine
        for gid, engines in _MANUAL_GROUP_ENGINES.items():
            self.query_one(f"#{gid}").display = engine in engines

    # --- populate / collect ---------------------------------------------

    def _populate(self, s: ServerConfig) -> None:
        def text(i: str, val) -> None:
            self.query_one(f"#{i}", Input).value = "" if val is None else str(val)

        text("name", s.name)
        text("model", s.model)
        text("host", s.host)
        text("port", s.port)
        text("temp", s.temp)
        text("max_tokens", s.max_tokens)
        text("top_p", s.top_p)
        text("top_k", s.top_k)
        text("min_p", s.min_p)
        text("adapter_path", s.adapter_path)
        text("prompt_cache_size", s.prompt_cache_size)
        text("prompt_cache_bytes", s.prompt_cache_bytes)
        text("kv_bits", s.kv_bits)
        text("kv_group_size", s.kv_group_size)
        text("max_kv_size", s.max_kv_size)
        text("quantized_kv_start", s.quantized_kv_start)
        text("tool_call_parser", s.tool_call_parser)
        text("reasoning_parser", s.reasoning_parser)
        text("draft_model", s.draft_model)
        text("num_draft_tokens", s.num_draft_tokens)
        text("allowed_origins", s.allowed_origins)
        text("chat_template", s.chat_template)
        text("chat_template_args", s.chat_template_args)
        text("decode_concurrency", s.decode_concurrency)
        text("prompt_concurrency", s.prompt_concurrency)
        text("prefill_step_size", s.prefill_step_size)
        text("mlx_server_path", s.mlx_server_path)
        text("custom_params", s.custom_params)
        text("ctx", s.ctx)
        text("n_gpu_layers", s.n_gpu_layers)
        text("n_threads", s.n_threads)
        text("parallel", s.parallel)
        text("cache_type_k", s.cache_type_k)
        text("cache_type_v", s.cache_type_v)
        self.query_one("#engine", Select).value = s.engine
        self.query_one("#log_level", Select).value = s.log_level
        self.query_one("#turboquant", Switch).value = s.kv_quant_scheme == "turboquant"
        self.query_one("#trust_remote_code", Switch).value = s.trust_remote_code
        self.query_one("#use_default_chat_template", Switch).value = s.use_default_chat_template
        self.query_one("#pipeline", Switch).value = s.pipeline
        self.query_one("#continuous_batching", Switch).value = s.continuous_batching
        self.query_one("#flash_attn", Switch).value = s.flash_attn
        self.query_one("#jinja", Switch).value = s.jinja

    def _collect(self) -> ServerConfig:
        def s(i: str) -> str:
            return self.query_one(f"#{i}", Input).value.strip()

        def opt_str(i: str) -> Optional[str]:
            return s(i) or None

        def opt_int(i: str) -> Optional[int]:
            v = s(i)
            if not v:
                return None
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"'{i.replace('_', ' ')}' must be a whole number")

        def opt_float(i: str) -> Optional[float]:
            v = s(i)
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"'{i.replace('_', ' ')}' must be a number")

        def sw(i: str) -> bool:
            return self.query_one(f"#{i}", Switch).value

        model = self.query_one("#model", PathInput).resolved()
        name = s("name") or os.path.basename(model.rstrip("/")) or "Untitled server"
        port = opt_int("port") or 8080
        if not (1 <= port <= 65535):
            raise ValueError("port must be between 1 and 65535")
        cta = opt_str("chat_template_args")
        if cta:
            try:
                json.loads(cta)
            except json.JSONDecodeError:
                raise ValueError("chat template args must be valid JSON")
        engine = self.query_one("#engine", Select).value
        kv_bits = opt_str("kv_bits")
        if kv_bits is not None:
            try:
                float(kv_bits)
            except ValueError:
                raise ValueError("KV bits must be a number (e.g. 8, 4, or 3.5)")
            if engine == "vllm-mlx" and kv_bits not in ("4", "8"):
                raise ValueError("vllm-mlx KV bits must be 4 or 8")
        # turboquant toggle drives the scheme; only meaningful with KV quant on (mlx-vlm)
        kv_scheme = "turboquant" if (sw("turboquant") and kv_bits) else None
        custom = s("custom_params")
        if custom:
            try:
                shlex.split(custom)  # the launch + preview split this; reject unbalanced quotes here
            except ValueError as exc:
                raise ValueError(f"Custom params: {exc} — check the quotes")

        kwargs = dict(
            name=name,
            engine=engine,
            model=model,
            host=s("host") or "127.0.0.1",
            port=port,
            temp=opt_float("temp"),
            max_tokens=opt_int("max_tokens"),
            top_p=opt_float("top_p"),
            top_k=opt_int("top_k"),
            min_p=opt_float("min_p"),
            adapter_path=opt_str("adapter_path"),
            prompt_cache_size=opt_int("prompt_cache_size"),
            prompt_cache_bytes=opt_str("prompt_cache_bytes"),
            kv_bits=kv_bits,
            kv_quant_scheme=kv_scheme,
            kv_group_size=opt_int("kv_group_size"),
            max_kv_size=opt_int("max_kv_size"),
            quantized_kv_start=opt_int("quantized_kv_start"),
            continuous_batching=sw("continuous_batching"),
            tool_call_parser=opt_str("tool_call_parser"),
            reasoning_parser=opt_str("reasoning_parser"),
            ctx=opt_int("ctx"),
            n_gpu_layers=opt_int("n_gpu_layers"),
            n_threads=opt_int("n_threads"),
            parallel=opt_int("parallel"),
            cache_type_k=opt_str("cache_type_k"),
            cache_type_v=opt_str("cache_type_v"),
            flash_attn=sw("flash_attn"),
            jinja=sw("jinja"),
            log_level=self.query_one("#log_level", Select).value,
            draft_model=opt_str("draft_model"),
            num_draft_tokens=opt_int("num_draft_tokens"),
            allowed_origins=opt_str("allowed_origins"),
            chat_template=opt_str("chat_template"),
            chat_template_args=cta,
            decode_concurrency=opt_int("decode_concurrency"),
            prompt_concurrency=opt_int("prompt_concurrency"),
            prefill_step_size=opt_int("prefill_step_size"),
            trust_remote_code=sw("trust_remote_code"),
            use_default_chat_template=sw("use_default_chat_template"),
            pipeline=sw("pipeline"),
            mlx_server_path=opt_str("mlx_server_path"),
            custom_params=custom,
        )
        if self.server is not None:
            kwargs["id"] = self.server.id
        return ServerConfig(**kwargs)

    def _save(self) -> Optional[ServerConfig]:
        try:
            cfg = self._collect()
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return None
        store.upsert_server(self.app.config, cfg)
        self.app.config.settings.last_used_id = cfg.id
        self.app.save_config()
        return cfg

    # --- events ----------------------------------------------------------

    @on(DropPathInput.PathDropped)
    def _path_dropped(self, event: DropPathInput.PathDropped) -> None:
        # a folder/file dropped on any drop-aware field → set the model path
        self._set_model_path(event.path)

    def on_paste(self, event: events.Paste) -> None:
        # fallback: a path dropped while focus is on a tab/button/select/switch
        # (or nothing) bubbles up to the screen — route it to the model field too.
        text = event.text.splitlines()[0] if event.text else ""
        resolved = resolve_path(text)
        if resolved and os.path.isabs(resolved) and os.path.exists(resolved):
            event.stop()
            self._set_model_path(resolved)

    def _set_model_path(self, path: str) -> None:
        self._suppress_suggest = True  # we're filling the field — don't pop the dropdown back open
        self.query_one("#model", PathInput).value = path
        self.query_one("#model-hint", Label).update(path_hint(path))
        self._update_preview()
        self.notify(f"Model path set: {path}")

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model":
            cleaned = sanitize_drag(event.value)
            if cleaned != event.value:
                event.input.value = cleaned
                return  # re-fires with the cleaned value
            self.query_one("#model-hint", Label).update(path_hint(cleaned))
            if self._suppress_suggest:
                self._suppress_suggest = False  # consume: this change was our own fill
                self._hide_suggest()
            elif self.query_one("#model", PathInput).has_focus:
                self._show_suggest()  # the user is typing → filter the downloaded-model list
        self._update_preview()

    @on(Select.Changed)
    def _select_changed(self) -> None:
        self._refresh_engine_hint()
        self._apply_engine_gating()
        self._update_preview()

    @on(Switch.Changed)
    def _switch_changed(self) -> None:
        self._update_preview()

    def _update_preview(self) -> None:
        try:
            cfg = self._collect()
            if cfg.engine == "llama-cpp":  # show the resolved .gguf file, as the launch will use it
                cfg = cfg.model_copy(update={"model": discovery.resolve_gguf(cfg.model)})
            preview = flags.preview_command(cfg)  # inside the try: shlex.split can raise on bad quotes
        except Exception:
            return
        self.query_one("#cmd-preview", Label).update(Content(f"$ {preview}"))

    @on(Button.Pressed, "#save")
    def _on_save(self) -> None:
        if self._save() is not None:
            self.notify("Saved")
            self.app.pop_screen()

    @on(Button.Pressed, "#save_launch")
    def _on_save_launch(self) -> None:
        cfg = self._save()
        if cfg is not None:
            from .running import RunningScreen

            self.app.pop_screen()
            self.app.push_screen(RunningScreen(cfg))

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.app.pop_screen()

    def action_cancel(self) -> None:
        if self.query_one("#model-suggest", OptionList).display:
            self._hide_suggest()  # Escape closes the dropdown before it closes the editor
            return
        self.app.pop_screen()

    def action_save(self) -> None:
        if self._save() is not None:
            self.notify("Saved")
            self.app.pop_screen()

    # --- HuggingFace search / download -----------------------------------

    @on(Button.Pressed, "#hf-search")
    def _hf_search(self) -> None:
        self.run_worker(self._hf_search_flow(), exclusive=True)

    async def _hf_search_flow(self) -> None:
        from .hf_browse import HFBrowseScreen
        current = str(self.query_one("#engine", Select).value)
        result = await self.app.push_screen_wait(
            HFBrowseScreen(allow_mlx=hf.is_apple_silicon(), current_engine=current))
        if result is not None:
            self._apply_hf_result(result)

    def _engine_for_format(self, fmt: str, current: str) -> str:
        """GGUF → llama-cpp; MLX → keep the current MLX engine if one's selected, else mlx-lm."""
        if fmt == "gguf":
            return "llama-cpp"
        return current if current in _MLX_ENGINES else "mlx-lm"

    def _apply_hf_result(self, result) -> None:
        """A model picked in the HF browser: switch the engine to match its format, then fill
        the model field (the repo id — the engines resolve it from the HF cache on launch)."""
        sel = self.query_one("#engine", Select)
        engine = self._engine_for_format(result.fmt, str(sel.value))
        if str(sel.value) != engine:
            sel.value = engine  # fires Select.Changed → hint + gating + preview
        self._set_model_path(result.repo_id)
        self._apply_engine_gating()  # idempotent — correct even when the engine didn't change
        self._update_preview()

    # --- downloaded-model autocomplete (model field) ---------------------

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if getattr(event.widget, "id", None) == "model":
            self._show_suggest()  # focusing the model field opens the downloaded-models dropdown

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if getattr(event.widget, "id", None) == "model":
            self._hide_suggest()  # leaving the field closes it (the list never holds focus itself)

    def on_key(self, event: events.Key) -> None:
        # the field keeps focus, so route arrow keys to the (unfocusable) dropdown ourselves
        sug = self.query_one("#model-suggest", OptionList)
        if not sug.display or not self.query_one("#model", PathInput).has_focus:
            return
        if event.key == "down":
            sug.action_cursor_down()
            event.stop()
        elif event.key == "up":
            sug.action_cursor_up()
            event.stop()

    @on(Input.Submitted, "#model")
    def _model_submitted(self, event: Input.Submitted) -> None:
        # Enter while the dropdown is open picks the highlighted model instead of submitting
        sug = self.query_one("#model-suggest", OptionList)
        if sug.display and sug.highlighted is not None and 0 <= sug.highlighted < len(self._suggest_items):
            self._apply_hf_result(self._suggest_items[sug.highlighted])
            self._hide_suggest()
            event.stop()

    @on(OptionList.OptionSelected, "#model-suggest")
    def _suggest_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._suggest_items):
            self._apply_hf_result(self._suggest_items[event.option_index])  # fills field + sets engine
        self._hide_suggest()
        event.stop()

    def _looks_like_path(self, text: str) -> bool:
        """A filesystem path (not a repo id) — don't offer HF suggestions while one is typed."""
        t = text.strip()
        return t.startswith(("/", "~", ".")) or (len(t) >= 2 and t[1] == ":")  # POSIX/home or Windows drive

    def _show_suggest(self) -> None:
        sug = self.query_one("#model-suggest", OptionList)
        typed = self.query_one("#model", PathInput).value.strip()
        if self._looks_like_path(typed):
            sug.display = False
            return
        if self._cached is None:  # first use → scan the cache off-thread, then re-open if still focused
            self._ensure_cached_loading()
            sug.display = False
            return
        low = typed.lower()
        items = [m for m in self._cached if low in m.repo_id.lower()] if low else list(self._cached)
        self._suggest_items = items[:50]
        sug.clear_options()
        if not self._suggest_items:
            sug.display = False
            return
        sug.add_options([self._suggest_option(m) for m in self._suggest_items])
        sug.highlighted = 0
        sug.display = True

    def _hide_suggest(self) -> None:
        self.query_one("#model-suggest", OptionList).display = False

    def _suggest_option(self, m: hf.LocalModel) -> Option:
        return Option(Content.assemble(
            (m.repo_id, ""),
            (f"   {hf.human(m.size_bytes)} · {m.fmt.upper()}", "dim"),
        ))

    def _ensure_cached_loading(self) -> None:
        if self._loading_cached:
            return
        self._loading_cached = True
        self.run_worker(self._load_cached_models())

    async def _load_cached_models(self) -> None:
        import asyncio

        try:
            self._cached = await asyncio.to_thread(hf.cached_models)
        except Exception:  # noqa: BLE001 — never let a cache-scan failure break the editor
            self._cached = []
        finally:
            self._loading_cached = False
        if self.query_one("#model", PathInput).has_focus:
            self._show_suggest()  # the user is still on the field → open it now that we have the list
