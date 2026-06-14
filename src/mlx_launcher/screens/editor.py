"""Create/edit a server profile."""

from __future__ import annotations

import json
import os
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Switch,
    TabbedContent,
    TabPane,
)

from ..config import flags, store
from ..config.models import ServerConfig
from ..widgets.path_input import DropPathInput, PathInput, path_hint, resolve_path, sanitize_drag

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_ENGINE_HINTS = {
    "mlx-lm": "Text LLMs via mlx_lm.server. The sampling & cache fields below apply.",
    "mlx-vlm": (
        "Vision-language models via mlx_vlm.server. Sampling is per-request — put "
        "--kv-bits / --max-kv-size / --enable-thinking in the Custom tab."
    ),
}


class EditorScreen(Screen):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, server: Optional[ServerConfig] = None) -> None:
        super().__init__()
        self.server = server

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            with TabbedContent(initial="tab-basic"):
                with TabPane("Basic", id="tab-basic"):
                    yield Label("Engine")
                    yield Select(
                        [
                            ("mlx-lm  ·  text LLMs", "mlx-lm"),
                            ("mlx-vlm  ·  vision-language models", "mlx-vlm"),
                        ],
                        value="mlx-lm",
                        allow_blank=False,
                        id="engine",
                    )
                    yield Label("", id="engine-hint", classes="hint")
                    yield Label("Name")
                    yield DropPathInput(id="name", placeholder="My model server")
                    yield Label("Model — drag a folder onto the terminal, or paste a HuggingFace repo id")
                    yield PathInput(id="model", placeholder="/path/to/model  or  mlx-community/Qwen2.5-7B-4bit")
                    yield Label("", id="model-hint", classes="hint")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Host")
                            yield DropPathInput(id="host", value="127.0.0.1")
                        with Vertical(classes="col"):
                            yield Label("Port")
                            yield DropPathInput(id="port", value="8080")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Temperature")
                            yield DropPathInput(id="temp", placeholder="0.0")
                        with Vertical(classes="col"):
                            yield Label("Max tokens")
                            yield DropPathInput(id="max_tokens", placeholder="512")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("top-p")
                            yield DropPathInput(id="top_p", placeholder="1.0")
                        with Vertical(classes="col"):
                            yield Label("top-k")
                            yield DropPathInput(id="top_k", placeholder="0")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("min-p")
                            yield DropPathInput(id="min_p", placeholder="0.0")
                        with Vertical(classes="col"):
                            yield Label("")
                with TabPane("Advanced", id="tab-adv"):
                    yield Label("Adapter path (LoRA)")
                    yield Input(id="adapter_path")
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Prompt cache size")
                            yield Input(id="prompt_cache_size", placeholder="10")
                        with Vertical(classes="col"):
                            yield Label("Prompt cache bytes")
                            yield Input(id="prompt_cache_bytes", placeholder="e.g. 2GB")
                    yield Label("Log level")
                    yield Select(
                        [(lvl, lvl) for lvl in _LOG_LEVELS],
                        value="INFO",
                        allow_blank=False,
                        id="log_level",
                    )
                    with Horizontal(classes="row"):
                        with Vertical(classes="col"):
                            yield Label("Draft model (speculative)")
                            yield Input(id="draft_model")
                        with Vertical(classes="col"):
                            yield Label("Num draft tokens")
                            yield Input(id="num_draft_tokens", placeholder="3")
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
                    yield Label("prefill step size")
                    yield Input(id="prefill_step_size", placeholder="2048")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="trust_remote_code")
                        yield Label("trust remote code")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="use_default_chat_template")
                        yield Label("use default chat template")
                    with Horizontal(classes="switch-row"):
                        yield Switch(id="pipeline")
                        yield Label("pipeline (multi-device)")
                    yield Label("mlx_lm.server path (optional override)")
                    yield Input(id="mlx_server_path", placeholder="leave blank to use PATH")
                with TabPane("Custom", id="tab-custom"):
                    yield Label("Custom params — appended to the command line verbatim.")
                    yield Label(
                        "Quantized KV cache / context size live here, e.g.  --kv-bits 4 --max-kv-size 8192",
                        classes="hint",
                    )
                    yield Input(id="custom_params", placeholder="--kv-bits 4 --max-kv-size 8192")
                    yield Label("", id="cmd-preview", classes="preview")
        with Horizontal(id="buttons"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Save & Launch", id="save_launch", variant="success")
            yield Button("Cancel", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        if self.server is not None:
            self._populate(self.server)
        self._refresh_engine_hint()
        self._update_preview()
        self.query_one("#name", Input).focus()

    def _refresh_engine_hint(self) -> None:
        engine = self.query_one("#engine", Select).value
        self.query_one("#engine-hint", Label).update(_ENGINE_HINTS.get(str(engine), ""))

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
        self.query_one("#engine", Select).value = s.engine
        self.query_one("#log_level", Select).value = s.log_level
        self.query_one("#trust_remote_code", Switch).value = s.trust_remote_code
        self.query_one("#use_default_chat_template", Switch).value = s.use_default_chat_template
        self.query_one("#pipeline", Switch).value = s.pipeline

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

        kwargs = dict(
            name=name,
            engine=self.query_one("#engine", Select).value,
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
            custom_params=s("custom_params"),
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
        self._update_preview()

    @on(Select.Changed)
    def _select_changed(self) -> None:
        self._refresh_engine_hint()
        self._update_preview()

    @on(Switch.Changed)
    def _switch_changed(self) -> None:
        self._update_preview()

    def _update_preview(self) -> None:
        try:
            cfg = self._collect()
        except Exception:
            return
        self.query_one("#cmd-preview", Label).update(f"$ {flags.preview_command(cfg)}")

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
        self.app.pop_screen()

    def action_save(self) -> None:
        if self._save() is not None:
            self.notify("Saved")
            self.app.pop_screen()
