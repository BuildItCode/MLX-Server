"""Search HuggingFace and download a model — opened from the server editor's "Search HF"
button. Filters by GGUF or MLX (MLX only on Apple Silicon), shows a size-vs-device-memory
fit badge per result, and downloads the chosen model into the HuggingFace cache with a live
progress log. Dismisses with an `HFResult(repo_id, fmt)` the editor uses to fill the model
field + set the matching engine.

GGUF repos usually hold many quant files, so picking one opens a second step listing each
file with its exact size; MLX repos download whole."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from .. import hf
from ..widgets.toggle_chip import ToggleChip

_FIT_COLOR = {"fits": "#7fb069", "tight": "#d19a66", "too_big": "#e06c75", "unknown": "#888888"}
_FIT_WORD = {"fits": "fits", "tight": "tight", "too_big": "too big", "unknown": ""}


@dataclass(frozen=True)
class HFResult:
    repo_id: str
    fmt: str  # "gguf" | "mlx"


def _k(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


class HFBrowseScreen(Screen):
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, *, allow_mlx: bool, current_engine: str = "mlx-lm") -> None:
        super().__init__()
        self._allow_mlx = allow_mlx
        # default format: GGUF on non-Mac or when editing a llama-cpp profile, else MLX
        self._fmt = "gguf" if (not allow_mlx or current_engine == "llama-cpp") else "mlx"
        self._budget: Optional[int] = None
        self._last_query = ""
        self._downloading = False
        self._cancel = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="hf-body"):
            with Horizontal(id="hf-controls"):
                yield Input(placeholder="Search HuggingFace models…", id="hf-query")
                yield Button("Search", id="hf-go", variant="primary")
                yield ToggleChip("GGUF", "gguf", id="fmt-gguf")
                yield ToggleChip("MLX", "mlx", id="fmt-mlx")
            yield Static("", id="hf-budget")
            yield Static("Type a name and Search — e.g. “qwen3”, “llama 8b”, “gemma”.", id="hf-status")
            yield VerticalScroll(id="hf-results")
            yield RichLog(id="hf-log", markup=False, wrap=True, max_lines=4000)
        with Horizontal(id="hf-buttons"):
            yield Button("← Back to results", id="hf-back")
            yield Button("Close", id="hf-cancel")
        yield Footer()

    def on_mount(self) -> None:
        # device-memory budget header
        ram = hf.total_ram_bytes()
        self._budget = hf.recommended_budget_bytes()
        if ram and self._budget:
            unified = " (unified memory — shared with the GPU)" if hf.is_apple_silicon() else ""
            self.query_one("#hf-budget", Static).update(
                Content.assemble(
                    ("Device memory: ", "dim"), (hf.human(ram), "bold"),
                    ("  ·  model budget ≈ ", "dim"), (hf.human(self._budget), "bold"),
                    (unified, "dim"),
                ))
        else:
            self.query_one("#hf-budget", Static).update(
                Content(("Device memory unknown — size recommendations are off.")))
        # format chips: select the default; lock MLX off-Mac
        self.query_one("#fmt-gguf", ToggleChip).set_value(self._fmt == "gguf")
        mlx_chip = self.query_one("#fmt-mlx", ToggleChip)
        mlx_chip.set_enabled(self._allow_mlx)
        mlx_chip.set_value(self._fmt == "mlx")
        if not self._allow_mlx:
            mlx_chip.tooltip = "MLX runs only on Apple Silicon"
        self.query_one("#hf-log", RichLog).display = False
        self.query_one("#hf-back", Button).display = False
        self.query_one("#hf-query", Input).focus()

    # --- format toggle ---------------------------------------------------

    @on(ToggleChip.Changed)
    def _fmt_changed(self, event: ToggleChip.Changed) -> None:
        gguf, mlx = self.query_one("#fmt-gguf", ToggleChip), self.query_one("#fmt-mlx", ToggleChip)
        if not event.value:
            # clicking the active chip off → keep it on (exactly one format is always selected)
            self.query_one(f"#fmt-{event.key}", ToggleChip).set_value(True)
            return
        self._fmt = event.key
        (mlx if event.key == "gguf" else gguf).set_value(False)  # mutually exclusive
        if self._last_query:
            self._run_search(self._last_query)

    # --- search ----------------------------------------------------------

    @on(Button.Pressed, "#hf-go")
    @on(Input.Submitted, "#hf-query")
    def _search_pressed(self) -> None:
        self._run_search(self.query_one("#hf-query", Input).value.strip())

    def _run_search(self, query: str) -> None:
        if self._downloading:
            return
        self._last_query = query
        self.run_worker(self._search_flow(query), group="hf-search", exclusive=True)

    async def _search_flow(self, query: str) -> None:
        self.query_one("#hf-back", Button).display = False
        results = self.query_one("#hf-results", VerticalScroll)
        await results.remove_children()
        self._status("Searching HuggingFace…")
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(hf.search_models, query, self._fmt, 30), timeout=20)
        except asyncio.TimeoutError:
            self._status("Search timed out — try again.")
            return
        except hf.HFError as exc:
            self._status(f"Couldn't search: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self._status(f"Search error: {exc}")
            return
        if not hits:
            self._status("No results — try a different name.")
            return
        self._status(f"{len(hits)} results for “{query or 'top downloads'}” · {self._fmt.upper()}")
        for h in hits:
            await results.mount(self._result_row(h))

    def _result_row(self, h: hf.ModelHit) -> Horizontal:
        name = Static(
            Content.assemble((h.repo_id, "bold"), (" · gated" if h.gated else "", "dim")),
            classes="hf-row-name")
        fit = Static(self._fit_content(h.size_bytes, h.size_exact), classes="hf-row-fit")
        meta = Static(Content.assemble((f"{_k(h.downloads)} dl", "dim")), classes="hf-row-meta")
        if h.fmt == "gguf":
            btn = Button("Choose ▸", classes="hf-choose")
        else:
            btn = Button("Download", variant="success", classes="hf-pick")
            btn._allow = None  # whole repo
        btn._repo_id = h.repo_id
        btn._fmt = h.fmt
        return Horizontal(name, fit, meta, btn, classes="hf-row")

    def _fit_content(self, size_bytes: Optional[int], size_exact: bool) -> Content:
        state = hf.fit(size_bytes, self._budget)
        color = _FIT_COLOR[state]
        text = (("" if size_exact else "~") + hf.human(size_bytes)) if size_bytes else "size ?"
        word = _FIT_WORD[state]
        return Content.assemble((text, color), (f"  {word}" if word else "", color))

    # --- GGUF quant picker ----------------------------------------------

    @on(Button.Pressed, ".hf-choose")
    def _choose(self, event: Button.Pressed) -> None:
        if self._downloading:
            return
        self.run_worker(self._choose_flow(getattr(event.button, "_repo_id", "")), group="hf-search", exclusive=True)

    async def _choose_flow(self, repo_id: str) -> None:
        results = self.query_one("#hf-results", VerticalScroll)
        await results.remove_children()
        self._status(f"Loading {repo_id} files…")
        try:
            files, mmproj = await asyncio.to_thread(hf.list_repo_gguf_files, repo_id)
        except hf.HFError as exc:
            self._status(f"Couldn't list files: {exc}")
            return
        if not files:
            self._status("No .gguf files found in that repo.")
            return
        extra = [mmproj] if mmproj else []
        if len(files) == 1:  # one quant → straight to download
            self._download(repo_id, "gguf", allow_patterns=[files[0].filename, *extra])
            return
        self.query_one("#hf-back", Button).display = True
        self._status(f"{repo_id} · pick a quant ({len(files)} files)" + (" · +mmproj vision" if mmproj else ""))
        for f in files:
            await results.mount(self._gguf_file_row(repo_id, f, extra))

    def _gguf_file_row(self, repo_id: str, f: hf.GgufFile, extra: list[str]) -> Horizontal:
        name = Static(Content((f.filename)), classes="hf-row-name")
        fit = Static(self._fit_content(f.size_bytes, True), classes="hf-row-fit")
        btn = Button("Download", variant="success", classes="hf-pick")
        btn._repo_id = repo_id
        btn._fmt = "gguf"
        btn._allow = [f.filename, *extra]
        return Horizontal(name, fit, btn, classes="hf-row")

    @on(Button.Pressed, "#hf-back")
    def _back(self) -> None:
        if not self._downloading:
            self._run_search(self._last_query)

    # --- download --------------------------------------------------------

    @on(Button.Pressed, ".hf-pick")
    def _pick(self, event: Button.Pressed) -> None:
        if self._downloading:
            return
        btn = event.button
        self._download(getattr(btn, "_repo_id", ""), getattr(btn, "_fmt", self._fmt),
                       allow_patterns=getattr(btn, "_allow", None))

    def _download(self, repo_id: str, fmt: str, *, allow_patterns: Optional[list[str]]) -> None:
        if not repo_id or self._downloading:
            return
        self.run_worker(self._download_flow(repo_id, fmt, allow_patterns), group="hf-dl", exclusive=True)

    async def _download_flow(self, repo_id: str, fmt: str, allow_patterns: Optional[list[str]]) -> None:
        self._downloading = True
        self._cancel = False
        self._set_busy(True)
        log = self.query_one("#hf-log", RichLog)
        log.display = True
        log.clear()
        try:
            size = await asyncio.to_thread(hf.exact_to_fetch_bytes, repo_id, allow_patterns=allow_patterns)
            if size is not None:
                log.write(f"Will fetch {hf.human(size)} (already-cached files skipped).")
                if hf.fit(size, self._budget) == "too_big":
                    log.write("⚠ Larger than the recommended memory budget — it may run slowly or fail to load.")
        except hf.HFError:
            pass  # dry-run is best-effort; the download itself reports real errors

        # download_model runs in a worker thread; marshal its progress lines back onto the
        # UI thread. call_from_thread lives on App, not Screen.
        def on_progress(line: str) -> None:
            try:
                self.app.call_from_thread(log.write, line)
            except Exception:  # noqa: BLE001 — the screen may have been dismissed mid-download
                pass

        try:
            await asyncio.to_thread(
                hf.download_model, repo_id, on_progress,
                allow_patterns=allow_patterns, cancel=lambda: self._cancel,
            )
        except hf.HFCancelled:
            self.notify("Download cancelled", severity="warning")
            self._downloading = False
            self._set_busy(False)
            return
        except hf.HFError as exc:
            log.write(f"✗ {exc}")
            self.notify(str(exc), severity="error", timeout=10)
            self._downloading = False
            self._set_busy(False)
            return
        except Exception as exc:  # noqa: BLE001
            log.write(f"✗ {exc}")
            self.notify(f"Download failed: {exc}", severity="error", timeout=10)
            self._downloading = False
            self._set_busy(False)
            return
        self._downloading = False
        self.notify(f"Downloaded {repo_id}")
        self.dismiss(HFResult(repo_id, fmt))

    def _set_busy(self, busy: bool) -> None:
        for wid in ("#hf-query", "#hf-go", "#hf-results", "#hf-back"):
            try:
                self.query_one(wid).disabled = busy
            except Exception:  # noqa: BLE001
                pass
        for chip in self.query(ToggleChip):
            chip.set_enabled(not busy and (chip.id != "fmt-mlx" or self._allow_mlx))
        # keep the format selection reflected after re-enabling
        if not busy:
            self.query_one("#fmt-gguf", ToggleChip).set_value(self._fmt == "gguf")
            self.query_one("#fmt-mlx", ToggleChip).set_value(self._fmt == "mlx")

    # --- close -----------------------------------------------------------

    @on(Button.Pressed, "#hf-cancel")
    def _cancel_btn(self) -> None:
        self._close_or_cancel()

    def action_close(self) -> None:
        self._close_or_cancel()

    def _close_or_cancel(self) -> None:
        if self._downloading and not self._cancel:
            # first press: ask the download to stop cooperatively
            self._cancel = True
            self.query_one("#hf-log", RichLog).write(
                "Cancelling… (press Close again to leave now; a partial download resumes next time)")
            return
        # not downloading, or a second press while a stalled transfer hasn't observed the cancel
        self.dismiss(None)

    def _status(self, text: str) -> None:
        self.query_one("#hf-status", Static).update(Content((text)))
