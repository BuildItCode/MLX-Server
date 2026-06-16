"""HuggingFace search/download logic + the editor wiring + the browse screen. All hermetic:
huggingface_hub is patched via `hf._hf`, so nothing touches the network."""

from types import SimpleNamespace

import pytest

from mlx_launcher import hf


def _fake_hub(**attrs):
    attrs.setdefault("get_token", lambda: None)
    return SimpleNamespace(**attrs)


# --- size heuristic / fit / human ----------------------------------------

def test_size_heuristic():
    assert hf.estimate_size_from_name("mlx-community/whisper-base-mlx", "mlx") is None  # no param token
    assert hf.estimate_size_from_name("org/Model-4bit", "mlx") is None                 # quant ≠ params
    s = hf.estimate_size_from_name("bartowski/Llama-3-8B-Instruct-Q4_K_M-GGUF", "gguf")
    assert 4e9 < s < 6e9
    big = hf.estimate_size_from_name("TheBloke/Mixtral-8x7B-v0.1-GGUF", "gguf")
    assert big > 30e9
    mlx = hf.estimate_size_from_name("mlx-community/Qwen2.5-7B-4bit", "mlx")
    assert 3e9 < mlx < 5e9


def test_fit_and_budget(monkeypatch):
    monkeypatch.setattr(hf, "total_ram_bytes", lambda: 32 * 1024 ** 3)
    budget = hf.recommended_budget_bytes()
    assert budget == int(32 * 1024 ** 3 * 0.70)
    assert hf.fit(int(budget * 0.5), budget) == "fits"
    assert hf.fit(int(budget * 0.9), budget) == "tight"
    assert hf.fit(int(budget * 1.5), budget) == "too_big"
    assert hf.fit(None, budget) == "unknown"
    assert hf.fit(int(budget * 0.5), None) == "unknown"
    monkeypatch.setattr(hf, "total_ram_bytes", lambda: None)
    assert hf.recommended_budget_bytes() is None


def test_total_ram_is_positive_on_this_machine():
    ram = hf.total_ram_bytes()
    assert ram is None or ram > 1024 ** 3  # plausible (>1 GiB) or undetectable


def test_human():
    assert hf.human(None) == "?" and hf.human(0) == "0 B"
    assert hf.human(900) == "900 B"
    assert hf.human(812_000_000).endswith(" MB")
    assert hf.human(5_400_000_000).endswith(" GB")


# --- search ---------------------------------------------------------------

def test_search_models_maps_and_filters(monkeypatch):
    captured = {}

    class FakeApi:
        def list_models(self, **kw):
            captured.update(kw)
            return [
                SimpleNamespace(id="mlx-community/Qwen2.5-7B-4bit", downloads=1000, likes=5, tags=["mlx"], gated=False),
                SimpleNamespace(id="org/Private-13B", downloads=10, likes=0, tags=[], private=True),
            ]

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(HfApi=lambda: FakeApi()))
    hits = hf.search_models("qwen", "mlx", 30)
    assert captured["filter"] == "mlx" and captured["sort"] == "downloads" and captured["limit"] == 30
    assert hits[0].repo_id == "mlx-community/Qwen2.5-7B-4bit"
    assert hits[0].size_bytes and hits[0].size_exact is False  # heuristic (used_storage absent)
    assert hits[1].gated is True                               # private ⇒ gated marker


def test_search_models_wraps_errors(monkeypatch):
    class FakeApi:
        def list_models(self, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(HfApi=lambda: FakeApi()))
    with pytest.raises(hf.HFError):
        hf.search_models("x", "gguf")


# --- GGUF file listing + exact size --------------------------------------

def test_list_repo_gguf_files(monkeypatch):
    info = SimpleNamespace(siblings=[
        SimpleNamespace(rfilename="model.Q8_0.gguf", size=8_000_000_000),
        SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=4_000_000_000),
        SimpleNamespace(rfilename="mmproj-f16.gguf", size=500_000_000),
        SimpleNamespace(rfilename="README.md", size=1000),
    ])

    class FakeApi:
        def model_info(self, repo, files_metadata=False, token=None):
            assert files_metadata is True
            return info

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(HfApi=lambda: FakeApi()))
    files, mmproj = hf.list_repo_gguf_files("org/x")
    assert [f.filename for f in files] == ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]  # sorted by size, mmproj/README excluded
    assert mmproj == "mmproj-f16.gguf"


def test_exact_to_fetch_bytes_sums_will_download(monkeypatch):
    def fake_snap(repo, dry_run=False, allow_patterns=None, token=None, tqdm_class=None):
        assert dry_run is True
        return [
            SimpleNamespace(file_size=4_000_000_000, will_download=True),
            SimpleNamespace(file_size=1_000_000_000, will_download=False),  # already cached
        ]

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(snapshot_download=fake_snap))
    assert hf.exact_to_fetch_bytes("org/x") == 4_000_000_000


# --- download -------------------------------------------------------------

def test_download_model_streams_progress_and_returns_path(monkeypatch):
    lines = []

    def fake_snap(repo, allow_patterns=None, token=None, tqdm_class=None):
        bar = tqdm_class(total=100, unit="B")  # the aggregated bytes bar hf would create
        bar.update(40)
        bar.update(60)
        return "/cache/models--" + repo.replace("/", "--")

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(
        snapshot_download=fake_snap, constants=SimpleNamespace(HF_HUB_CACHE="/cache")))
    path = hf.download_model("org/x", lambda line: lines.append(line))
    assert path == "/cache/models--org--x"
    assert any("download complete" in line for line in lines)
    assert any("%" in line for line in lines)  # progress emitted from the tqdm hook


def test_download_model_cancel(monkeypatch):
    lines = []

    def fake_snap(repo, allow_patterns=None, token=None, tqdm_class=None):
        bar = tqdm_class(total=100, unit="B")
        bar.update(10)  # cancel() is True → update raises HFCancelled

    monkeypatch.setattr(hf, "_hf", lambda: _fake_hub(
        snapshot_download=fake_snap, constants=SimpleNamespace(HF_HUB_CACHE="/cache")))
    with pytest.raises(hf.HFCancelled):
        hf.download_model("org/x", lambda line: lines.append(line), cancel=lambda: True)
    assert any("Cancelled" in line for line in lines)


# --- editor wiring --------------------------------------------------------

def test_engine_for_format():
    from mlx_launcher.screens.editor import EditorScreen
    cs = EditorScreen.__new__(EditorScreen)
    assert cs._engine_for_format("gguf", "mlx-lm") == "llama-cpp"
    assert cs._engine_for_format("gguf", "vllm-mlx") == "llama-cpp"
    assert cs._engine_for_format("mlx", "mlx-vlm") == "mlx-vlm"    # keep current MLX engine
    assert cs._engine_for_format("mlx", "vllm-mlx") == "vllm-mlx"
    assert cs._engine_for_format("mlx", "llama-cpp") == "mlx-lm"   # not MLX → default


def test_apply_hf_result_sets_engine_and_model():
    from mlx_launcher.screens.editor import EditorScreen
    from mlx_launcher.screens.hf_browse import HFResult

    cs = EditorScreen.__new__(EditorScreen)
    sel = SimpleNamespace(value="mlx-lm")
    calls = {"model": None, "gating": 0, "preview": 0}
    cs.query_one = lambda *a, **k: sel
    cs._set_model_path = lambda p: calls.__setitem__("model", p)
    cs._apply_engine_gating = lambda: calls.__setitem__("gating", calls["gating"] + 1)
    cs._update_preview = lambda: calls.__setitem__("preview", calls["preview"] + 1)

    cs._apply_hf_result(HFResult("bartowski/Foo-GGUF", "gguf"))
    assert sel.value == "llama-cpp" and calls["model"] == "bartowski/Foo-GGUF"

    sel.value = "mlx-vlm"
    cs._apply_hf_result(HFResult("mlx-community/Bar-4bit", "mlx"))
    assert sel.value == "mlx-vlm" and calls["model"] == "mlx-community/Bar-4bit"  # kept current MLX engine


# --- browse screen render -------------------------------------------------

def test_hf_browse_gates_mlx_by_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # hermetic
    import asyncio

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.hf_browse import HFBrowseScreen
    from mlx_launcher.widgets.toggle_chip import ToggleChip

    monkeypatch.setattr(hf, "search_models", lambda *a, **k: [])  # never network

    async def go(allow_mlx, current_engine):
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(HFBrowseScreen(allow_mlx=allow_mlx, current_engine=current_engine))
            await pilot.pause(0.2)
            scr = app.screen
            gguf = scr.query_one("#fmt-gguf", ToggleChip)
            mlx = scr.query_one("#fmt-mlx", ToggleChip)
            scr.query_one("#hf-results")  # mounts
            scr.query_one("#hf-budget")
            return gguf.value, mlx.value, mlx._locked

    # non-Mac: MLX disabled, GGUF selected
    gguf_on, mlx_on, mlx_locked = asyncio.run(go(False, "llama-cpp"))
    assert gguf_on is True and mlx_on is False and mlx_locked is True
    # Apple Silicon: MLX enabled and the default selection
    gguf_on, mlx_on, mlx_locked = asyncio.run(go(True, "mlx-lm"))
    assert mlx_on is True and mlx_locked is False and gguf_on is False


def test_hf_browse_download_flow_marshals_progress_and_dismisses(tmp_path, monkeypatch):
    # exercises HFBrowseScreen._download_flow end-to-end under a real app: the progress
    # callback runs in a worker thread and must use app.call_from_thread (not screen.*),
    # then the screen dismisses with the HFResult. (Regression: 'Screen has no call_from_thread'.)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.hf_browse import HFBrowseScreen, HFResult

    def fake_dl(repo, on_progress, allow_patterns=None, cancel=None):
        on_progress("fetching…")   # → self.app.call_from_thread(log.write, …) from a worker thread
        on_progress("done")
        return "/cache/" + repo.replace("/", "--")

    monkeypatch.setattr(hf, "download_model", fake_dl)
    monkeypatch.setattr(hf, "exact_to_fetch_bytes", lambda *a, **k: 1234)

    result = {}

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(HFBrowseScreen(allow_mlx=True, current_engine="mlx-lm"),
                                  lambda res: result.__setitem__("v", res))
            await pilot.pause(0.2)
            app.screen._download("mlx-community/Test-4bit", "mlx", allow_patterns=None)
            for _ in range(80):
                await pilot.pause(0.05)
                if "v" in result:
                    break

    asyncio.run(go())
    assert isinstance(result.get("v"), HFResult)
    assert result["v"].repo_id == "mlx-community/Test-4bit" and result["v"].fmt == "mlx"


def test_editor_mounts_search_hf_button(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # hermetic
    import asyncio

    from textual.widgets import Button

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.editor import EditorScreen

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(EditorScreen())
            await pilot.pause(0.2)
            scr = app.screen
            assert str(scr.query_one("#hf-search", Button).label) == "Search HF"
            assert scr.query_one("#model-row").query_one("#model")  # the model field sits in the row

    asyncio.run(go())
