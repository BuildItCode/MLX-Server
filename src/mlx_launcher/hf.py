"""HuggingFace model search + download — the network/logic layer behind the editor's
"Search HF" flow. Deliberately Textual-free so it's unit-testable; `huggingface_hub` is
imported lazily inside functions (keeps app startup cheap and lets tests patch a fake).

Two formats are supported, matching the engines:
  · **gguf** → llama.cpp. A repo usually holds *many* quant files, so the caller picks one
    (`list_repo_gguf_files`) and we download only that file (+ its mmproj sidecar).
  · **mlx**  → the MLX engines (Apple Silicon). A repo is one model; download it whole.

Sizes drive a device-memory recommendation: a name/quant heuristic for instant list badges
(`estimate_size_from_name`), upgraded to the exact "bytes to fetch" via a dry-run before a
download. On Apple Silicon, GPU shares system RAM (unified memory), so the budget is a
fraction of total RAM.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from ._util import is_apple_silicon, silence_native_stderr

Format = Literal["gguf", "mlx"]

# list_models(filter=…) tag tokens (verified against the live Hub).
_FILTER = {"gguf": "gguf", "mlx": "mlx"}

# Fraction of total RAM a model should fit within (leaves headroom for the OS, app, and a
# growing KV cache). On Apple Silicon this RAM *is* the GPU budget (unified memory).
_BUDGET_FRACTION = 0.70

ProgressCb = Callable[[str], None]


class HFError(RuntimeError):
    """A HuggingFace operation failed (network, auth, gated repo, no results, …)."""


class HFCancelled(Exception):
    """The user cancelled an in-flight download."""


@dataclass(frozen=True)
class ModelHit:
    repo_id: str
    fmt: str                      # "gguf" | "mlx"
    downloads: int
    likes: int
    tags: tuple[str, ...]
    size_bytes: Optional[int]     # exact when known, else heuristic, else None
    size_exact: bool              # False ⇒ heuristic estimate (UI prefixes it with "~")
    gated: bool = False


@dataclass(frozen=True)
class GgufFile:
    filename: str
    size_bytes: Optional[int]


# --- huggingface_hub access (lazy) ---------------------------------------

_hf_quieted = False


def _hf():
    try:
        import huggingface_hub as hub
    except Exception as exc:  # noqa: BLE001
        raise HFError(f"huggingface_hub is not available: {exc}") from exc
    global _hf_quieted
    if not _hf_quieted:
        # Silence hf's own tqdm bars + the "unauthenticated request" warning — under the TUI
        # they'd corrupt the screen. Our download bar (a plain tqdm subclass that forces itself
        # enabled) is unaffected by disable_progress_bars().
        try:
            hub.utils.disable_progress_bars()
        except Exception:  # noqa: BLE001
            pass
        try:
            hub.utils.logging.set_verbosity_error()
        except Exception:  # noqa: BLE001
            pass
        _hf_quieted = True
    return hub


def _token() -> Optional[str]:
    try:
        return _hf().get_token()
    except Exception:  # noqa: BLE001
        return None


def _explain(exc: Exception) -> str:
    """A friendlier message for the common HF failures."""
    name = type(exc).__name__
    if "Gated" in name:
        return "This model is gated — accept its terms on huggingface.co and set HF_TOKEN, then retry."
    if "RepositoryNotFound" in name or "EntryNotFound" in name:
        return "Not found on HuggingFace (or private without a token)."
    return f"{name}: {exc}"


# --- search ---------------------------------------------------------------

def search_models(query: str, fmt: Format, limit: int = 30) -> list[ModelHit]:
    """Search the Hub for `fmt` models (blocking — call via asyncio.to_thread). Never raises
    for "no results"; raises HFError on a network/auth failure."""
    hub = _hf()
    api = hub.HfApi()
    try:
        rows = list(api.list_models(
            filter=_FILTER[fmt], search=(query or None), sort="downloads",
            limit=limit, full=True, token=_token(),
        ))
    except Exception as exc:  # noqa: BLE001
        raise HFError(_explain(exc)) from exc

    hits: list[ModelHit] = []
    for m in rows:
        repo_id = getattr(m, "id", "") or ""
        if not repo_id:
            continue
        est = estimate_size_from_name(repo_id, fmt)
        hits.append(ModelHit(
            repo_id=repo_id,
            fmt=fmt,
            downloads=int(getattr(m, "downloads", 0) or 0),
            likes=int(getattr(m, "likes", 0) or 0),
            tags=tuple(getattr(m, "tags", ()) or ()),
            size_bytes=est,
            size_exact=False,
            gated=bool(getattr(m, "gated", False) or getattr(m, "private", False)),
        ))
    return hits


# --- size estimation ------------------------------------------------------

# Effective bits-per-weight (incl. metadata overhead), longest tokens first so e.g. "q4_k_m"
# matches before "q4_k". Values are rough — enough for a fit badge.
_GGUF_BPW: tuple[tuple[str, float], ...] = (
    ("q2_k", 2.6), ("q3_k_s", 3.5), ("q3_k_m", 3.9), ("q3_k_l", 4.3), ("q3_k", 3.9),
    ("q4_k_s", 4.6), ("q4_k_m", 4.8), ("q4_k", 4.8), ("q4_0", 4.5), ("q4_1", 5.0),
    ("q5_k_s", 5.5), ("q5_k_m", 5.7), ("q5_k", 5.7), ("q5_0", 5.5), ("q5_1", 6.0),
    ("q6_k", 6.6), ("q8_0", 8.5), ("iq1", 1.7), ("iq2", 2.4), ("iq3", 3.3), ("iq4", 4.3),
    ("bf16", 16.0), ("fp16", 16.0), ("f16", 16.0), ("f32", 32.0), ("fp32", 32.0),
)
_MLX_BPW: tuple[tuple[str, float], ...] = (
    ("2bit", 2.5), ("3bit", 3.4), ("4bit", 4.3), ("5bit", 5.4), ("6bit", 6.4), ("8bit", 8.5),
    ("bf16", 16.0), ("fp16", 16.0), ("f16", 16.0), ("fp32", 32.0), ("f32", 32.0),
)


def _params_billions(name: str) -> Optional[float]:
    """Parameter count in billions parsed from a model name, or None. Handles MoE `8x7B`
    and avoids matching quant tokens like `4bit` (the `b` there is followed by a letter)."""
    moe = re.search(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)
    if moe:
        return int(moe.group(1)) * float(moe.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", name)
    return float(m.group(1)) if m else None


def _bits_per_weight(name: str, fmt: Format) -> float:
    low = name.lower()
    table = _GGUF_BPW if fmt == "gguf" else _MLX_BPW
    for token, bpw in table:
        if token in low:
            return bpw
    return 5.0 if fmt == "gguf" else 16.0  # GGUF uploads are usually quantized; MLX defaults full-precision


def estimate_size_from_name(repo_id: str, fmt: Format) -> Optional[int]:
    """A rough on-disk byte estimate from the name's parameter count × quant bits. Returns
    None when no parameter count is parseable (the UI then shows "size ?")."""
    params = _params_billions(repo_id)
    if params is None:
        return None
    return int(params * 1e9 * _bits_per_weight(repo_id, fmt) / 8)


# --- GGUF file listing + exact sizes -------------------------------------

def list_repo_gguf_files(repo_id: str) -> tuple[list[GgufFile], Optional[str]]:
    """The downloadable `.gguf` quant files in a repo (each with its exact size), plus the
    name of an `mmproj` vision projector if present. Blocking."""
    hub = _hf()
    try:
        info = hub.HfApi().model_info(repo_id, files_metadata=True, token=_token())
    except Exception as exc:  # noqa: BLE001
        raise HFError(_explain(exc)) from exc
    files: list[GgufFile] = []
    mmproj: Optional[str] = None
    for sib in getattr(info, "siblings", None) or []:
        name = getattr(sib, "rfilename", "") or ""
        if not name.lower().endswith(".gguf"):
            continue
        if "mmproj" in name.lower():
            mmproj = name
            continue
        files.append(GgufFile(filename=name, size_bytes=getattr(sib, "size", None)))
    files.sort(key=lambda f: (f.size_bytes or 0))
    return files, mmproj


def exact_to_fetch_bytes(repo_id: str, *, allow_patterns: Optional[list[str]] = None) -> Optional[int]:
    """Exact bytes a download WOULD fetch (excludes already-cached files), via a dry-run.
    Returns None if the dry-run API isn't available. Blocking."""
    hub = _hf()
    try:
        with silence_native_stderr():
            plan = hub.snapshot_download(
                repo_id, dry_run=True, allow_patterns=allow_patterns, token=_token(),
                tqdm_class=_progress_tqdm(lambda _line: None, None),  # silent — no stray bar under the TUI
            )
    except TypeError:
        return None  # older huggingface_hub without dry_run
    except Exception as exc:  # noqa: BLE001
        raise HFError(_explain(exc)) from exc
    total = 0
    for f in plan or []:
        if getattr(f, "will_download", True):
            total += int(getattr(f, "file_size", 0) or 0)
    return total


# --- device memory + fit --------------------------------------------------

def total_ram_bytes() -> Optional[int]:
    """Total physical RAM in bytes, cross-platform, or None if undetectable."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=3)
            return int(out.strip())
        if sys.platform == "win32":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = _Mem()
            stat.dwLength = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        # Linux / other POSIX
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        return None
    return None


def recommended_budget_bytes() -> Optional[int]:
    """Memory budget a model should fit within, or None if RAM is unknown. On Apple Silicon
    this is a fraction of unified memory (shared with the GPU)."""
    ram = total_ram_bytes()
    return int(ram * _BUDGET_FRACTION) if ram else None


FitState = Literal["fits", "tight", "too_big", "unknown"]


def fit(size_bytes: Optional[int], budget_bytes: Optional[int]) -> FitState:
    if not size_bytes or not budget_bytes:
        return "unknown"
    ratio = size_bytes / budget_bytes
    if ratio <= 0.80:
        return "fits"
    if ratio <= 1.0:
        return "tight"
    return "too_big"


def human(n: Optional[int]) -> str:
    """Bytes → a short human string ("5.2 GB", "812 MB", "0 B", "?" when unknown)."""
    if n is None or n < 0:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    val = float(n)
    for unit in units:
        if val < 1024 or unit == "TB":
            prec = 0 if (unit in ("B", "KB") or val >= 100) else 1
            return f"{val:.{prec}f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


# --- download -------------------------------------------------------------

def _progress_tqdm(on_progress: ProgressCb, cancel: Optional[Callable[[], bool]]):
    """A tqdm subclass for snapshot_download's aggregated bytes bar: forwards throttled
    `done/total` lines to `on_progress`, renders nothing to the terminal, and raises
    HFCancelled (best-effort interrupt) when `cancel()` turns true."""
    from tqdm.auto import tqdm as _Tqdm

    class _ProgressTqdm(_Tqdm):
        def __init__(self, *args, **kwargs):
            # force-enable: under the TUI there's no TTY, and a disabled tqdm's update()
            # won't advance self.n. We render nothing (refresh/display are no-ops below), so
            # nothing reaches the terminal — we only read self.n to emit our own lines.
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._last_pct = -1

        def refresh(self, *a, **k):  # no terminal rendering under the TUI
            return

        def display(self, *a, **k):
            return

        def update(self, n=1):
            if cancel is not None and cancel():
                raise HFCancelled()
            super().update(n)
            total = self.total
            if total:
                pct = int(self.n * 100 / total)
                if pct != self._last_pct:
                    self._last_pct = pct
                    on_progress(f"  ↓ {human(int(self.n))} / {human(int(total))}  ({pct}%)")
            return True

    return _ProgressTqdm


def download_model(
    repo_id: str,
    on_progress: ProgressCb,
    *,
    allow_patterns: Optional[list[str]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Download `repo_id` (optionally only `allow_patterns`) into the HuggingFace cache,
    streaming progress to `on_progress`. Returns the local snapshot path. Blocking + heavy —
    MUST run in a worker thread. Raises HFCancelled if cancelled, HFError on failure."""
    hub = _hf()
    try:
        from huggingface_hub import constants as _const
        cache = getattr(_const, "HF_HUB_CACHE", "the HuggingFace cache")
    except Exception:  # noqa: BLE001
        cache = "the HuggingFace cache"
    on_progress(f"Resolving {repo_id} …  → {cache}")
    # silence huggingface_hub's own per-file stderr bars (would corrupt the TUI); our custom
    # aggregate tqdm is a plain tqdm subclass and is unaffected by this flag.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        with silence_native_stderr():
            path = hub.snapshot_download(
                repo_id, allow_patterns=allow_patterns, token=_token(),
                tqdm_class=_progress_tqdm(on_progress, cancel),
            )
    except HFCancelled:
        on_progress("Cancelled — the partial download stays cached and resumes next time.")
        raise
    except Exception as exc:  # noqa: BLE001
        raise HFError(_explain(exc)) from exc
    on_progress("✓ download complete")
    return path
