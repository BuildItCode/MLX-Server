"""Optional voice I/O for the chat: speech-to-text (mic) and text-to-speech (read aloud).

Everything heavy is imported lazily, so the core app keeps a *zero* hard dependency on
audio libraries — the chat buttons probe `availability()` and show an install hint when the
optional ``voice`` extra isn't present. The stack is local + key-free, chosen for
"small but good":

  · capture / playback — sounddevice (PortAudio) + numpy
  · speech-to-text     — mlx-whisper on Apple Silicon, faster-whisper elsewhere (Whisper ``base``)
  · text-to-speech     — Kokoro-82M via kokoro-onnx (model auto-downloaded once); falls back to
                         the OS voice (macOS ``say`` / ``espeak-ng``) so "read aloud" works even
                         before the extra is installed.

All public entry points are safe to call without the optional deps installed: they either
return availability info or raise :class:`VoiceError` with an actionable message.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config.store import config_dir

# Whisper wants 16 kHz mono; Kokoro emits 24 kHz. (Kokoro reports its own rate at runtime.)
STT_SR = 16_000

DEFAULT_STT_MODEL = "base"       # whisper size keyword, or a full HF repo id
DEFAULT_TTS_VOICE = "af_heart"   # a Kokoro voice

# mlx-whisper (Apple Silicon) HF conversions, keyed by size. A setting containing a "/" is
# treated as an explicit repo id and used verbatim, so any of these is overridable.
_MLX_WHISPER_REPO = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# Kokoro model files (downloaded once into ~/.config/mlx-launcher/voice/).
_KOKORO_ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class VoiceError(RuntimeError):
    """A voice operation failed (missing backend, mic error, download error, …)."""


@contextlib.contextmanager
def _silence_stderr_fd():
    """Mute the OS-level stderr fd for the duration. mlx_whisper's first-use model download is
    xet-backed, and hf_xet (Rust) prints progress + an "unauthenticated" notice straight to fd 2,
    bypassing Python — which would glitch the TUI. Runs in the transcription worker thread; the
    screen renders on stdout, so this is safe and the fd is always restored."""
    saved = None
    devnull = None
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:  # noqa: BLE001
        if saved is not None:  # don't leak the duped fd if open/dup2 failed
            try:
                os.close(saved)
            except Exception:  # noqa: BLE001
                pass
            saved = None
    finally:
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:  # noqa: BLE001
                pass
    try:
        yield
    finally:
        if saved is not None:
            try:
                os.dup2(saved, 2)
            finally:
                os.close(saved)


# --- capability probing ---------------------------------------------------

def _have(module: str) -> bool:
    """True if an importable top-level module exists, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001 — a broken namespace package shouldn't crash the probe
        return False


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


_tqdm_lock_set = False


def ensure_threading_tqdm_lock() -> None:
    """Swap tqdm's default write lock for a plain threading lock.

    tqdm's default is a *multiprocessing* lock, and creating it spawns Python's
    multiprocessing `resource_tracker` via `fork_exec`. Triggered from a worker thread under
    the Textual app that raises ``ValueError: bad value(s) in fds_to_keep`` (and the lingering
    tracker can hang interpreter shutdown). mlx_whisper (and our own download bar) use tqdm
    internally, so we never want its multiprocessing lock — a threading lock is plenty. Called
    once at app startup on the main thread; idempotent."""
    global _tqdm_lock_set
    if _tqdm_lock_set:
        return
    try:
        import threading

        import tqdm
        tqdm.tqdm.set_lock(threading.RLock())
        _tqdm_lock_set = True
    except Exception:  # noqa: BLE001 — tqdm absent or API change → leave it alone
        pass
    # also quiet huggingface_hub's own download bars (mlx_whisper's "Fetching files") — they'd
    # write to the terminal; our HF-download bar uses its own force-enabled tqdm, so it's safe.
    try:
        import huggingface_hub
        huggingface_hub.utils.disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass


def _system_tts_argv(text: str) -> Optional[list[str]]:
    """An OS text-to-speech command for `text`, or None if the platform has none."""
    say = shutil.which("say")  # macOS — always present there
    if say:
        return [say, text]
    for cmd in ("espeak-ng", "espeak"):  # common on Linux
        path = shutil.which(cmd)
        if path:
            return [path, text]
    return None


@dataclass(frozen=True)
class Availability:
    """What's installed right now, used to gate the chat's mic / read-aloud buttons."""

    audio_io: bool      # sounddevice + numpy (mic capture and Kokoro playback)
    stt_mlx: bool       # mlx-whisper (Apple Silicon)
    stt_faster: bool    # faster-whisper (any platform)
    tts_kokoro: bool    # kokoro-onnx + audio_io
    tts_system: bool    # an OS voice (say / espeak)

    @property
    def can_record(self) -> bool:
        return self.audio_io

    @property
    def can_transcribe(self) -> bool:
        # mic input needs both capture (sounddevice) and a Whisper backend
        return self.audio_io and (self.stt_mlx or self.stt_faster)

    @property
    def can_speak(self) -> bool:
        return self.tts_kokoro or self.tts_system


def availability() -> Availability:
    audio_io = _have("sounddevice") and _have("numpy")
    return Availability(
        audio_io=audio_io,
        stt_mlx=is_apple_silicon() and _have("mlx_whisper"),
        stt_faster=_have("faster_whisper"),
        tts_kokoro=_have("kokoro_onnx") and audio_io,
        tts_system=_system_tts_argv("") is not None,
    )


def install_command() -> str:
    """The pip command that enables the full voice feature on this platform. Installs into
    the *running* interpreter (so the imports resolve) — engines go on PATH, these don't."""
    pkgs = ["sounddevice", "numpy", "kokoro-onnx"]
    pkgs.append("mlx-whisper" if is_apple_silicon() else "faster-whisper")
    return f"{Path(sys.executable).name} -m pip install " + " ".join(pkgs)


# --- speech-to-text -------------------------------------------------------

class Recorder:
    """Push-to-talk mic capture via a sounddevice input stream. start() opens the stream
    (PortAudio fills a buffer on its own thread); stop() returns float32 mono @ 16 kHz."""

    def __init__(self, samplerate: int = STT_SR) -> None:
        self._sr = samplerate
        self._frames: list = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            raise VoiceError(f"audio capture unavailable ({exc}); install: {install_command()}") from exc

        self._frames = []

        def _cb(indata, _frames, _time, _status) -> None:  # PortAudio thread
            with self._lock:
                self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(samplerate=self._sr, channels=1, dtype="float32", callback=_cb)
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 — no mic, busy device, unsupported rate, …
            self._stream = None
            raise VoiceError(f"couldn't open the microphone: {exc}") from exc

    def stop(self):
        """Stop capturing and return the recorded audio (np.float32, mono, 16 kHz)."""
        import numpy as np

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)


_faster_model = None  # cache: loading a WhisperModel is expensive


def _mlx_whisper_repo(setting: str) -> str:
    s = (setting or DEFAULT_STT_MODEL).strip()
    if "/" in s:  # already a HF repo id
        return s
    return _MLX_WHISPER_REPO.get(s.lower(), _MLX_WHISPER_REPO["base"])


def transcribe(audio, model_setting: str = DEFAULT_STT_MODEL) -> str:
    """Transcribe float32 mono 16 kHz audio to text. Uses mlx-whisper on Apple Silicon,
    else faster-whisper. Blocking + CPU/GPU heavy — call from a worker thread."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < STT_SR // 5:  # < 0.2 s — almost certainly an accidental tap
        return ""

    ensure_threading_tqdm_lock()  # mlx_whisper uses tqdm internally; avoid its mp-lock fork_exec

    if is_apple_silicon() and _have("mlx_whisper"):
        import mlx_whisper

        with _silence_stderr_fd():  # mute hf_xet's first-download progress/warning on fd 2
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=_mlx_whisper_repo(model_setting))
        return (result.get("text") or "").strip()

    if _have("faster_whisper"):
        global _faster_model
        from faster_whisper import WhisperModel

        size = model_setting if "/" not in (model_setting or "") else "base"
        size = (size or "base").strip() or "base"
        with _silence_stderr_fd():
            if _faster_model is None or getattr(_faster_model, "_lis_size", None) != size:
                _faster_model = WhisperModel(size, device="cpu", compute_type="int8")
                _faster_model._lis_size = size  # type: ignore[attr-defined]
            segments, _info = _faster_model.transcribe(audio, beam_size=1)
            return " ".join(seg.text for seg in segments).strip()

    raise VoiceError(f"no speech-to-text backend installed; install: {install_command()}")


# --- text-to-speech -------------------------------------------------------

def _speakable(markdown: str) -> str:
    """Reduce assistant markdown to plain prose worth reading aloud: drop code blocks,
    unwrap links/emphasis, strip heading/list markers."""
    t = markdown or ""
    t = re.sub(r"```.*?```", " (code block) ", t, flags=re.S)
    t = re.sub(r"~~~.*?~~~", " (code block) ", t, flags=re.S)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)          # ![alt](url) -> alt
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)            # [text](url) -> text
    t = t.replace("`", "")
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)               # headings
    t = re.sub(r"(?m)^\s{0,3}>\s?", "", t)                    # blockquotes
    t = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", t)                # bullet lists
    t = re.sub(r"(?m)^\s{0,3}\d+\.\s+", "", t)                # ordered lists
    t = re.sub(r"(\*\*|__|\*|_|~~)", "", t)                   # emphasis
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def _split_text(text: str, max_len: int = 480) -> list[str]:
    """Sentence-ish chunks under max_len, so playback starts sooner and Stop is responsive."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？\n])\s+", text)
    chunks: list[str] = []
    cur = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if cur and len(cur) + len(part) + 1 > max_len:
            chunks.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip() if cur else part
    if cur:
        chunks.append(cur)
    return chunks


_kokoro = None  # cache: loading the ONNX model is expensive


def _download(url: str, dest: Path) -> None:
    import httpx

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        raise VoiceError(f"couldn't download the Kokoro voice model: {exc}") from exc


def _ensure_kokoro_files() -> tuple[str, str]:
    d = config_dir() / "voice"
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / "kokoro-v1.0.onnx"
    voices = d / "voices-v1.0.bin"
    if not onnx.exists():
        _download(_KOKORO_ONNX_URL, onnx)
    if not voices.exists():
        _download(_KOKORO_VOICES_URL, voices)
    return str(onnx), str(voices)


def _kokoro_instance():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        onnx, voices = _ensure_kokoro_files()
        _kokoro = Kokoro(onnx, voices)
    return _kokoro


def kokoro_models_ready() -> bool:
    """True if the Kokoro model files are already downloaded (so we won't block on a ~325 MB
    fetch). Used to message the user before the first synthesis."""
    d = config_dir() / "voice"
    return (d / "kokoro-v1.0.onnx").exists() and (d / "voices-v1.0.bin").exists()


class Speaker:
    """Synthesizes + plays `text`. Prefers Kokoro (best quality); falls back to the OS voice.
    run() blocks (call from a worker thread); stop() interrupts it from any thread."""

    def __init__(self, text: str, voice: str = DEFAULT_TTS_VOICE) -> None:
        self.text = text
        self.voice = voice or DEFAULT_TTS_VOICE
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001 — sounddevice may not be installed (system path)
            pass
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    def run(self) -> None:
        text = _speakable(self.text)
        if not text:
            return
        if _have("kokoro_onnx") and _have("sounddevice"):
            try:
                self._run_kokoro(text)
                return
            except Exception:  # noqa: BLE001 — model/download/runtime issue → fall back to the OS voice
                if self._stop.is_set():
                    return
        self._run_system(text)

    def _run_kokoro(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        kokoro = _kokoro_instance()
        # Synthesize every sentence-chunk first, then play ONE contiguous buffer. A separate
        # sd.play() per chunk opens/closes a fresh stream each time → clicks + gaps at every
        # boundary (the "artifacts"); a single stream over joined audio is gapless.
        parts: list = []
        sample_rate = 24000
        for chunk in _split_text(text):
            if self._stop.is_set():
                return
            samples, sample_rate = kokoro.create(chunk, voice=self.voice, speed=1.0, lang="en-us")
            parts.append(np.ascontiguousarray(samples, dtype=np.float32))
        if not parts or self._stop.is_set():
            return
        if len(parts) == 1:
            audio = parts[0]
        else:
            gap = np.zeros(int(0.06 * sample_rate), dtype=np.float32)  # ~60ms pause between sentences
            joined: list = []
            for i, part in enumerate(parts):
                joined.append(part)
                if i < len(parts) - 1:
                    joined.append(gap)
            audio = np.concatenate(joined)
        # latency="high" → a larger output buffer, which avoids the underrun clicks that the
        # default low-latency buffer produces while the UI/model keep the CPU busy.
        try:
            sd.play(audio, int(sample_rate), latency="high")
        except Exception:  # noqa: BLE001 — some backends reject the latency hint
            sd.play(audio, int(sample_rate))
        if self._stop.is_set():  # stop() raced in just as playback started → cut it immediately
            sd.stop()
            return
        sd.wait()  # returns early when stop() calls sd.stop()

    def _run_system(self, text: str) -> None:
        argv = _system_tts_argv(text)
        if not argv:
            raise VoiceError(f"no text-to-speech available; install: {install_command()}")
        if self._stop.is_set():
            return
        self._proc = subprocess.Popen(argv)
        if self._stop.is_set():  # stop() raced in right after spawn → terminate immediately
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            return
        self._proc.wait()
