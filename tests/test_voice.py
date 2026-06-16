"""Voice I/O: capability probing, markdown→speech cleanup, model resolution, and the
chat mic / read-aloud wiring. All hermetic — no audio hardware, no optional deps required
(the heavy backends are imported lazily, so these never touch them)."""

import importlib.util

from mlx_launcher import bootstrap
from mlx_launcher.chat import voice


# --- pure helpers ---------------------------------------------------------

def test_speakable_strips_markdown():
    md = "# Title\n\nSome **bold** and `code` and a [link](http://x.com).\n\n```py\nprint(1)\n```\n- a\n- b\n"
    out = voice._speakable(md)
    assert "Title" in out and "bold" in out and "link" in out  # text kept
    assert "**" not in out and "`" not in out                  # emphasis / inline-code markers gone
    assert not out.lstrip().startswith("#")                    # heading marker stripped
    assert "print(1)" not in out and "(code block)" in out     # fenced code replaced, not read
    assert "http://x.com" not in out                           # link URL dropped, link text kept


def test_split_text_chunks_under_max_and_preserves_content():
    text = "Sentence one. Sentence two! Sentence three? " * 12
    chunks = voice._split_text(text, max_len=120)
    assert chunks
    assert max(len(c) for c in chunks) <= 120
    assert "Sentence one" in " ".join(chunks)
    assert voice._split_text("") == []
    assert voice._split_text("   ") == []


def test_mlx_whisper_repo_maps_sizes_and_passes_through_repo_ids():
    assert voice._mlx_whisper_repo("base") == voice._MLX_WHISPER_REPO["base"]
    assert voice._mlx_whisper_repo("BASE") == voice._MLX_WHISPER_REPO["base"]   # case-insensitive
    assert voice._mlx_whisper_repo("org/custom-whisper") == "org/custom-whisper"  # explicit repo id
    assert voice._mlx_whisper_repo("") == voice._MLX_WHISPER_REPO["base"]       # default
    assert voice._mlx_whisper_repo("nonsense") == voice._MLX_WHISPER_REPO["base"]


def test_system_tts_argv_prefers_say_then_espeak(monkeypatch):
    monkeypatch.setattr(voice.shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    assert voice._system_tts_argv("hello") == ["/usr/bin/say", "hello"]

    monkeypatch.setattr(voice.shutil, "which",
                        lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None)
    argv = voice._system_tts_argv("hi")
    assert argv and argv[0].endswith("espeak-ng") and argv[1] == "hi"

    monkeypatch.setattr(voice.shutil, "which", lambda name: None)
    assert voice._system_tts_argv("x") is None


# --- availability ---------------------------------------------------------

def test_availability_reflects_installed_backends(monkeypatch):
    present = {"sounddevice", "numpy", "mlx_whisper", "kokoro_onnx"}
    monkeypatch.setattr(voice, "_have", lambda m: m in present)
    monkeypatch.setattr(voice, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(voice, "_system_tts_argv", lambda t: ["say", t])

    a = voice.availability()
    assert a.audio_io and a.stt_mlx and a.tts_kokoro and a.tts_system
    assert a.can_record and a.can_transcribe and a.can_speak

    present.discard("sounddevice")  # no capture → no mic, and Kokoro can't play
    a = voice.availability()
    assert not a.can_record and not a.can_transcribe and not a.tts_kokoro
    assert a.can_speak  # the system voice still works


def test_availability_system_voice_only(monkeypatch):
    monkeypatch.setattr(voice, "_have", lambda m: False)  # nothing pip-installed
    monkeypatch.setattr(voice, "_system_tts_argv", lambda t: ["say", t])
    a = voice.availability()
    assert not a.can_transcribe and a.can_speak  # read-aloud works out of the box, mic doesn't


def test_install_command_is_platform_aware(monkeypatch):
    monkeypatch.setattr(voice, "is_apple_silicon", lambda: True)
    assert "mlx-whisper" in voice.install_command() and "faster-whisper" not in voice.install_command()
    monkeypatch.setattr(voice, "is_apple_silicon", lambda: False)
    cmd = voice.install_command()
    assert "faster-whisper" in cmd and "mlx-whisper" not in cmd
    assert "kokoro-onnx" in cmd and "sounddevice" in cmd


def test_kokoro_models_ready_false_in_fresh_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert voice.kokoro_models_ready() is False


def test_transcribe_ignores_too_short_audio():
    if importlib.util.find_spec("numpy") is None:
        import pytest
        pytest.skip("numpy (voice extra) not installed")
    import numpy as np
    # < 0.2 s of audio is an accidental tap → empty transcript, no backend needed
    assert voice.transcribe(np.zeros(100, dtype=np.float32)) == ""


# --- bootstrap install argv ----------------------------------------------

def test_voice_install_argv_lists_packages_for_this_interpreter():
    argv = bootstrap.voice_install_argv()
    assert argv[1:4] == ["-m", "pip", "install"]
    assert "sounddevice" in argv and "numpy" in argv and "kokoro-onnx" in argv
    assert ("mlx-whisper" in argv) ^ ("faster-whisper" in argv)  # exactly one STT backend


def test_voice_packages_non_arm_uses_faster_whisper(monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    pkgs = bootstrap.voice_packages()
    assert "faster-whisper" in pkgs and "mlx-whisper" not in pkgs


# --- chat screen wiring (unit, no full mount) ----------------------------

def _bare_screen():
    from mlx_launcher.screens.chat import ChatScreen
    return ChatScreen.__new__(ChatScreen)


def test_last_assistant_text_returns_most_recent_reply():
    from mlx_launcher.chat.models import Chat, ChatMessage
    cs = _bare_screen()
    cs.chat = Chat(title="t", model="m")
    cs.chat.messages = [
        ChatMessage(role="user", text="hi"),
        ChatMessage(role="assistant", text="first"),
        ChatMessage(role="user", text="more"),
        ChatMessage(role="assistant", text="second"),
    ]
    assert cs._last_assistant_text() == "second"
    cs.chat.messages = []
    assert cs._last_assistant_text() == ""


def test_maybe_autoread_speaks_only_when_enabled(monkeypatch):
    from mlx_launcher.chat.models import Chat, ChatMessage
    from mlx_launcher.screens.chat import ChatScreen

    cs = _bare_screen()
    cs.chat = Chat(title="t", model="m")
    cs.chat.messages = [ChatMessage(role="assistant", text="hello there")]
    cs._speaker = None
    spoken = []
    cs._speak = lambda text: spoken.append(text)

    class S:
        voice_autoread = False

    class A:
        class config:
            settings = S()

    monkeypatch.setattr(ChatScreen, "app", property(lambda self: A()))
    monkeypatch.setattr(voice, "availability",
                        lambda: voice.Availability(True, True, False, True, True))

    cs._maybe_autoread()
    assert spoken == []                      # off → silent
    S.voice_autoread = True
    cs._maybe_autoread()
    assert spoken == ["hello there"]         # on → reads the last reply
    cs._speaker = object()                   # already reading → don't stack a second
    cs._maybe_autoread()
    assert spoken == ["hello there"]


def test_start_reading_hints_when_no_tts(monkeypatch):
    from mlx_launcher.chat.models import Chat, ChatMessage

    cs = _bare_screen()
    cs.chat = Chat(title="t", model="m")
    cs.chat.messages = [ChatMessage(role="assistant", text="a reply")]
    cs._speaker = None
    notes = []
    cs.notify = lambda msg, **k: notes.append(msg)
    monkeypatch.setattr(voice, "availability",
                        lambda: voice.Availability(False, False, False, False, False))
    cs._start_reading()
    assert notes and "Install" in notes[0]   # actionable hint, no crash


def test_mic_button_label_toggles():
    cs = _bare_screen()

    class Btn:
        def __init__(self):
            self.label = ""
            self.classes = set()

        def set_class(self, add, name):
            (self.classes.add if add else self.classes.discard)(name)

    btn = Btn()
    cs.query_one = lambda *a, **k: btn
    cs._set_mic_button(recording=True)
    assert btn.label == "■ Listening…" and "-recording" in btn.classes
    cs._set_mic_button(recording=False)
    assert btn.label == "🎙 Mic" and "-recording" not in btn.classes


def test_chat_screen_mounts_voice_buttons(tmp_path, monkeypatch):
    # full render, hermetic (isolated XDG so it never opens/persists the user's real chats)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import asyncio

    from textual.widgets import Button

    from mlx_launcher.app import MlxLauncherApp
    from mlx_launcher.screens.chat import ChatScreen

    async def go():
        app = MlxLauncherApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.2)
            await app.push_screen(ChatScreen())
            await pilot.pause(0.3)
            scr = app.screen
            mic = scr.query_one("#mic-btn", Button)
            read = scr.query_one("#read-aloud", Button)
            assert str(mic.label) == "🎙 Mic" and str(read.label) == "🔊 Read aloud"

            # the live-state helpers drive the real widgets' label + style class
            scr._set_mic_button(recording=True)
            scr._set_read_button(reading=True)
            await pilot.pause()
            assert str(mic.label) == "■ Listening…" and mic.has_class("-recording")
            assert str(read.label) == "■ Stop" and read.has_class("-reading")

    asyncio.run(go())
