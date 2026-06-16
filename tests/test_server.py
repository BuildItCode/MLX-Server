import socket

from mlx_launcher.server import discovery
from mlx_launcher.server.readiness import STARTING_RE


def test_exit_message_unsupported_architecture():
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    mgr = ServerManager(ServerConfig(model="/m", port=8080))
    mgr.log_buffer.append(("stderr", "ValueError: Model type step3p7 not supported."))
    msg = mgr._exit_message(1)
    # text engine: name the arch, the engine, and suggest mlx-vlm for vision models
    assert "step3p7" in msg and "mlx-lm" in msg and "mlx-vlm" in msg


def test_exit_message_unsupported_architecture_vlm_engine():
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    mgr = ServerManager(ServerConfig(model="/m", engine="mlx-vlm"))
    mgr.log_buffer.append(("stderr", "ValueError: Model type llava_next not supported."))
    msg = mgr._exit_message(1)
    assert "llava_next" in msg and "mlx-vlm" in msg


def test_binary_name_maps_engines():
    assert discovery.binary_name("mlx-lm") == "mlx_lm.server"
    assert discovery.binary_name("mlx-vlm") == "mlx_vlm.server"
    assert discovery.binary_name() == "mlx_lm.server"


def test_exit_message_generic_fallback():
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    mgr = ServerManager(ServerConfig(model="/m"))
    mgr.log_buffer.append(("stderr", "RuntimeError: something else"))
    assert "code 7" in mgr._exit_message(7)


def test_start_is_a_noop_when_already_running(monkeypatch):
    import asyncio

    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    mgr = ServerManager(ServerConfig(model="/m", port=8080))

    class FakeProc:
        returncode = None  # alive
        pid = 999999

    mgr.proc = sentinel = FakeProc()
    spawned = {"n": 0}

    async def boom(*a, **k):
        spawned["n"] += 1

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    asyncio.run(mgr.start())
    # the guard returns before spawning, so the live process is neither overwritten nor
    # joined by a second orphan
    assert mgr.proc is sentinel and spawned["n"] == 0


def test_is_alive_is_false_without_a_process():
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    assert ServerManager(ServerConfig(model="/m")).is_alive() is False


def test_readiness_regex():
    m = STARTING_RE.search("Starting httpd at 127.0.0.1 on port 8080...")
    assert m and m.group(1) == "8080"


def test_is_port_free_true_for_unused_high_port():
    # very unlikely to be bound
    assert discovery.is_port_free("127.0.0.1", 59999) is True


def test_binary_name_llama_cpp():
    assert discovery.binary_name("llama-cpp") == "llama-server"


def test_resolve_gguf_picks_main_file_from_a_model_folder(tmp_path):
    # LM Studio layout: a folder with the model .gguf + a vision projector (mmproj).
    d = tmp_path / "gemma-4-31B-GGUF"
    d.mkdir()
    (d / "gemma-4-31B-Q4_0.gguf").write_bytes(b"x")
    (d / "mmproj-gemma-4-31B-BF16.gguf").write_bytes(b"x")
    assert discovery.resolve_gguf(str(d)) == str(d / "gemma-4-31B-Q4_0.gguf")  # model, not mmproj
    assert discovery.find_mmproj(str(d)) == str(d / "mmproj-gemma-4-31B-BF16.gguf")


def test_resolve_gguf_picks_first_shard(tmp_path):
    d = tmp_path / "big"
    d.mkdir()
    for i in (1, 2, 3):
        (d / f"model-0000{i}-of-00003.gguf").write_bytes(b"x")
    assert discovery.resolve_gguf(str(d)).endswith("-00001-of-00003.gguf")


def test_resolve_gguf_passes_through_files_and_repos(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert discovery.resolve_gguf(str(f)) == str(f)             # already a file → unchanged
    assert discovery.resolve_gguf("org/repo-GGUF:Q4_K_M") == "org/repo-GGUF:Q4_K_M"  # HF repo → unchanged
    assert discovery.find_mmproj(str(f)) is None


def test_exit_message_gguf_load_error():
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    mgr = ServerManager(ServerConfig(model="/m.gguf", engine="llama-cpp"))
    mgr.log_buffer.append(("stderr", "error loading model: failed to load model from '/m.gguf'"))
    msg = mgr._exit_message(1)
    assert "GGUF" in msg and "llama-server" in msg


def test_spawn_kwargs_per_platform(monkeypatch):
    # the server must spawn into its own group on every OS: POSIX session / Windows group
    import mlx_launcher.server.manager as m

    monkeypatch.setattr(m.sys, "platform", "linux")
    assert m._spawn_kwargs() == {"start_new_session": True}

    monkeypatch.setattr(m.sys, "platform", "win32")
    # CREATE_NEW_PROCESS_GROUP doesn't exist on this (non-Windows) host — patch it in
    monkeypatch.setattr(m.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    assert m._spawn_kwargs() == {"creationflags": 0x200}


def test_is_alive_windows_uses_returncode_not_os_kill(monkeypatch):
    # os.kill(pid, 0) TERMINATES the process on Windows — is_alive must never call it there.
    import mlx_launcher.server.manager as m
    from mlx_launcher.config.models import ServerConfig
    from mlx_launcher.server.manager import ServerManager

    monkeypatch.setattr(m.sys, "platform", "win32")
    monkeypatch.setattr(m.os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("os.kill on win32")))

    class FakeProc:
        pid = 4321
        returncode = None

    mgr = ServerManager(ServerConfig(model="/m.gguf", engine="llama-cpp"))
    mgr.proc = FakeProc()
    assert mgr.is_alive() is True       # returncode is None → alive
    mgr.proc.returncode = 0
    assert mgr.is_alive() is False      # exited


def test_terminate_and_kill_route_per_platform_on_windows(monkeypatch):
    import mlx_launcher.server.manager as m
    from mlx_launcher.server.manager import ServerManager

    monkeypatch.setattr(m.sys, "platform", "win32")
    monkeypatch.setattr(m.os, "killpg", lambda *a: (_ for _ in ()).throw(AssertionError("killpg on win32")))
    taskkilled = []
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **k: taskkilled.append(argv))

    class FakeProc:
        pid = 99
        def __init__(self): self.terminated = False
        def terminate(self): self.terminated = True

    p = FakeProc()
    ServerManager._terminate_proc(p)
    assert p.terminated is True                              # graceful → proc.terminate()
    ServerManager._kill_proc(p)
    assert taskkilled and taskkilled[0][:2] == ["taskkill", "/F"]  # force → taskkill /F /T


def test_is_port_free_false_when_listening():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert discovery.is_port_free("127.0.0.1", port) is False
    finally:
        srv.close()
