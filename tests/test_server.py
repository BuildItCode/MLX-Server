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


def test_readiness_regex():
    m = STARTING_RE.search("Starting httpd at 127.0.0.1 on port 8080...")
    assert m and m.group(1) == "8080"


def test_is_port_free_true_for_unused_high_port():
    # very unlikely to be bound
    assert discovery.is_port_free("127.0.0.1", 59999) is True


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
