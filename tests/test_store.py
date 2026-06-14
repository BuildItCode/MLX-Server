from mlx_launcher.config import store
from mlx_launcher.config.models import ConfigFile, ServerConfig


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert store.config_path() == tmp_path / "mlx-launcher" / "servers.json"

    cfg = ConfigFile()
    s = ServerConfig(name="A", model="/m", port=1234, custom_params="--kv-bits 4")
    store.upsert_server(cfg, s)
    store.save(cfg)

    loaded = store.load()
    assert [x.name for x in loaded.servers] == ["A"]
    assert store.find_server_by_id(s.id).port == 1234


def test_upsert_updates_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = ConfigFile()
    s = ServerConfig(name="A", model="/m")
    store.upsert_server(cfg, s)
    s2 = ServerConfig(id=s.id, name="A2", model="/m2")
    store.upsert_server(cfg, s2)
    assert len(cfg.servers) == 1 and cfg.servers[0].name == "A2"


def test_corrupt_file_is_backed_up(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = store.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    loaded = store.load()  # must not raise
    assert loaded.servers == []
    assert list(path.parent.glob("servers.corrupt-*.json"))
