import json

from mlx_launcher.config.models import ServerConfig
from mlx_launcher.xcode import helpers


def test_openai_provider():
    p = helpers.openai_provider(ServerConfig(model="m", host="127.0.0.1", port=8080))
    assert p.base_url == "http://127.0.0.1:8080/v1"
    assert p.api_key  # non-empty placeholder
    assert p.port == 8080


def test_acp_registration_explicit():
    reg = helpers.acp_registration(ServerConfig(model="org/m", port=8080))
    assert "--base-url" in reg.args
    assert "http://127.0.0.1:8080/v1" in reg.args
    assert reg.args[reg.args.index("--model") + 1] == "org/m"
    obj = json.loads(reg.json_block)
    assert obj["command"] == reg.command
    assert obj["args"] == reg.args


def test_acp_registration_by_config_id():
    c = ServerConfig(model="org/m")
    reg = helpers.acp_registration(c, by_config_id=True)
    assert reg.args[reg.args.index("--config-id") + 1] == c.id
