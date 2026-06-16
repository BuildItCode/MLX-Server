import json

import httpx
import respx
from acp import text_block
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    FileSystemCapabilities,
    ReadTextFileResponse,
    RequestPermissionResponse,
    WriteTextFileResponse,
)

from mlx_launcher.acp.agent import MlxAcpAgent

BASE = "http://127.0.0.1:8080/v1"
CHAT = BASE + "/chat/completions"


class FakeClient:
    """Captures the session_update notifications the agent emits."""

    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))


def _sse(*chunks: str) -> str:
    body = ""
    for c in chunks:
        body += f"data: {c}\n\n"
    return body + "data: [DONE]\n\n"


async def _new_agent():
    agent = MlxAcpAgent(BASE, "test-model")
    client = FakeClient()
    agent.on_connect(client)
    init = await agent.initialize(1)
    assert init.protocol_version == 1
    ns = await agent.new_session("/tmp")
    return agent, client, ns.session_id


async def test_prompt_streams_message_chunks():
    with respx.mock:
        respx.post(CHAT).mock(
            return_value=httpx.Response(
                200,
                text=_sse(
                    '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                    '{"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
                ),
            )
        )
        agent, client, sid = await _new_agent()
        resp = await agent.prompt([text_block("hi")], sid)

    assert resp.stop_reason == "end_turn"
    texts = [u.content.text for _, u in client.updates]
    assert "".join(texts) == "Hello world"


async def test_prompt_rejects_unknown_session():
    import acp
    import pytest

    agent = MlxAcpAgent(BASE, "test-model")
    agent.on_connect(FakeClient())
    await agent.initialize(1)
    # a prompt for a session that was never created is a protocol error, not a silent
    # empty-history turn
    with pytest.raises(acp.RequestError):
        await agent.prompt([text_block("hi")], "never-created")


async def test_initialize_negotiates_protocol_version_down():
    import acp

    agent = MlxAcpAgent(BASE, "test-model")
    # a client announcing a newer protocol gets our (older) version back, not a higher claim
    init = await agent.initialize(999)
    assert init.protocol_version == acp.PROTOCOL_VERSION


async def test_finish_reason_length_maps_to_max_tokens():
    with respx.mock:
        respx.post(CHAT).mock(
            return_value=httpx.Response(
                200,
                text=_sse('{"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}'),
            )
        )
        agent, client, sid = await _new_agent()
        resp = await agent.prompt([text_block("hi")], sid)
    assert resp.stop_reason == "max_tokens"


async def test_unreachable_server_is_reported_not_raised():
    with respx.mock:
        respx.post(CHAT).mock(side_effect=httpx.ConnectError("refused"))
        agent, client, sid = await _new_agent()
        resp = await agent.prompt([text_block("hi")], sid)
    # graceful: a friendly chunk is emitted and the turn ends
    assert resp.stop_reason == "end_turn"
    assert any("could not reach" in u.content.text.lower() for _, u in client.updates)


# --- agentic tool-calling ----------------------------------------------------


def _tool_call_msg(name, args, call_id="c1"):
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )


def _final_msg(text):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


class FakeFsClient:
    def __init__(self, allow=True):
        self.updates = []
        self.reads = []
        self.writes = []
        self._allow = allow

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)

    async def read_text_file(self, path, session_id, **kwargs):
        self.reads.append(path)
        return ReadTextFileResponse(content="print('hi')\n")

    async def write_text_file(self, content, path, session_id, **kwargs):
        self.writes.append((path, content))
        return WriteTextFileResponse()

    async def request_permission(self, options, session_id, tool_call, **kwargs):
        opt = "allow" if self._allow else "reject"
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=opt, outcome="selected"))


async def _agentic_agent(client):
    agent = MlxAcpAgent(BASE, "m", use_tools=True)
    agent.on_connect(client)
    await agent.initialize(
        1, client_capabilities=ClientCapabilities(fs=FileSystemCapabilities(read_text_file=True, write_text_file=True))
    )
    ns = await agent.new_session("/tmp")
    return agent, ns.session_id


async def test_agentic_read_file():
    with respx.mock:
        respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, text=_tool_call_msg("read_file", {"path": "/a.py"})),
                httpx.Response(200, text=_final_msg("The file prints hi.")),
            ]
        )
        client = FakeFsClient()
        agent, sid = await _agentic_agent(client)
        resp = await agent.prompt([text_block("read /a.py")], sid)

    assert resp.stop_reason == "end_turn"
    assert client.reads == ["/a.py"]
    kinds = [getattr(u, "session_update", None) for u in client.updates]
    assert "tool_call" in kinds and "tool_call_update" in kinds
    msgs = [u.content.text for u in client.updates if getattr(u, "session_update", None) == "agent_message_chunk"]
    assert any("prints hi" in m for m in msgs)


async def test_agentic_write_file_allowed():
    with respx.mock:
        respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, text=_tool_call_msg("write_file", {"path": "/b.py", "content": "x=1\n"})),
                httpx.Response(200, text=_final_msg("done")),
            ]
        )
        client = FakeFsClient(allow=True)
        agent, sid = await _agentic_agent(client)
        resp = await agent.prompt([text_block("write it")], sid)

    assert resp.stop_reason == "end_turn"
    assert client.writes == [("/b.py", "x=1\n")]


async def test_agentic_write_file_denied():
    with respx.mock:
        respx.post(CHAT).mock(
            side_effect=[
                httpx.Response(200, text=_tool_call_msg("write_file", {"path": "/b.py", "content": "x=1\n"})),
                httpx.Response(200, text=_final_msg("ok, skipped")),
            ]
        )
        client = FakeFsClient(allow=False)
        agent, sid = await _agentic_agent(client)
        await agent.prompt([text_block("write it")], sid)

    assert client.writes == []  # permission denied → no write
