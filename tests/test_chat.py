from mlx_launcher.chat import capabilities as cap
from mlx_launcher.chat import store
from mlx_launcher.chat.blocks import split_blocks
from mlx_launcher.chat.client import ThinkSplitter, build_openai_messages
from mlx_launcher.chat.models import Attachment, Chat, ChatMessage, Project


def test_web_search_spec():
    from mlx_launcher.chat.tools import web_search_spec

    spec = web_search_spec()
    assert spec["type"] == "function" and spec["function"]["name"] == "web_search"
    assert "query" in spec["function"]["parameters"]["properties"]


def test_mcp_slug_and_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from mlx_launcher.chat import mcp_client
    from mlx_launcher.chat.models import McpServer

    assert mcp_client.slug("My Tools!") == "My_Tools"
    data = store.load()
    srv = McpServer(name="x", command="echo", args="hi")
    store.upsert_mcp(data, srv)
    store.save(data)
    assert len(store.load().mcp_servers) == 1
    store.delete_mcp(data, srv.id)
    store.save(data)
    assert store.load().mcp_servers == []


def test_split_blocks():
    blocks = split_blocks('intro text\n\n```json\n{"a": 1}\n```\n\noutro')
    assert [b[0] for b in blocks] == ["prose", "code", "prose"]
    code = next(b for b in blocks if b[0] == "code")
    assert code[1] == "json" and '"a"' in code[2]
    assert split_blocks("just prose")[0] == ("prose", "just prose")


def test_think_splitter_across_chunks():
    sp = ThinkSplitter()
    out = []
    for chunk in ["Hello <th", "ink>thinking ", "here</thi", "nk> world"]:
        out += sp.feed(chunk)
    out += sp.flush()
    reason = "".join(t for k, t in out if k == "reason")
    content = "".join(t for k, t in out if k == "content")
    assert reason == "thinking here"
    assert content == "Hello  world"


def test_capabilities_heuristics():
    assert cap.supports_vision("mlx-community/Qwen2.5-VL-7B-Instruct")
    assert not cap.supports_vision("Qwen2.5-7B-Instruct")
    assert cap.supports_reasoning("DeepSeek-R1-Distill-Qwen-7B")
    assert not cap.supports_reasoning("Llama-3.1-8B-Instruct")
    assert cap.classify("/x/pic.PNG") == "image"
    assert cap.classify("/x/notes.md") == "text"


def test_build_messages_inlines_text_and_system(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("remember this")
    chat = Chat(model="Qwen2.5-7B")
    chat.messages.append(ChatMessage(role="user", text="summarize", attachments=[Attachment(path=str(f), name="notes.txt", kind="text")]))
    msgs = build_openai_messages(chat, Project(name="P", instructions="Be terse."))
    assert msgs[0] == {"role": "system", "content": "Be terse."}
    assert "remember this" in msgs[1]["content"]  # text attachment inlined


def test_build_messages_image_becomes_multimodal(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)  # not a real png, but has .png ext
    chat = Chat(model="Qwen2.5-VL")
    chat.messages.append(ChatMessage(role="user", text="describe", attachments=[Attachment(path=str(img), name="a.png", kind="image")]))
    content = build_openai_messages(chat)[0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_chat_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data = store.load()
    proj = Project(name="Work")
    chat = Chat(title="hi", model="m", project_id=proj.id)
    store.upsert_project(data, proj)
    store.upsert_chat(data, chat)
    store.save(data)

    again = store.load()
    assert [p.name for p in again.projects] == ["Work"]
    assert len(store.chats_in(again, proj.id)) == 1
    assert len(store.chats_in(again, None)) == 1

    store.delete_project(again, proj.id)
    assert store.get_chat(again, chat.id).project_id is None  # detached, not deleted
