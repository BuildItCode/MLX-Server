import pytest

from mlx_launcher.chat import skills
from mlx_launcher.chat.client import build_openai_messages
from mlx_launcher.chat.models import Chat, ChatMessage, Project


def test_frontmatter_inline_folded_quoted():
    fm = "name: web\ndescription: >\n  line one\n  line two\nallowed-tools: Read"
    assert skills._fm_value(fm, "name") == "web"
    assert skills._fm_value(fm, "description") == "line one line two"
    fm2 = 'name: "quoted name"\ndescription: single line value'
    assert skills._fm_value(fm2, "name") == "quoted name"
    assert skills._fm_value(fm2, "description") == "single line value"


def test_strip_frontmatter():
    text = "---\nname: a\ndescription: b\n---\n\n# Body\nhi"
    assert skills._strip_frontmatter(text).strip().startswith("# Body")
    assert skills._strip_frontmatter("no frontmatter here").strip() == "no frontmatter here"


def test_bundled_discovery_present():
    items = skills._discover(skills.ORIGIN_BUNDLED)
    names = {s.name for s in items}
    assert {"web", "cli", "ios"} <= names
    for s in items:
        assert s.origin == skills.ORIGIN_BUNDLED and not s.is_custom
        assert s.id.startswith("bundled:")


def test_custom_crud(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    s = skills.create_custom_skill("My Guide", "house style", "# Rules\nBe nice.")
    assert s.is_custom and s.id == "custom:my-guide"
    got = skills.get_skill(s.id)
    assert got is not None and got.name == "My Guide"
    assert "Be nice." in got.body()
    assert "house style" in got.description
    # name collision gets a numeric suffix, not an overwrite
    s2 = skills.create_custom_skill("My Guide", "again", "# More\nstuff")
    assert s2.id == "custom:my-guide-2"
    # update in place
    skills.update_custom_skill(s, "My Guide", "house style v2", "# Rules\nBe kind.")
    assert "Be kind." in skills.get_skill(s.id).body()
    # delete
    skills.delete_custom_skill(s)
    assert skills.get_skill(s.id) is None
    assert skills.get_skill(s2.id) is not None  # the other survives


def test_delete_or_edit_noncustom_rejected():
    items = skills._discover(skills.ORIGIN_BUNDLED)
    assert items, "expected bundled skills to exist"
    with pytest.raises(ValueError):
        skills.delete_custom_skill(items[0])
    with pytest.raises(ValueError):
        skills.update_custom_skill(items[0], "x", "y", "z")


def test_instructions_for_wraps_body():
    items = skills._discover(skills.ORIGIN_BUNDLED)
    web = next(s for s in items if s.name == "web")
    instr = skills.instructions_for(web.id)
    assert "web" in instr and "skill" in instr.lower()
    assert skills.instructions_for(None) is None
    assert skills.instructions_for("custom:does-not-exist") is None


def test_build_messages_injects_skill_and_project():
    chat = Chat(messages=[ChatMessage(role="user", text="hi")])
    proj = Project(name="p", instructions="PROJECT RULES")
    msgs = build_openai_messages(chat, proj, "SKILL GUIDANCE")
    assert msgs[0]["role"] == "system"
    assert "SKILL GUIDANCE" in msgs[0]["content"] and "PROJECT RULES" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}
    # nothing to inject → first message is the user turn
    assert build_openai_messages(chat)[0]["role"] == "user"
    # skill only (no project)
    only = build_openai_messages(chat, None, "SKILL GUIDANCE")
    assert only[0]["role"] == "system" and only[0]["content"] == "SKILL GUIDANCE"


def test_plan_mode_injects_instructions():
    from mlx_launcher.chat.client import PLAN_MODE_INSTRUCTIONS

    chat = Chat(plan_mode=True, messages=[ChatMessage(role="user", text="add a feature")])
    msgs = build_openai_messages(chat)
    assert msgs[0]["role"] == "system"
    assert "PLAN MODE" in msgs[0]["content"]
    assert PLAN_MODE_INSTRUCTIONS in msgs[0]["content"]
    # off by default → no system message
    assert build_openai_messages(Chat(messages=[ChatMessage(role="user", text="hi")]))[0]["role"] == "user"
    # plan mode rides alongside a skill + project
    proj = Project(name="p", instructions="PROJECT RULES")
    combined = build_openai_messages(chat, proj, "SKILL GUIDANCE")
    c = combined[0]["content"]
    assert "SKILL GUIDANCE" in c and "PROJECT RULES" in c and "PLAN MODE" in c


def test_bmad_targets_and_slugify():
    assert skills._slugify("My Cool Skill!! v2") == "my-cool-skill-v2"
    assert skills._slugify("") == "skill"
    assert len(skills.BMAD_SKILLS) == 9
