"""Discover, parse, create and install *skills* — `SKILL.md` instruction files
that can be injected into a chat as system-prompt guidance.

A skill is a folder containing a `SKILL.md`: YAML-ish frontmatter (`name`,
`description`) followed by a Markdown instruction body. Three origins:

  - ``bundled`` — ships with the app (the repo's ``./skills`` tree); read-only.
  - ``custom``  — user-created, under ``~/.config/mlx-launcher/skills``; editable.
  - ``bmad``    — the BMAD-METHOD agent skills, installed under ``…/skills-bmad``.

Skill ids are ``"<origin>:<slug>"`` and are stable, so a chat can persist which
skill it uses. We parse frontmatter by hand (no PyYAML dependency)."""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import bootstrap
from ..config.store import config_dir

ORIGIN_BUNDLED = "bundled"
ORIGIN_CUSTOM = "custom"
ORIGIN_BMAD = "bmad"

# BMAD-METHOD skills, in Claude-Code SKILL.md format (community port).
_BMAD_BASE = "https://raw.githubusercontent.com/aj-geddes/claude-code-bmad-skills/main/bmad-skills"
BMAD_SKILLS = [
    "bmad-orchestrator",
    "business-analyst",
    "product-manager",
    "system-architect",
    "developer",
    "scrum-master",
    "ux-designer",
    "builder",
    "creative-intelligence",
]

LogCb = Callable[[str], None]


@dataclass
class Skill:
    id: str  # "<origin>:<slug>"
    name: str
    description: str
    origin: str
    path: Path  # the SKILL.md file

    @property
    def is_custom(self) -> bool:
        return self.origin == ORIGIN_CUSTOM

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def body(self) -> str:
        """The Markdown instruction body (frontmatter stripped)."""
        return _strip_frontmatter(self.read()).strip()

    def instructions(self) -> str:
        """The system-prompt text injected when this skill is active."""
        body = self.body()
        preamble = f"You are operating with the “{self.name}” skill active. Follow this guidance for the conversation:"
        return f"{preamble}\n\n{body}" if body else preamble


# --- roots ---------------------------------------------------------------

def _bundled_root() -> Optional[Path]:
    """The shipped skills, found wherever the package is installed.

    The skills live *inside* the package (`mlx_launcher/skills`), so they survive
    a global/pipx install (where there's no source checkout). Resolved via
    importlib.resources, which works for both editable and wheel installs."""
    env = os.environ.get("MLX_SKILLS_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        from importlib.resources import files

        packaged = Path(str(files("mlx_launcher"))) / "skills"
        if packaged.is_dir():
            return packaged
    except (ModuleNotFoundError, TypeError, OSError):
        pass
    # last resort: a pre-packaging source layout with a repo-root ./skills
    root = bootstrap.project_root()
    if root is not None and (root / "skills").is_dir():
        return root / "skills"
    return None


def custom_root() -> Path:
    return config_dir() / "skills"


def bmad_root() -> Path:
    return config_dir() / "skills-bmad"


def _root_for(origin: str) -> Optional[Path]:
    return {
        ORIGIN_BUNDLED: _bundled_root(),
        ORIGIN_CUSTOM: custom_root(),
        ORIGIN_BMAD: bmad_root(),
    }.get(origin)


# --- frontmatter parsing -------------------------------------------------

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[str, str]:
    m = _FM_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def _strip_frontmatter(text: str) -> str:
    return _split_frontmatter(text)[1]


def _fm_value(fm: str, key: str) -> str:
    """Read one scalar from frontmatter — handles inline, quoted, and YAML
    folded/literal block scalars (``key: >`` / ``key: |`` + indented lines)."""
    lines = fm.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^{re.escape(key)}\s*:\s*(.*)$", line)
        if not m:
            continue
        val = m.group(1).strip()
        if val in ("", ">", "|", ">-", "|-", ">+", "|+"):
            block: list[str] = []
            for cont in lines[i + 1:]:
                if cont.strip() == "":
                    continue
                if re.match(r"^\s+\S", cont):  # an indented continuation line
                    block.append(cont.strip())
                else:
                    break
            return " ".join(block).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            return val[1:-1]
        return val
    return ""


def _parse_meta(md: Path) -> tuple[str, str]:
    try:
        fm, _ = _split_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return "", ""
    return _fm_value(fm, "name"), _fm_value(fm, "description")


# --- discovery -----------------------------------------------------------

def _discover(origin: str) -> list[Skill]:
    root = _root_for(origin)
    if root is None or not root.is_dir():
        return []
    out: list[Skill] = []
    for md in sorted(root.rglob("SKILL.md")):
        slug = "/".join(md.parent.relative_to(root).parts) or md.parent.name
        name, desc = _parse_meta(md)
        out.append(Skill(id=f"{origin}:{slug}", name=name or slug, description=desc, origin=origin, path=md))
    return out


def all_skills() -> list[Skill]:
    """Every available skill, bundled → bmad → custom."""
    return _discover(ORIGIN_BUNDLED) + _discover(ORIGIN_BMAD) + _discover(ORIGIN_CUSTOM)


def get_skill(skill_id: Optional[str]) -> Optional[Skill]:
    if not skill_id:
        return None
    return next((s for s in all_skills() if s.id == skill_id), None)


def instructions_for(skill_id: Optional[str]) -> Optional[str]:
    s = get_skill(skill_id)
    return s.instructions() if s else None


# --- custom skill CRUD ---------------------------------------------------

def _slugify(name: str) -> str:
    # transliterate accents (café → cafe) so non-ASCII names get readable slugs
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")
    return s or "skill"


def _render_skill_md(name: str, description: str, body: str) -> str:
    desc = " ".join(description.split())  # frontmatter description stays single-line
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n{body.strip()}\n"


def create_custom_skill(name: str, description: str, body: str) -> Skill:
    root = custom_root()
    slug = _slugify(name)
    dest = root / slug
    n = 2
    while dest.exists():
        dest = root / f"{slug}-{n}"
        n += 1
    dest.mkdir(parents=True, exist_ok=True)
    md = dest / "SKILL.md"
    md.write_text(_render_skill_md(name, description, body), encoding="utf-8")
    return Skill(
        id=f"{ORIGIN_CUSTOM}:{dest.name}",
        name=name,
        description=" ".join(description.split()),
        origin=ORIGIN_CUSTOM,
        path=md,
    )


def update_custom_skill(skill: Skill, name: str, description: str, body: str) -> None:
    if skill.origin != ORIGIN_CUSTOM:
        raise ValueError("only custom skills can be edited")
    skill.path.write_text(_render_skill_md(name, description, body), encoding="utf-8")


def delete_custom_skill(skill: Skill) -> None:
    if skill.origin != ORIGIN_CUSTOM:
        raise ValueError("only custom skills can be deleted")
    shutil.rmtree(skill.path.parent, ignore_errors=True)


# --- BMAD install --------------------------------------------------------

def bmad_installed() -> bool:
    root = bmad_root()
    return root.is_dir() and any(root.rglob("SKILL.md"))


async def install_bmad(on_log: LogCb) -> int:
    """Download the BMAD skills into the bmad root. Returns the count installed."""
    import httpx

    root = bmad_root()
    root.mkdir(parents=True, exist_ok=True)
    ok = 0
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for role in BMAD_SKILLS:
            url = f"{_BMAD_BASE}/{role}/SKILL.md"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                dest = root / role / "SKILL.md"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(resp.text, encoding="utf-8")
                on_log(f"✓ {role}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                on_log(f"✗ {role}: {exc}")
    on_log(f"Installed {ok}/{len(BMAD_SKILLS)} BMAD skills")
    return ok
