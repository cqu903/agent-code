from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from ._frontmatter import _split_frontmatter, _unquote


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    allowed_tools: list[str] | None
    body: str
    path: Path


def _parse_allowed_tools(raw: str | None) -> list[str] | None:
    """三态：字段缺失=None；[]=禁止工具；[a,b]=只允许列表。"""
    if raw is None:
        return None
    value = raw.strip()
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_unquote(part.strip()) for part in inner.split(",") if part.strip()]
    return [_unquote(value)]


class SkillLoader:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.skills_dir = cwd / ".agent" / "skills"
        self.warnings: list[str] = []

    def list(self) -> list[SkillMeta]:
        skills: list[SkillMeta] = []
        if not self.skills_dir.exists():
            return skills
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = self._load_file(skill_md)
            if skill is not None:
                skills.append(skill)
        return skills

    def load(self, name: str) -> SkillMeta | None:
        for skill in self.list():
            if skill.name == name:
                return skill
        return None

    def render_list(self) -> str:
        skills = self.list()
        if not skills:
            return "(no skills found)"
        return "\n".join(f"{skill.name}  {skill.description}" for skill in skills)

    def render_available_skills(self) -> str:
        skills = self.list()
        if not skills:
            return ""
        lines = ["<available-skills>"]
        lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
        lines.append("</available-skills>")
        return "\n".join(lines)

    def _load_file(self, path: Path) -> SkillMeta | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.warnings.append(f"{path}: {exc}")
            return None
        fields, body = _split_frontmatter(text)
        name = _unquote(fields.get("name", "")).strip()
        description = _unquote(fields.get("description", "")).strip()
        if not name or not description:
            self.warnings.append(f"{path}: missing name or description")
            return None
        return SkillMeta(
            name=name,
            description=description,
            allowed_tools=_parse_allowed_tools(fields.get("allowed_tools")),
            body=body,
            path=path,
        )
