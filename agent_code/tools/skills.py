"""skill 工具：skill_list（列出本地 skill 目录）、skill_load（按名加载 skill 正文）。

和 REPL 的 /skills 命令共用同一份 SkillLoader，这里只是把能力暴露给模型。
两个工具都是纯读、安全可并行——白名单见 base._READ_ONLY_TOOL_NAMES。
"""

from __future__ import annotations

from typing import Any

from ..skills import SkillLoader
from .base import ToolContext, register_tool


@register_tool(
    name="skill_list",
    description="List available local skills with their descriptions.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def skill_list(args: dict[str, Any], ctx: ToolContext) -> str:
    """给模型看的 skill 目录，和 /skills 共用一份 loader。"""
    loader = SkillLoader(ctx.cwd)
    return loader.render_list()


@register_tool(
    name="skill_load",
    description="Load the full body of a local skill by name.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name, e.g. debug-test."},
        },
        "required": ["name"],
    },
)
def skill_load(args: dict[str, Any], ctx: ToolContext) -> str:
    """按需加载 skill 正文，只返回知识，不改变当前工具白名单。"""
    name = str(args.get("name", "")).strip()
    if not name:
        return "error: missing required argument 'name'"
    skill = SkillLoader(ctx.cwd).load(name)
    if skill is None:
        return f"error: skill not found: {name}"
    return skill.body
