"""待办工具：todo_write（整表覆盖）、todo_read（读取当前列表）。"""

from __future__ import annotations

from typing import Any

from ..runtime import TodoItem
from .base import ToolContext, register_tool


def _render_todos(items: list[TodoItem]) -> str:
    icon = {"pending": "○", "in_progress": "◉", "completed": "✓"}
    return (
        "\n".join(f"  {icon.get(t.status, '?')} {t.content}" for t in items)
        or "(no todos)"
    )


@register_tool(
    name="todo_write",
    description=(
        "Create and manage a structured task list. Use for multi-step tasks (3+ steps). "
        "Keep exactly ONE item in_progress. Mark completed immediately when done. "
        "The todos array is a FULL replacement—always send the entire list."
    ),
    parameters={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative task name.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "activeForm": {
                            "type": "string",
                            "description": "Present-continuous form.",
                        },
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    },
)
def todo_write(args: dict[str, Any], ctx: ToolContext) -> str:
    """整表覆盖待办版，每次调用传来的 todos 就是新列表的全部。"""
    state = ctx.runtime_state
    if state is None:
        return "error: no runtime state"
    items = [
        TodoItem(
            content=t.get("content", ""),
            status=t.get("status", "pending"),
            active_form=t.get("activeForm", ""),
        )
        for t in args.get("todos", [])
    ]
    state.todo_store = items

    lines = [_render_todos(items), "", "Todos updated."]
    ## verification nudge：本次关掉 3+ 个任务、且整张表没有任何验证项 → 提醒先验证
    completed = sum(1 for t in items if t.status == "completed")
    kws = ("test", "pytest", "verify", "lint", "check")
    has_verify = any(any(k in t.content.lower() for k in kws) for t in items)
    if completed >= 3 and not has_verify:
        lines.append(
            "提示：关掉了3+个任务但没有验证步骤，建议先加一个测试/验证项再收尾"
        )
    return "\n".join(lines)


@register_tool(
    name="todo_read",
    description="Read the current todo list.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def todo_read(args: dict[str, Any], ctx: ToolContext) -> str:
    """读取当前待办列表。"""
    state = ctx.runtime_state
    return _render_todos(state.todo_store) if state else "(no todos)"
