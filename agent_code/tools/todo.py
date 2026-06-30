from typing import Any

from agent_code.runtime import TodoItem
from agent_code.tools import ToolContext


def _render_todos(items: list[TodoItem]) -> str:
    icon = {"pending": "○", "in_progress": "◉", "completed": "✓"}
    return (
        "\n".join(f"  {icon.get(t.status, '?')} {t.content}" for t in items)
        or "(no todos)"
    )


def todo_write(args: dict[str, Any], ctx: ToolContext) -> str:
    """整表覆盖待办版，每次调用传来的todos就是新列表的全部"""
    state = ctx.runtime_state
    if state is None:
        return "error: no runtime state"
    items = [
        TodoItem(
            content=t.get("content", ""),
            status=t.get("status", "pending"),
            active_form=t.get("active_form", ""),
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


def todo_read(args: dict[str, Any], ctx: ToolContext) -> str:
    state = ctx.runtime_state
    return _render_todos(state.todo_store) if state else "(no todos)"
