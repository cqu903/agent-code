"""plan 模式工具：enter_plan_mode（进入计划模式）、exit_plan_mode（提交计划等批准）。

exit_plan_mode 函数体很薄——渲染计划、等批准、翻模式都在 agent.py 的拦截块里做；
这里只负责把工具注册进 _REGISTERED_TOOLS，让模型能调用。
"""

from __future__ import annotations

from typing import Any

from .base import ToolContext, register_tool


@register_tool(
    name="enter_plan_mode",
    description=(
        "Enter plan mode: draft a plan before writing. Write tools are denied until approval. "
        "When the plan is ready, call exit_plan_mode(plan_summary). Do not ask for approval in final text."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def enter_plan_mode(args: dict[str, Any], ctx: ToolContext) -> str:
    """模型主动请求进 plan 模式。"""
    state = ctx.runtime_state
    if state is None:
        return "error: no runtime state"
    state.permission_mode = "plan"
    return (
        "Plan mode on. Draft a plan—write tools are denied. "
        "When the plan is ready, you MUST call exit_plan_mode(plan_summary). "
        "Do not ask for approval in final text."
    )


@register_tool(
    name="exit_plan_mode",
    description=(
        "Submit your plan for user approval. Use this when the plan is ready. "
        "Write tools unlock only after the user approves."
    ),
    parameters={
        "type": "object",
        "properties": {
            "plan_summary": {"type": "string", "description": "The plan to review."}
        },
        "required": ["plan_summary"],
    },
)
def exit_plan_mode(args: dict[str, Any], ctx: ToolContext) -> str:
    """函数体很薄——渲染计划、等批准、翻模式都在 agent.py 的拦截块里做。"""
    return "Plan approved. Write tools are now enabled."
