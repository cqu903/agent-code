"""bash 工具：执行 shell 命令（含后台执行）。"""

from __future__ import annotations

from typing import Any

from .base import ToolContext, register_tool
from ..bash_runner import run_sync as _bash_run_sync


@register_tool(
    name="bash",
    description=(
        "Execute a shell command. Working directory persists but shell state "
        "does not (each call is a fresh shell). timeout in seconds (default 30). "
        "Avoid cd; use the tool's implicit cwd instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds, default 30.",
                "default": 30,
            },
            "background": {
                "type": "boolean",
                "description": "Run in background. Returns immediately with a background_id. Default false.",
                "default": False,
            },
        },
        "required": ["command"],
    },
)
def bash(args: dict[str, Any], ctx: ToolContext) -> str:
    """执行 shell 命令，前置校验和用户确认在agent.py拦截块完成"""
    command = args.get("command", "")
    if not command:
        return "error: missing required argument 'command'"
    timeout = args.get("timeout", 30)
    background = bool(args.get("background", False))

    # v1只做同步，v4接background=True分支
    if background:
        # 后台执行：启动子进程后立即返回结构化信息，不阻塞 Agent Loop
        from ..bg_manager import start_background

        result = start_background(command, ctx.cwd)
        return (
            f"Command running in background with ID: {result['background_id']}.\n"
            f"Output is being written to: {result['output_file']}\n"
            f"Stderr is being written to: {result['stderr_file']}\n"
            f"PID: {result['pid']}\n\n"
            f"{result['message']}"
        )
    return _bash_run_sync(command, ctx.cwd, timeout=timeout)
