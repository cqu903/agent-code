"""git 只读工具：git_status、git_diff。"""

from __future__ import annotations

from typing import Any

from .base import ToolContext, register_tool
from ..bash_runner import run_sync as _bash_run_sync


@register_tool(
    name="git_status",
    description="Run git status to see the current state of the working directory.",
)
def _git_status(args: dict[str, Any], ctx: ToolContext) -> str:
    """薄包装 git status 只读，默认 allow"""
    return _bash_run_sync("git status", ctx.cwd, timeout=10)


@register_tool(
    name="git_diff",
    description="Run git diff to see unstaged changes in the working directory.",
)
def _git_diff(args: dict[str, Any], ctx: ToolContext) -> str:
    """薄包装 git diff——只读、默认 allow。"""
    return _bash_run_sync("git diff", ctx.cwd, timeout=10)
