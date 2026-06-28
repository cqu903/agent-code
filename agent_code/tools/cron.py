"""cron 工具：cron_create、cron_list、cron_cancel + scheduler 接线。"""

from __future__ import annotations

from typing import Any

from .base import ToolContext, register_tool
from ..scheduler import CronScheduler

# 全局 scheduler 单例——由 cli.py 在启动 REPL 时设置
_scheduler: Any = None


def set_scheduler(scheduler: Any) -> None:
    """cli.py 在创建 CronScheduler 后调用这个函数，让工具函数能访问同一个实例。"""
    global _scheduler
    _scheduler = scheduler


def _get_scheduler(ctx: ToolContext) -> CronScheduler:
    """REPL 里复用正在运行的 scheduler；一次性模式临时读写 cron.json。"""
    if _scheduler is not None:
        return _scheduler
    return CronScheduler(ctx.cwd)


@register_tool(
    name="cron_create",
    description=(
        "Create a recurring cron job. The job will re-run the given slash/prompt "
        "every N seconds. Use for periodic checks like PR status polling."
    ),
    parameters={
        "type": "object",
        "properties": {
            "slash": {
                "type": "string",
                "description": "Slash command or prompt to run.",
            },
            "every_seconds": {"type": "integer", "description": "Interval in seconds."},
            "label": {
                "type": "string",
                "description": "Optional human-readable label.",
            },
        },
        "required": ["slash", "every_seconds"],
    },
)
def cron_create(args: dict[str, Any], ctx: ToolContext) -> str:
    """创建一条 cron job"""
    scheduler = _get_scheduler(ctx)
    slash = args.get("slash", "")
    every_seconds = int(args.get("every_seconds", 0))
    label = args.get("label", "")
    if not slash:
        return "error: missing required argument: 'slash'"
    if every_seconds <= 0:
        return "error: every_seconds must be positive"
    job = scheduler.add_job(slash, every_seconds, label)
    return f"Cron job created: {job.id} - every {every_seconds}s: {slash}"


@register_tool(
    name="cron_list",
    description="List all active cron jobs with their IDs, intervals, and last-run times.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def cron_list(args: dict[str, Any], ctx: ToolContext) -> str:
    """列出当前所有cron job"""
    scheduler = _get_scheduler(ctx)
    jobs = scheduler.list_jobs()
    if not jobs:
        return "(no cron jobs)"
    lines = []
    for j in jobs:
        last = j.last_run_at or "never"
        label = f" - {j.label}" if j.label else ""
        lines.append(
            f"  [{j.id}] every {j.every_seconds}s: {j.slash}{label}  (last: {last})"
        )
    return "\n".join(lines)


@register_tool(
    name="cron_cancel",
    description="Cancel a cron job by its ID.",
    parameters={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Cron job ID to cancel."},
        },
        "required": ["id"],
    },
)
def cron_cancel(args: dict[str, Any], ctx: ToolContext) -> str:
    """取消一条 cron job"""
    scheduler = _get_scheduler(ctx)
    jid = args.get("id", "")
    if not jid:
        return "error: missing required argument 'id'"
    if scheduler.cancel_job(jid):
        return f"Cron job cancelled: {jid}"
    return f"error: job not found: {jid}"
