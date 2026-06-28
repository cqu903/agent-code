"""杂项工具：echo、system_date、ask_user_question。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import ToolContext, register_tool


@register_tool(
    name="echo",
    description="Return the input text",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to return."}},
        "required": ["text"],
    },
)
def echo(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", ""))


@register_tool(
    name="system_date",
    description="Return the current system date and time.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional IANA timezone name (e.g. UTC, Asia/Shanghai, America/New_York). Defaults to the system local timezone.",
            }
        },
        "required": [],
    },
)
def system_date(args: dict[str, Any], ctx: ToolContext) -> str:
    tz_time = args.get("timezone")
    if tz_time:
        try:
            tz = ZoneInfo(tz_time)
        except ZoneInfoNotFoundError:
            return f"unknown timezone: {tz_time}"
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@register_tool(
    name="ask_user_question",
    description=(
        "Ask the user a structured single-choice question. "
        "Use when you need to decide between multiple approaches "
        "or need user preference before proceeding."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The question to ask the user. Should end with ?.",
            },
            "options": {
                "type": "array",
                "description": "List of option labels (2-4 recommended).",
                "items": {"type": "string"},
            },
        },
        "required": ["prompt", "options"],
    },
)
def _ask_user_question(args: dict[str, Any], ctx: ToolContext) -> str:
    """由 agent.py 拦截块处理——工具函数本身不读 stdin。
    拦截块识别 call.name == "ask_user_question"，调 prompt_ui 后把结果作为 observation 返回。"""
    prompt = args.get("prompt", "")
    options = args.get("options", [])
    if not prompt:
        return "error: missing required argument 'prompt'"
    if not options or not isinstance(options, list):
        return "error: options must be a non-empty list"
    # 实际交互在 agent.py 拦截块里完成——这里只返回占位。
    # 正常路径不会走到这里，因为拦截块会先处理。
    return (
        "error: ask_user_question must be handled by the harness, not executed directly"
    )
