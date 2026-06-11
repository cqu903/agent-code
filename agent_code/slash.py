from __future__ import annotations
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class SlashContext:
    """
    slash handler 接收的运行时上下文。不暴露 provider、session 内部状态，
    只给 handler 它需要的只读信息。
    """

    cwd: Path
    permission_mode: str  # "default" / "acceptEdits" / "plan"
    model: str  # 当前模型名，如 "deepseek-v4-flash"
    provider: str  # 当前 provider，如 "anthropic"
    session_id: str | None  # 当前 session id（可能为 None）


class SlashResult:
    """
    slash command 执行结果，handled=True表示已处理，CLI不再把输入当普通prompt。
    should_query=True时CLI把prompt字段作为新的user消息喂给模型
    """

    def __init__(
        self,
        handled: bool = True,
        should_query: bool = False,
        prompt: str = "",
        message: str = "",
    ):
        self.handled = handled  # True 命令已受理
        self.should_query = should_query  # True 把prompt作为新用户输入再跑一圈
        self.prompt = prompt  # should_query=True时的模型prompt
        self.message = message  # 打给用户的终端消息（本地命令用）


# 返回 SlashResult 的 handler 签名
SlashHandler = Callable[[list[str], SlashContext], SlashResult]


@dataclass
class SlashCommand:
    """一条 slash command 的注册消息，name 不加前缀 /。"""

    name: str
    description: str  # /help 列出时显示
    handler: SlashHandler  # 实际执行函数


# 全局注册表：模块加载后所有内置命令都注册在这里
_registry: dict[str, SlashCommand] = {}


def register(name: str, description: str, handler: SlashHandler) -> None:
    """注册一条 slash command。name 不要 / 前缀"""
    _registry[name] = SlashCommand(name=name, description=description, handler=handler)


def dispatch_splash(line: str, ctx: SlashContext) -> SlashResult:
    """
    解析 "/name args" 并分派到已注册mingling，未匹配时返回 handled=False。
    """
    if not line.startswith("/"):
        return SlashResult(handled=False)
    # 去掉首字符 /，用 shlex 拆 command name 和 args，这样 --label “PR 状态轮询” 能保留空格
    try:
        parts = shlex.split(line[1:].strip())
    except ValueError as exc:
        return SlashResult(handled=False, message=f"Invalid command syntax: {exc}")
    if not parts:
        return SlashResult(handled=False)
    name = parts[0]
    args = parts[1:]
    cmd = _registry.get(name)
    if cmd is None:
        return SlashResult(handled=True, message=f"Unknown command: {name}")
    return cmd.handler(args, ctx)
