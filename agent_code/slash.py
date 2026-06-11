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
