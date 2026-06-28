"""工具基础设施：Tool 数据模型、注册装饰器、ToolRegistry、default_tools 收集器。

本模块是依赖叶子——绝不 import 任何具体工具模块，只依赖 stdlib + rich + fs_safety。
具体工具模块（fs/web/git/...）反向依赖这里，由 tools/__init__.py 顶部统一 import 触发注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from ..fs_safety import ReadFileState, SkipPolicy


@dataclass
class ToolContext:
    # 工具运行时上下文，装cwd，skip规则，ReadFileState
    cwd: Path
    skip_policy: SkipPolicy = field(default_factory=SkipPolicy.default)
    read_state: ReadFileState = field(default_factory=ReadFileState)


ToolFunc = Callable[[dict[str, Any], ToolContext], str]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


_REGISTERED_TOOLS: list[Tool] = []


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[ToolFunc], ToolFunc]:
    def decorator(fn: ToolFunc) -> ToolFunc:
        _REGISTERED_TOOLS.append(
            Tool(
                name=name,
                description=description,
                run=fn,
                parameters=parameters
                or {"type": "object", "properties": {}, "required": []},
            )
        )
        return fn

    return decorator


class ToolRegistry:
    def __init__(self) -> None:
        # 注册表是工具名和python函数之间的harness边界
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        # 未知工具也返回observation，不让Agent Loop崩溃
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=call.id,
            content=tool.run(call.arguments, ctx),
        )

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def print_help(self, console: Console | None = None) -> None:
        out = console or Console()
        out.print("[bold]Available tools:[/bold]")
        if not self._tools:
            out.print("[dim](none)[/dim]")
            return
        for name in sorted(self._tools):
            tool = self._tools[name]
            out.print(f"  [cyan]{name}[/cyan] — {tool.description}")


def default_tools() -> ToolRegistry:
    # tools/__init__.py 顶部已 import 所有工具模块，触发各自的 @register_tool，
    # 调到这里时 _REGISTERED_TOOLS 已填满——纯收集，无需任何延迟导入。
    registry = ToolRegistry()
    for tool in _REGISTERED_TOOLS:
        registry.register(tool)
    return registry
