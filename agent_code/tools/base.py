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
from ..runtime import RuntimeState


@dataclass
class ToolContext:
    # 工具运行时上下文，装cwd，skip规则，ReadFileState
    cwd: Path
    skip_policy: SkipPolicy = field(default_factory=SkipPolicy.default)
    read_state: ReadFileState = field(default_factory=ReadFileState)
    runtime_state: RuntimeState | None = None


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
    is_read_only: bool = False


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

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


# 只读工具白名单：无副作用、可安全并行执行。
# 其余（写类 / 交互类 / 网络类）保持 is_read_only=False，串行执行。
_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "list_files",
        "glob",
        "grep",
        "project_tree",
        "git_status",
        "git_diff",
        "system_date",
        "echo",
        "memory_recall",
        "cron_list",
    }
)


def default_tools() -> ToolRegistry:
    # tools/__init__.py 顶部已 import 所有工具模块，触发各自的 @register_tool，
    # 调到这里时 _REGISTERED_TOOLS 已填满——纯收集，无需任何延迟导入。
    # 并发工具编排：白名单内的工具打 is_read_only=True，供上层判定可否并行。
    registry = ToolRegistry()
    for tool in _REGISTERED_TOOLS:
        if tool.name in _READ_ONLY_TOOL_NAMES:
            tool.is_read_only = True
        registry.register(tool)
    return registry
