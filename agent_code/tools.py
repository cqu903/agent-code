from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console

from .model import ToolCall, ToolResult

ToolFunc = Callable[[dict[str,Any]],str]

@dataclass
class Tool:
    name:str
    description:str
    run:ToolFunc

def echo(args:dict[str,Any]) -> str:
    return str(args.get("text",""))

def uppercase(args:dict[str,Any])->str:
    return str(args.get("text","")).upper()

class ToolRegistry:
    def __init__(self) -> None:
        #注册表是工具名和python函数之间的harness边界
        self._tools:dict[str,Tool]={}

    def register(self,tool:Tool)->None:
        self._tools[tool.name]=tool

    def run(self,call:ToolCall)->ToolResult:
        #未知工具也返回observation，不让Agent Loop崩溃
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=call.id,
            content=tool.run(call.arguments),
        )

    def print_help(self, console: Console | None = None) -> None:
        out = console or Console()
        out.print("[bold]Available tools:[/bold]")
        if not self._tools:
            out.print("[dim](none)[/dim]")
            return
        for name in sorted(self._tools):
            tool = self._tools[name]
            out.print(f"  [cyan]{name}[/cyan] — {tool.description}")

def default_tools()->ToolRegistry:
    # 后续会加入文件工具和bash工具
    registry = ToolRegistry()
    registry.register(Tool(
        name="echo",
        description="Return the input text",
        run=echo
    ))
    registry.register(Tool(
        name="uppercase",
        description="Return the uppercase of the input text",
        run=uppercase
    ))
    return registry