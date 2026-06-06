from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .messages import Message
from .model import Provider, ModelResponse
from .tools import ToolRegistry, ToolContext, ToolResult
from .fs_safety import (
    SkipPolicy,
    apply_single_replace,
    check_mtime_conflict,
    ensure_read_before_edit,
    load_gitignore,
    resolve_in_cwd,
)
from rich.console import Console
from .diff_ui import confirm_edit, render_diff

console = Console()


@dataclass
class AgentResult:
    final: str
    trace: list[str]
    messages: list[Message]


def run_agent(
    prompt: str,
    provider: Provider,
    tools: ToolRegistry,
    max_steps: int = 99,
    cwd: Path | None = None,
) -> AgentResult:
    resolved_cwd = cwd or Path.cwd()
    ctx = ToolContext(
        cwd=resolved_cwd,
        skip_policy=SkipPolicy.default(gitignore=load_gitignore(resolved_cwd)),
    )

    def emit(line: str) -> None:
        # 流式输出trace： append给测试用
        trace.append(line)
        console.print(line)

    messages: list[Message] = [Message(role="user", text=prompt)]
    trace: list[str] = []

    for _step in range(max_steps):
        response = provider.complete(messages, tools=tools.list())
        messages.append(
            Message(
                role="assistant",
                text=response.text,
                tool_calls=response.tool_calls,
                provider_data=response.provider_data,
            )
        )

        if not response.tool_calls:
            final = response.text or ""
            emit(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)

        for call in response.tool_calls:
            emit(f"tool_call: {call.name} {call.arguments}")

            if call.name in ("file_write", "file_edit"):
                path_str = call.arguments.get("file_path", "")
                # 路径解析，越界 cwd 直接当作 error
                try:
                    path = resolve_in_cwd(ctx.cwd, path_str)
                except (ValueError, OSError) as exc:
                    result = ToolResult(
                        tool_call_id=call.id, content=f"error: {exc}", is_error=True
                    )
                    emit(f"observation: {result.content}")
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            content=result.content,
                            is_error=result.is_error,
                        )
                    )
                    continue
                old_content = path.read_text(encoding="utf-8") if path.exists() else ""
                # file_write前置校验，read-before-edit + mtime 冲突
                validation_error: str | None = None
                if call.name == "file_write":
                    if path.exists():
                        validation_error = ensure_read_before_edit(
                            ctx.read_state, path
                        ) or check_mtime_conflict(ctx.read_state, path)
                else:
                    if not path.exists():
                        validation_error = f"error: file does not exist: {path_str}"
                    else:
                        validation_error = ensure_read_before_edit(
                            ctx.read_state, path
                        ) or check_mtime_conflict(ctx.read_state, path)
                # 计算 new_content，file_write直接拿content; file_edit试跑替换
                new_content: str | None = None
                if call.name == "file_write":
                    new_content = call.arguments.get("content", "")
                elif call.name == "file_edit" and validation_error is None:
                    new_content, replace_err = apply_single_replace(
                        old_content,
                        call.arguments.get("old_string", ""),
                        call.arguments.get("new_string", ""),
                        bool(call.arguments.get("replace_all", False)),
                    )
                    if replace_err is not None:
                        validation_error = replace_err
                # 校验失败：不渲染diff，不问用户，直接error observation返回给模型
                if validation_error is not None:
                    result = ToolResult(call.id, validation_error, is_error=True)
                    emit(f"observation: {result.content}")
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            content=result.content,
                            is_error=result.is_error,
                        )
                    )
                    continue
                # 校验成功：渲染diff，问用户是否应用
                if new_content is not None:
                    diff_text = render_diff(old_content, new_content, path_str)
                    console.print(f"\n[bold]Diff for {path_str}:[/bold]")
                    console.print(diff_text)
                    if not confirm_edit(path_str):
                        result = ToolResult(
                            call.id, "error: edit rejected by user", is_error=True
                        )
                        emit(f"observation: {result.content}")
                        messages.append(
                            Message(
                                role="tool",
                                tool_call_id=call.id,
                                content=result.content,
                                is_error=result.is_error,
                            )
                        )
                        continue

            result = tools.run(call, ctx)
            emit(f"observation: {result.content}")
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=result.tool_call_id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
