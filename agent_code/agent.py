from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from .prompt_ui import confirm_command, confirm_edit, confirm_tool_use, render_diff
from .permissions import PermissionsRequest, decide_permission

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
    permission_mode: Literal["default", "acceptEdits", "plan"] = "default",
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

            # 权限引擎统一入口：所有工具调用先包装成 PermissionRequest
            request = PermissionsRequest(
                tool_name=call.name,
                args=call.arguments,
                mode=permission_mode,
                cwd=ctx.cwd,
            )
            decision = decide_permission(request)

            edit_preview: tuple[str, str, str] | None = None
            if call.name in ("file_write", "file_edit") and decision.behavior != "deny":
                # acceptEdits 只跳过确认 UI，不能跳过 Day 4 的安全校验
                path_str = call.arguments.get("file_path", "")
                try:
                    path = resolve_in_cwd(ctx.cwd, path_str)
                except (ValueError, OSError) as exc:
                    result = ToolResult(call.id, f"error: {exc}", is_error=True)
                    emit(f"observation: {result.content}")
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=result.tool_call_id,
                            content=result.content,
                            is_error=True,
                        )
                    )
                    continue

                old_content = path.read_text(encoding="utf-8") if path.exists() else ""

                validation_error: str | None = None
                if call.name == "file_write":
                    if path.exists():
                        validation_error = ensure_read_before_edit(
                            ctx.read_state, path
                        ) or check_mtime_conflict(ctx.read_state, path)
                    new_content = call.arguments.get("content", "")
                else:  # file_edit
                    new_content = ""
                    if not path.exists():
                        validation_error = f"error: file does not exist: {path_str}"
                    else:
                        validation_error = ensure_read_before_edit(
                            ctx.read_state, path
                        ) or check_mtime_conflict(ctx.read_state, path)
                    if validation_error is None:
                        new_content, replace_err = apply_single_replace(
                            old_content,
                            call.arguments.get("old_string", ""),
                            call.arguments.get("new_string", ""),
                            bool(call.arguments.get("replace_all", False)),
                        )
                        if replace_err is not None:
                            validation_error = replace_err

                if validation_error is not None:
                    result = ToolResult(call.id, validation_error, is_error=True)
                    emit(f"observation: {result.content}")
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=result.tool_call_id,
                            content=result.content,
                            is_error=True,
                        )
                    )
                    continue

                edit_preview = (path_str, old_content, new_content)

            if decision.behavior == "deny":
                # deny 路径：直接返回 error observation，不弹 UI
                result = ToolResult(
                    call.id, f"error: {decision.message}", is_error=True
                )
                emit(f"observation: {result.content}")
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=result.tool_call_id,
                        content=result.content,
                        is_error=True,
                    )
                )
                continue

            elif decision.behavior == "ask":
                # ask 路径：按工具类型分发不同的预览和确认 UI
                if call.name in ("file_write", "file_edit"):
                    # --- 文件编辑：安全校验已经做过；ask 模式只负责 diff + confirm ---
                    if edit_preview is not None:
                        path_str, old_content, new_content = edit_preview
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
                                    tool_call_id=result.tool_call_id,
                                    content=result.content,
                                    is_error=True,
                                )
                            )
                            continue

                elif call.name == "bash":
                    # --- bash：命令预览 + confirm ---
                    command = call.arguments.get("command", "")
                    timeout = call.arguments.get("timeout", 30)
                    console.print(f"\n[bold yellow]Command:[/bold yellow] {command}")
                    console.print(f"[dim]timeout: {timeout}s  cwd: {ctx.cwd}[/dim]")
                    if not confirm_command(command):
                        result = ToolResult(
                            call.id, "error: command rejected by user", is_error=True
                        )
                        emit(f"observation: {result.content}")
                        messages.append(
                            Message(
                                role="tool",
                                tool_call_id=result.tool_call_id,
                                content=result.content,
                                is_error=True,
                            )
                        )
                        continue

                elif call.name in ("web_fetch", "web_search"):
                    # --- 网络工具：不写本地文件，但要让用户确认是否访问外部资源 ---
                    detail = (
                        call.arguments.get("url")
                        or call.arguments.get("query")
                        or str(call.arguments)
                    )
                    if not confirm_tool_use(call.name, detail):
                        result = ToolResult(
                            call.id, "error: tool use rejected by user", is_error=True
                        )
                        emit(f"observation: {result.content}")
                        messages.append(
                            Message(
                                role="tool",
                                tool_call_id=result.tool_call_id,
                                content=result.content,
                                is_error=True,
                            )
                        )
                        continue

                elif call.name == "ask_user_question":
                    # v3 接上
                    pass

            # allow 路径 + ask 通过：执行工具
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
