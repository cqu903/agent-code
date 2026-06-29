from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .messages import Message
from .model import Provider, ModelResponse
from .runtime import RuntimeState
from .tools import ToolRegistry, ToolContext, ToolCall
from .fs_safety import (
    SkipPolicy,
    apply_single_replace,
    check_mtime_conflict,
    ensure_read_before_edit,
    load_gitignore,
    resolve_in_cwd,
)
from rich.console import Console
from .prompt_ui import (
    confirm_command,
    confirm_edit,
    confirm_tool_use,
    render_diff,
    prompt_single_choice,
)
from .permissions import PermissionsRequest, decide_permission
from .session import Session
from .project_memory import load_agent_md
from .compact_basic import compact
from .hooks import run_hooks, run_hooks_raw

console = Console()


@dataclass
class AgentResult:
    final: str
    trace: list[str]
    messages: list[Message]


_SYSTEM_CORE = (
    "You are an AI coding agent running inside a CLI harness. "
    "You have access to tools for reading/writing files, running shell commands, "
    "searching the web, and asking the user questions. "
    "Use tools when needed; respond directly when you can."
)


def build_system_prompt(cwd: Path) -> str:
    """
    组装 system prompt：核心指南 + AGENT.md + MEMORY.md 索引。
    注入顺序：core prompt → 项目规则 → 跨 session 记忆索引。
    """
    from .memdir.store import load_index as load_memory_index

    parts: list[str] = [_SYSTEM_CORE]
    agent_md = load_agent_md(cwd)
    if agent_md:
        parts.append(agent_md)

    memory_index = load_memory_index(cwd)
    if memory_index:
        parts.append(f"<project-memory>\n{memory_index}\n</project-memory>")

    return "\n\n".join(parts)


def _format_call_args(args: dict[str, Any]) -> str:
    """trace 里的工具参数可能很大（file_write 的整段内容）。
    长字符串只截断到 80 字符做预览，完整参数仍照常传给工具。"""
    preview: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 80:
            preview[key] = value[:80] + "…"
        else:
            preview[key] = value
    return str(preview)


def partition_tool_calls(
    calls: list[ToolCall], tools: ToolRegistry
) -> list[list[ToolCall]]:
    """连续只读工具合成并行组；写 / 未知工具截断、自成单元素组。
    例：[Read, Read, Write, Read] → [[Read, Read], [Write], [Read]]
    未知工具（tools.get 为 None）fail-closed 当串行，不并入只读组。"""
    batches: list[list[ToolCall]] = []
    current: list[ToolCall] = []
    for call in calls:
        tool = tools.get(call.name)
        if tool is not None and tool.is_read_only:
            current.append(call)
        else:
            if current:  # 写 / 未知工具前先收掉前面攒的只读组
                batches.append(current)
                current = []
            batches.append([call])  # 写 / 未知工具单独一组，串行
    if current:
        batches.append(current)
    return batches


def execute_one_tool_call(
    call: ToolCall,
    ctx: ToolContext,
    state: RuntimeState,
    tools: ToolRegistry,
    emit: Callable[[str], None],
) -> Message:
    """跑单个工具，返回一条 Message(role="tool", ...)。

    内层 for call 循环体原样搬出：每处原来的 append(Message); continue 换成
    return Message，逻辑一行不改。并发编排时，只读组的多个调用会经
    ThreadPoolExecutor 并行进入这里——所以函数内不得触碰跨调用共享的可变状态
    （ctx.read_state 已加锁、trace.append 在 GIL 下原子，均安全）。
    """
    emit(f"tool_call: {call.name} {_format_call_args(call.arguments)}")

    # 权限引擎统一入口：所有工具调用先包装成 PermissionRequest
    request = PermissionsRequest(
        tool_name=call.name,
        args=call.arguments,
        mode=state.permission_mode,
        cwd=ctx.cwd,
    )
    decision = decide_permission(request)

    # plan模式等deny决策已经在上面算出，deny不再执行本地hook，避免hook副作用
    if decision.behavior != "deny":
        pre_hooks = run_hooks("PreToolUse", call.name, call.arguments, ctx.cwd)
        pre_blocked = [h for h in pre_hooks if not h["success"]]
        if pre_blocked:
            blocked_msgs = "\n".join(
                f" [hook] {h['command']}: {h['output']}" for h in pre_blocked
            )
            observation = f"tool blocked by PreToolUse hook:\n{blocked_msgs}"
            emit(f"observation: {observation}")
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=observation,
                is_error=True,
            )

    edit_preview: tuple[str, str, str] | None = None
    if call.name in ("file_write", "file_edit") and decision.behavior != "deny":
        # acceptEdits 只跳过确认 UI，不能跳过 Day 4 的安全校验
        path_str = call.arguments.get("file_path", "")
        if not path_str:
            return Message(
                role="tool",
                tool_call_id=call.id,
                content="error: missing required argument 'file_path'",
                is_error=True,
            )
        try:
            path = resolve_in_cwd(ctx.cwd, path_str)
        except (ValueError, OSError) as exc:
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=f"error: {exc}",
                is_error=True,
            )
        if path.is_dir():
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=f"error: path is a directory: {path_str}",
                is_error=True,
            )

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
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=validation_error,
                is_error=True,
            )

        edit_preview = (path_str, old_content, new_content)

    if decision.behavior == "deny":
        # deny 路径：直接返回 error observation，不弹 UI
        return Message(
            role="tool",
            tool_call_id=call.id,
            content=f"error: {decision.message}",
            is_error=True,
        )

    if decision.behavior == "ask":
        # ask 路径：按工具类型分发不同的预览和确认 UI。
        # 仅在用户拒绝时早 return；通过确认则落到末尾 tools.run() 统一执行。
        if call.name in ("file_write", "file_edit"):
            # --- 文件编辑：安全校验已经做过；ask 模式只负责 diff + confirm ---
            # diff 预览放进 confirm_edit 内部打印（和 bash 同理：预览要和确认
            # 问题一起在 in_terminal 里直写真实终端，否则卡在 patch_stdout）。
            if edit_preview is not None:
                path_str, old_content, new_content = edit_preview
                diff_text = render_diff(old_content, new_content, path_str)
                if not confirm_edit(path_str, diff_text=diff_text):
                    return Message(
                        role="tool",
                        tool_call_id=call.id,
                        content="error: edit rejected by user",
                        is_error=True,
                    )

        elif call.name == "bash":
            # --- bash：命令预览 + confirm ---
            # 预览（Command: / timeout）放进 confirm_command 内部打印，和确认
            # 问题一起经 run_on_main_terminal → in_terminal 直写真实终端，
            # 避免预览卡在 patch_stdout 代理里、排到用户回答之后才出现。
            command = call.arguments.get("command", "")
            timeout = call.arguments.get("timeout", 30)
            if not confirm_command(command, timeout=timeout, cwd=ctx.cwd):
                return Message(
                    role="tool",
                    tool_call_id=call.id,
                    content="error: command rejected by user",
                    is_error=True,
                )

        elif call.name in ("web_fetch", "web_search"):
            # --- 网络工具：不写本地文件，但要让用户确认是否访问外部资源 ---
            detail = (
                call.arguments.get("url")
                or call.arguments.get("query")
                or str(call.arguments)
            )
            if not confirm_tool_use(call.name, detail):
                return Message(
                    role="tool",
                    tool_call_id=call.id,
                    content="error: tool use rejected by user",
                    is_error=True,
                )

        elif call.name == "ask_user_question":
            # ask_user_question 的"结果"就是用户选择，内联算出后直接 return，
            # 不落到末尾 tools.run()。
            question = call.arguments.get("prompt", "")
            options = call.arguments.get("options", [])
            if not isinstance(options, list):
                options = []
            labels = [str(o) for o in options]
            selected = prompt_single_choice(question, labels)
            if selected is None:
                content = "User skipped the question."
            else:
                content = f'User selected: "{selected}"'
            emit(f"observation: {content}")
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=content,
                is_error=False,
            )

    # allow 路径 + ask 通过：执行工具
    result = tools.run(call, ctx)
    emit(f"observation: {result.content}")

    # PostToolUse hooks - 在工具执行成功后运行，失败不阻断
    if not result.is_error:
        post_hooks = run_hooks(
            "PostToolUse",
            call.name,
            call.arguments,
            ctx.cwd,
            tool_result=result.content,
        )
        for h in post_hooks:
            status = "ok" if h["success"] else f"warning: {h['output']}"
            console.print(f"[dim]hook: PostToolUse {call.name} {status}[/dim]")
    return Message(
        role="tool",
        tool_call_id=result.tool_call_id,
        content=result.content,
        is_error=result.is_error,
    )


def run_agent(
    prompt: str,
    provider: Provider,
    tools: ToolRegistry,
    max_steps: int = 99,
    cwd: Path | None = None,
    state: RuntimeState | None = None,
    session: Session | None = None,
    system_prompt: str | None = None,
) -> AgentResult:
    resolved_cwd = cwd or Path.cwd()
    state = state or RuntimeState()
    ctx = ToolContext(
        cwd=resolved_cwd,
        skip_policy=SkipPolicy.default(gitignore=load_gitignore(resolved_cwd)),
        runtime_state=state,
    )

    def emit(line: str) -> None:
        # 工具结果可能很长，且并行读时多份 observation 并发打印会乱序：
        # 完整 observation 只通过 tool_result 回填给模型（并落盘 session），
        # 终端 / trace 只保留 tool_call / final / interrupted / continue。
        if line.startswith("observation:"):
            return
        trace.append(line)
        # highlight=False：emit 打印的是模型文本/工具结果（含代码片段），
        # 默认 highlight=True 会在 TTY 下给 "func(" 之类套 ANSI 高亮码，
        # 经 patch_stdout 后 ESC 泄漏成可见的 '?'。关掉它，原样输出。
        console.print(line, markup=False, highlight=False)

    messages: list[Message] = []
    # 如果有 session 且已有历史，从历史恢复；否则从当前prompt冷启动
    if session and session.history:
        messages = list(session.history)
        messages.append(
            Message(
                role="user",
                content=prompt,
            )
        )
    else:
        messages = [Message(role="user", content=prompt)]
    # 刚加进 messages 的这条 user prompt 也要落盘，
    # 否则 --resume 时 session.history 里只有 assistant 没有起点 user
    if session:
        session.append_messages([messages[-1]])

    trace: list[str] = []
    # Stop hook 续跑次数，封顶 2 防死循环（hook 写错时不至于无限续跑）
    continuation_count = 0

    for _step in range(max_steps):
        # 消息超过40条自动压缩
        if len(messages) > 40:
            messages = compact(messages, keep=8)
            console.print(f"[dim]compacted: {len(messages)} messages remaining[/dim]")
        response = provider.complete(messages, tools=tools.list(), system=system_prompt)
        messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
                provider_data=response.provider_data,
            )
        )
        # ESC 半步中断——模型已返回（可能带 tool_calls），但任何工具都还没执行。
        # 主线程在用户按 ESC 时 set 了 state.abort_event；这里检测到就短路返回。
        if state.abort_event.is_set():
            emit("interrupted by user")
            if response.tool_calls:
                # 配对不变量：assistant 给了 N 个 tool_use，紧跟的下一条必须是带
                # 同样 N 个 tool_result 的 user 消息（按 tool_use_id 一一配对），
                # 否则下次请求会被 API 拒。本项目用扁平 Message：每个 tool_result
                # 是独立的 Message(role="tool")，_to_wire_messages 会把连续的 tool
                # 消息合并成一条 user 消息，所以逐条追加即可保证配对。
                interrupted = [
                    Message(
                        role="tool",
                        tool_call_id=c.id,
                        content="Interrupted by user",
                        is_error=True,
                    )
                    for c in response.tool_calls
                ]
                messages.extend(interrupted)
                if session:
                    # 落盘：刚追加的 assistant(tool_calls) + 全部 tool_result
                    session.append_messages(messages[-(len(interrupted) + 1) :])
            elif session:
                # 没有 tool_calls：messages[-1] 就是模型的文本回复，单独落盘
                session.append_messages([messages[-1]])
            return AgentResult(final="interrupted", trace=trace, messages=messages)

        if not response.tool_calls:
            final = response.text or ""
            # Stop hook：模型自认答完，给 hook 一次"再推一轮"的机会。
            # 任一 hook 退出码非 0 且 stdout/stderr 有内容 → 内容当"按这个继续"，
            # 注入一条合成 user 消息再跑一轮（封顶 continuation_count < 2，防死循环）。
            forced: str | None = None
            if continuation_count < 2:
                payload = {
                    "event": "Stop",
                    "final_text": final,
                    "cwd": str(ctx.cwd),
                    "continuation_count": continuation_count,
                }
                for h in run_hooks_raw("Stop", payload, ctx.cwd):
                    if not h["success"] and h["output"].strip():
                        forced = h["output"].strip()
                        break
            if forced is not None:
                # 先把模型这一轮的回答展示出来，再 emit continue:——与 day-08 §3.4
                # 验证输出一致（final: ... → continue: add a unit test）。否则 continue
                # 分支直接回到 loop 顶，模型刚说的内容被吞掉，用户只看到凭空冒出的
                # continue:，不知道模型答了什么。
                emit(f"final: {final}")
                continuation_count += 1
                emit(f"continue: {forced}")
                messages.append(Message(role="user", content=f"continue: {forced}"))
                if session:
                    # 落盘 assistant(final) + 合成 user(continue)，保证 --resume 还原完整
                    session.append_messages(messages[-2:])
                continue  # 回到 loop 顶，再跑一轮
            emit(f"final: {final}")
            # 把最终 assistant 消息落盘
            if session:
                session.append_messages([messages[-1]])
            return AgentResult(final=final, trace=trace, messages=messages)

        # 并发编排（day-08 §4.3）：连续只读工具凑成并行组，写 / 未知工具截断、串行执行。
        # 回填给模型的 tool_result 顺序必须 == tool_use 顺序（Anthropic 配对不变量）；
        # ThreadPoolExecutor.map 按输入顺序返回结果，天然对齐，无需手动重排。
        tool_messages: list[Message] = []
        for batch in partition_tool_calls(response.tool_calls, tools):
            if len(batch) == 1:
                tool_messages.append(
                    execute_one_tool_call(batch[0], ctx, state, tools, emit)
                )
            else:
                # 只读组并行（max_workers=4 够用，非性能调优重点）。
                with ThreadPoolExecutor(max_workers=4) as ex:
                    tool_messages.extend(
                        ex.map(
                            lambda c: execute_one_tool_call(c, ctx, state, tools, emit),
                            batch,
                        )
                    )
        messages.extend(tool_messages)
        # 落盘：assistant(tool_calls) + 全部 tool_result 各一次（扁平消息模型下
        # _to_wire_messages 会把连续 role="tool" 合并成一条 user 消息，配对不变量成立）。
        if session:
            session.append_messages(messages[-(len(tool_messages) + 1) :])

    final = f"reached max_steps={max_steps}"
    emit(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
