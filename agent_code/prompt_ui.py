from __future__ import annotations
import difflib
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

_terminal_asker = None

# 确认对话框的预览（Command: / Diff for …）用这个 console 打印。关键：预览必须和
# 确认问题一起在 _ask 的 func 里打印——在 REPL 里 func 经 run_on_main_terminal
# 调度到主线程 in_terminal 内、且 sys.stdout 被临时指回真实终端直写（见
# interactive.run_on_main_terminal）。若预览由 worker 线程单独 console.print，
# 会进 patch_stdout 代理，被 in_terminal 挡住，排到用户回答确认之后才出现。
console = Console()


def set_terminal_asker(asker) -> None:
    global _terminal_asker
    _terminal_asker = asker


def _ask(func):
    """worker 要问用户时走这里。交互 shell 注入了 asker → 丢回主线程事件循环问；
    one-shot 没注入（_terminal_asker is None）→ 直接问。"""
    if _terminal_asker is not None:
        return _terminal_asker(func)
    return func()


def render_diff(old: str, new: str, path: str) -> str:
    """用difflib生成unified diff，给增删行加rich markup着色"""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    colored: list[str] = []
    for line in diff_lines:
        line = line.rstrip()
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"[bold]{line}[/bold]")
        elif line.startswith("-"):
            colored.append(f"[red]{line}[/red]")
        elif line.startswith("+"):
            colored.append(f"[green]{line}[/green]")
        elif line.startswith("@@"):
            colored.append(f"[cyan]{line}[/cyan]")
        else:
            colored.append(line)
    return "\n".join(colored)


def confirm_edit(path: str, diff_text: str = "") -> bool:
    """文件编辑确认。diff 预览和确认问题都在 func 里打印（见模块 console 注释），
    一起进 run_on_main_terminal → in_terminal 直写真实终端，保证预览在问题之前、
    即时可见。"""

    def _do() -> bool:
        if diff_text:
            console.print(f"\n[bold]Diff for {path}:[/bold]")
            console.print(diff_text)
        return typer.confirm(f"Apply this edit to {path}?", default=False)

    return _ask(_do)


def confirm_command(command: str, timeout: int = 30, cwd: Path | None = None) -> bool:
    """bash 命令确认。命令预览（Command: / timeout）和确认问题都在 func 里打印
    （见模块 console 注释），一起进 run_on_main_terminal → in_terminal 直写。"""

    def _do() -> bool:
        console.print(f"\n[bold yellow]Command:[/bold yellow] {command}")
        console.print(f"[dim]timeout: {timeout}s  cwd: {cwd}[/dim]")
        return typer.confirm("Run this command?", default=False)

    return _ask(_do)


def confirm_tool_use(tool_name: str, detail: str) -> bool:
    return _ask(lambda: typer.confirm(f"Allow {tool_name}: {detail}?", default=False))


def confirm_plan(plan_summary: str) -> bool:
    """渲染计划面板并等用户批准（Day 8 v6 plan 闭环的审批门）。

    和 confirm_edit/confirm_command 同构：Plan 面板 + 确认问题都在 _do 里直接
    console.print/typer.confirm——不另起 StringIO 缓冲、不走 _write_real_terminal。
    因为 REPL 下 _ask 会把 _do 经 interactive.run_on_main_terminal 调度进
    in_terminal，期间 sys.stdout 被临时指回真实终端（见模块顶部 console 注释），
    所以 console.print 的 Panel 立即可见，不会卡在 patch_stdout 代理里排到
    确认之后；one-shot 没注入 asker，_do 直接跑、同样打印。"""

    def _do() -> bool:
        console.print(
            Panel(
                plan_summary or "(empty plan)",
                title="Plan",
                border_style="blue",
            )
        )
        return typer.confirm("Approve this plan and exit plan mode?", default=False)

    return _ask(_do)


def prompt_single_choice(question: str, labels: list[str]) -> str | None:
    """展示一个 number menu 让用户单选，返回被选中的label。
    渲染 + 读 stdin 都走 _ask —— REPL 模式下整块调度到主线程 run_in_terminal，
    避免和 prompt_toolkit 抢 stdin；one-shot 直接跑。"""

    def _do() -> str | None:
        console.print(f"\n[bold yellow]? {question}[/bold yellow]")
        for i, label in enumerate(labels, 1):
            console.print(f"   {i}. {label}")
        console.print(f"   0. [dim]Skip / Other[/dim]")
        try:
            choice = typer.prompt("Choice", default=0)
            idx = int(choice)
            if 1 <= idx <= len(labels):
                return labels[idx - 1]
            return None
        except (ValueError, IndexError):
            return None

    return _ask(_do)
