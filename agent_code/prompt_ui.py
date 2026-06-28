from __future__ import annotations
import difflib
import typer

_terminal_asker = None


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


def confirm_edit(path: str) -> bool:
    return _ask(lambda: typer.confirm(f"Apply this edit to {path}?", default=False))


def confirm_command(command: str) -> bool:
    return _ask(lambda: typer.confirm("Run this command?", default=False))


def confirm_tool_use(tool_name: str, detail: str) -> bool:
    return _ask(lambda: typer.confirm(f"Allow {tool_name}: {detail}?", default=False))


def prompt_single_choice(question: str, labels: list[str]) -> str | None:
    """展示一个 number menu 让用户单选，返回被选中的label。
    渲染 + 读 stdin 都走 _ask —— REPL 模式下整块调度到主线程 run_in_terminal，
    避免和 prompt_toolkit 抢 stdin；one-shot 直接跑。"""
    from rich.console import Console

    def _do() -> str | None:
        console = Console()
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
