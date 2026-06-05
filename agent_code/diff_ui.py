from __future__ import annotations
import difflib
import typer


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

def confirm_edit(path: str)->bool:
    """让用户确认是否应用这次编辑，默认不应用"""
    return typer.confirm(f"Apply this edit to {path}?, default=False")
