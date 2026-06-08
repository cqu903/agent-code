from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from .agent import run_agent
from .llm_log import create_llm_logger
from .model import AnthropicProvider
from .tools import default_tools
from typing import Literal
from .session import Session

console = Console()
tool_registry = default_tools()


def load_env() -> None:
    # 固定从包目录读取 .env，不依赖当前工作目录
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)


def render_header(cwd: Path):
    # cwd是后续文件工具和bash工具都要遵守的工作边界
    console.print("[bold]Agent Code[/bold]")
    console.print(f"[dim]cwd: {cwd}[/dim]\n")


def handle_slash(line: str) -> bool:
    if line == "/help":
        tool_registry.print_help(console)
        return True
    return False


def run_once(
    prompt: str,
    cwd: Path,
    log_dir: Path | None,
    permission_mode: Literal["default", "acceptEdits", "plan"],
    session: Session | None = None,
) -> None:
    llm_logger = create_llm_logger(log_dir)
    if llm_logger:
        console.print(f"[dim]llm log: {llm_logger.path}[/dim]\n")
    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    provider = AnthropicProvider(llm_logger=llm_logger)
    run_agent(
        prompt,
        provider,
        tool_registry,
        cwd=cwd,
        permission_mode=permission_mode,
        session=session,
    )


def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Directory for LLM request/response JSONL logs.",
        envvar="AGENT_CODE_LOG_DIR",
    ),
    permission_mode: Literal["default", "acceptEdits", "plan"] = typer.Option(
        "default",
        "--permission-mode",
        help="Permission mode: default, acceptEdits, plan",
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="按 session id 恢复指定会话"
    ),
    continue_: bool = typer.Option(
        False, "--continue", "-c", help="回复 cwd 最近一次会话"
    ),
) -> None:
    # 启动时只解析一次cwd，让整个运行共享同一个工作目录
    resolved_cwd = cwd.resolve()
    session: Session | None = None
    if continue_:
        session = Session.load_latest(resolved_cwd)
        if session is None:
            console.print("[red]没有找到历史会话，无法 --continue。[/red]")
            raise typer.Exit(code=1)
    elif resume:
        session = Session.load_id(resolved_cwd, resume)
        if session is None:
            console.print(f"[red]找不到 session: {resume}[/red]")
            raise typer.Exit(code=1)
    text = prompt.strip()

    if text:
        if session is None:
            session = Session.create(resolved_cwd)
        # 有prompt参数时进入一次性模式，运行一次就退出
        run_once(text, resolved_cwd, log_dir, permission_mode, session)
        return
    # 注释1: REPL分支——命令后没跟prompt，走下面交互循环
    render_header(resolved_cwd)
    console.print("输入 /help 查看命令，输入 /exit 退出。")
    while True:
        line = console.input("[bold]>[/bold] ").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        if line.startswith("/") and handle_slash(line):
            continue
        run_once(line, resolved_cwd, log_dir, permission_mode)


def main() -> None:
    load_env()
    typer.run(main_command)
