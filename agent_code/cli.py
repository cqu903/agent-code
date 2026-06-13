from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from rich.console import Console
from .agent import run_agent
from .llm_log import create_llm_logger
from .model import AnthropicProvider
from .tools import default_tools
from typing import Literal
from .session import Session
from .agent import build_system_prompt
from .slash import SlashContext, dispatch_slash

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


def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or "unknown"
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


def run_once(
    prompt: str,
    cwd: Path,
    provider: AnthropicProvider,
    max_steps: int,
    permission_mode: Literal["default", "acceptEdits", "plan"],
    session: Session | None = None,
    system_prompt: str | None = None,
) -> None:
    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    run_agent(
        prompt,
        provider,
        tool_registry,
        max_steps=max_steps,
        cwd=cwd,
        permission_mode=permission_mode,
        session=session,
        system_prompt=system_prompt,
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
    model: str = typer.Option(
        "glm-5.1", "--model", "-m", help="Model name."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="API base URL override."
    ),
    max_steps: int = typer.Option(
        99, "--max-steps", help="Max agent loop steps."
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

    llm_logger = create_llm_logger(log_dir)
    if llm_logger:
        console.print(f"[dim]llm log: {llm_logger.path}[/dim]\n")
    provider = AnthropicProvider(
        model=model, base_url=base_url, llm_logger=llm_logger
    )
    provider_name = _provider_name(provider.base_url)
    system_prompt = build_system_prompt(resolved_cwd)

    def run_user_input(line: str) -> None:
        """
        统一处理用户输入：先走 slash dispatch，未命中再进入 Agent Loop。
        REPL 用户输入和 cron pending prompt 都必须走这个入口。
        """
        nonlocal session
        slash_result = dispatch_slash(
            line,
            SlashContext(
                cwd=resolved_cwd,
                permission_mode=permission_mode,
                model=model,
                provider=provider_name,
                session_id=session.session_id if session else None,
            ),
        )
        if slash_result.handled:
            if slash_result.message:
                console.print(slash_result.message)
            if slash_result.should_query:
                if session is None:
                    session = Session.create(resolved_cwd)
                run_once(
                    slash_result.prompt,
                    resolved_cwd,
                    provider,
                    max_steps,
                    permission_mode,
                    session=session,
                    system_prompt=system_prompt,
                )
            return
        if session is None:
            session = Session.create(resolved_cwd)
        run_once(
            line,
            resolved_cwd,
            provider,
            max_steps,
            permission_mode,
            session=session,
            system_prompt=system_prompt,
        )

    if text:
        run_user_input(text.strip())
        return
    # 注释1: REPL分支——命令后没跟prompt，走下面交互循环
    render_header(resolved_cwd)
    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    console.print("输入 /help 查看命令，输入 /exit 退出。")
    while True:
        line = console.input("[bold]>[/bold] ").strip()
        if not line:
            continue
        if line == "/exit":
            console.print("Bye.")
            return
        run_user_input(line)


def main() -> None:
    load_env()
    typer.run(main_command)
