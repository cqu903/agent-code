from __future__ import annotations

import sys
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
import threading
from queue import Empty, Queue
from .scheduler import CronScheduler
from .tools.cron import set_scheduler

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
    model: str = typer.Option("glm-5.1", "--model", "-m", help="Model name."),
    base_url: str | None = typer.Option(
        None, "--base-url", help="API base URL override."
    ),
    max_steps: int = typer.Option(99, "--max-steps", help="Max agent loop steps."),
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
    provider = AnthropicProvider(model=model, base_url=base_url, llm_logger=llm_logger)
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

    # 启动scheduler
    scheduler = CronScheduler(resolved_cwd)
    set_scheduler(scheduler)
    scheduler.start()

    if session:
        suffix = " (resumed)" if session.resumed else ""
        console.print(f"[dim]session: {session.session_id}{suffix}[/dim]")
    console.print("输入 /help 查看命令，输入 /exit 退出。")

    # 输入线程只负责把用户输入放进队列，不打印提示符；
    # 提示符由主线程在“准备好接收下一条输入”时打印，避免它和命令输出
    # 在两条线程里交错后拼到同一行，导致输出结束后看不到 ">"。
    input_queue: Queue[str | None] = Queue()
    stop_repl = threading.Event()

    def _read_input() -> None:
        while not stop_repl.is_set():
            try:
                raw = sys.stdin.readline()
            except KeyboardInterrupt:
                input_queue.put(None)
                return
            if raw == "":  # EOF (Ctrl+D)
                input_queue.put(None)
                return
            input_queue.put(raw.strip())

    input_thread = threading.Thread(target=_read_input, daemon=True)
    input_thread.start()

    def _show_prompt() -> None:
        print("> ", end="", flush=True)

    try:
        while True:
            _show_prompt()
            # 等待用户输入的同时，主线程定期检查 cron pending queue；
            # 内层循环在拿到一行输入前持续轮询，保证空闲时 cron 也能触发。
            while True:
                for pp in scheduler.drain_pending():
                    # cron 输出前先换行离开提示符所在行，输出后再重新打印提示符，
                    # 否则会和 ">" 拼在同一行。
                    print(flush=True)
                    console.print(f"[dim]cron: running scheduled job → {pp}[/dim]")
                    run_user_input(pp)
                    _show_prompt()
                try:
                    line = input_queue.get(timeout=0.5)
                    break
                except Empty:
                    continue

            if line is None:
                print(flush=True)  # 离开提示符行再退出
                break
            if not line:
                continue  # 空行：回到循环顶部重新打印提示符
            if line == "/exit":
                print(flush=True)
                console.print("Bye.")
                break
            # 用户按下回车后终端已把光标移到新行，命令输出自然落在新行。
            run_user_input(line)
    finally:
        stop_repl.set()
        scheduler.stop()


def main() -> None:
    load_env()
    typer.run(main_command)
