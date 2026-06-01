from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from .agent import run_agent
from .llm_log import create_llm_logger
from .model import AnthropicProvider
from .tools import default_tools

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


def run_once(prompt: str, cwd: Path, log_dir: Path | None) -> None:
    llm_logger = create_llm_logger(log_dir)
    if llm_logger:
        console.print(f"[dim]llm log: {llm_logger.path}[/dim]\n")
    provider = AnthropicProvider(llm_logger=llm_logger)
    result = run_agent(prompt, provider, tool_registry)
    for line in result.trace:
        console.print(line)


def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Directory for LLM request/response JSONL logs.",
        envvar="AGENT_CODE_LOG_DIR",
    ),
) -> None:
    # 启动时只解析一次cwd，让整个运行共享同一个工作目录
    resolved_cwd = cwd.resolve()
    text = prompt.strip()

    if text:
        # 有prompt参数时进入一次性模式，运行一次就退出
        run_once(text, resolved_cwd, log_dir)
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
        run_once(line, resolved_cwd, log_dir)


def main() -> None:
    load_env()
    typer.run(main_command)
