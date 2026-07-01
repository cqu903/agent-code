# day8
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from rich.console import Console
from .agent import run_agent
from .model import AnthropicProvider, Provider
from .tools import default_tools
from typing import Callable, Literal
from .session import Session
from .agent import build_system_prompt
from .slash import SlashContext, dispatch_slash
from .scheduler import CronScheduler
from .tools.cron import set_scheduler
from .runtime import RuntimeState

console = Console()
tool_registry = default_tools()


def load_env() -> None:
    # 固定从包目录读取 .env，不依赖当前工作目录
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)


def render_header(
    cwd: Path,
    provider_name: str = "",
    model: str = "",
) -> None:
    # cwd是后续文件工具和bash工具都要遵守的工作边界
    console.print("[bold]Agent Code[/bold]")
    if provider_name:
        console.print(f"[dim]cwd: {cwd}  ·  {provider_name} / {model}[/dim]\n")
    else:
        console.print(f"[dim]cwd: {cwd}[/dim]\n")


def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or "unknown"
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else host


# provider 注册表：name -> 构造器（任意可调用对象，返回 Provider）。
# 用 Callable[..., Provider] 而非 type[Provider]：后者只描述实例接口
# （.complete），不保证构造器签名；不同后端构造器可异构，每项可用
# lambda 适配参数。新增后端（如 OpenAI）在此加一行。
_PROVIDERS: dict[str, Callable[..., Provider]] = {
    "anthropic": AnthropicProvider,
}


def create_provider(provider_name: str, model: str, base_url: str | None) -> Provider:
    """按 provider_name 分派到具体后端，返回通用 Provider。

    目前仅注册了 Anthropic 兼容后端；未知 name 回退到它
    （DeepSeek / GLM 等都走 Anthropic 兼容端点）。新增非兼容后端时
    在 _PROVIDERS 里注册一行即可。
    """
    factory = _PROVIDERS.get(provider_name, AnthropicProvider)
    return factory(model=model, base_url=base_url)


def run_once(
    prompt,
    cwd,
    provider,
    max_steps,
    permission_mode,
    session=None,
    skill_allowed_tools=None,
) -> None:
    provider_name = _provider_name(provider.base_url)
    render_header(cwd, provider_name, provider.model)
    state = RuntimeState(
        permission_mode=permission_mode, model=provider.model, provider=provider_name
    )
    # /skill 本轮白名单跟着这一次 run_once 走（one-shot 路径）。
    # system prompt 每次重新拼，避免启动时固化（v4 output-style 也靠这个）。
    state.skill_allowed_tools = skill_allowed_tools
    system_prompt = build_system_prompt(cwd, state)
    run_agent(
        prompt,
        provider,
        default_tools(),
        max_steps=max_steps,
        cwd=cwd,
        state=state,
        session=session,
        system_prompt=system_prompt,
    )


def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-C"),
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

    provider = AnthropicProvider(model=model, base_url=base_url)
    provider_name = _provider_name(provider.base_url)

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
                    skill_allowed_tools=slash_result.allowed_tools,
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
        )

    if text:
        run_user_input(text.strip())
        return

    from .interactive import run_interactive_shell

    render_header(resolved_cwd, provider_name, model)
    if session is None:
        session = Session.create(resolved_cwd)

    state = RuntimeState(
        permission_mode=permission_mode, model=model, provider=provider_name
    )
    tools = default_tools()

    # 启动 cron 调度器：后台线程到点把 prompt 排进 pending queue，
    # interactive shell 的 cron pump 负责定期 drain 并重放进输入流。
    scheduler = CronScheduler(resolved_cwd)
    set_scheduler(scheduler)
    scheduler.start()

    def run_turn(line: str) -> None:
        # slash 已在主线程处理过，这里只跑 agent。
        # provider 和 system prompt 都按 RuntimeState 每轮重建，所以 /model、
        # /output-style、/skill 白名单切换都是下一轮生效——不能在启动时固化。
        trun_provider = create_provider(state.provider, state.model, base_url)
        turn_system_prompt = build_system_prompt(resolved_cwd, state)
        run_agent(
            line,
            trun_provider,
            tools,
            max_steps=max_steps,
            cwd=resolved_cwd,
            state=state,
            session=session,
            system_prompt=turn_system_prompt,
        )

    def make_slash_context() -> SlashContext:
        return SlashContext(
            cwd=resolved_cwd,
            permission_mode=state.permission_mode,
            model=state.model,
            provider=state.provider,
            session_id=session.session_id if session else None,
            state=state,
        )

    console.print("输入 /help 查看命令，输入 /exit 退出。")
    try:
        run_interactive_shell(
            state, run_turn, make_slash_context, scheduler.drain_pending
        )
    finally:
        scheduler.stop()


def main() -> None:
    load_env()
    typer.run(main_command)
