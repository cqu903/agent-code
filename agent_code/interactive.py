"""
agent_code/interactive.py — prompt_toolkit 交互 shell。

主线程 = PromptSession（输入、键位、状态栏 + slash 分派）；
worker 线程 = run_agent（阻塞 provider.complete + 工具执行）。
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from . import prompt_ui
from .runtime import RuntimeState
from .slash import SlashContext, dispatch_slash


def run_interactive_shell(
    state: RuntimeState,
    run_turn: Callable[[str], None],
    make_slash_context: Callable[[], SlashContext],
    drain_pending: Callable[[], list[str]] | None = None,
) -> None:
    """启动交互 REPL。主线程读输入 + 分派 slash，worker 线程跑 Agent Loop。

    drain_pending：可选的 cron pending 取回回调。非 None 时主循环会定期
    drain 到期任务，与用户输入走同一条 submit 路径（保留 slash 分派语义），
    这样 cron 里存的 slash 命令（如 /compact）也会被正确分派。
    """
    job_queue: queue.Queue[str] = queue.Queue()

    def worker_loop() -> None:
        while True:
            text = job_queue.get()
            if text == "__EXIT__":
                break
            state.abort_event.clear()
            run_turn(text)

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    session: PromptSession[str] = PromptSession(
        key_bindings=build_key_bindings(state),
        bottom_toolbar=lambda: bottom_toolbar(state),
    )

    async def _run() -> None:
        # get_running_loop()：拿到 prompt_async 真正在跑的那条事件循环。
        # 线程拆开后，worker 要问用户（确认编辑、批准计划）不能直接抢 stdin——
        # terminal_asker 用 run_coroutine_threadsafe 把提问调度到这条循环上，
        # run_in_terminal 暂停输入框、问完再恢复，worker 阻塞在 .result() 等结果。
        # set_terminal_asker 在 1.5 的 prompt_ui 里定义。
        loop = asyncio.get_running_loop()

        def terminal_asker(func: Callable[[], Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(
                run_in_terminal(func), loop
            ).result()

        prompt_ui.set_terminal_asker(terminal_asker)

        def submit(text: str) -> None:
            # 用户输入和 cron 重放都走这里：slash 命中则主线程分派，否则丢给 worker。
            if text.startswith("/"):
                result = dispatch_slash(text, make_slash_context())
                if result.handled:
                    if result.message:
                        print(result.message)
                    if result.should_query:
                        job_queue.put(result.prompt)
                    return
            job_queue.put(text)

        async def _cron_pump() -> None:
            # 定期把到期的 cron job 重放进输入流，走和用户输入一样的 submit 路径。
            while True:
                await asyncio.sleep(0.5)
                if drain_pending is None:
                    continue
                for pp in drain_pending():
                    print(f"cron: 触发定时任务 → {pp}")
                    submit(pp)

        # patch_stdout:worker线程里run_agent的console.print会被安全的排到输入框上方
        with patch_stdout():
            cron_task = asyncio.create_task(_cron_pump())
            try:
                while True:
                    try:
                        text = (await session.prompt_async("> ")).strip()
                    except (KeyboardInterrupt, EOFError):
                        break
                    if not text:
                        continue
                    if text == "/exit":
                        break
                    submit(text)
            finally:
                cron_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cron_task

    asyncio.run(_run())
    job_queue.put("__EXIT__")


def build_key_bindings(state: RuntimeState) -> KeyBindings:
    """v1先绑 ESC"""
    kb = KeyBindings()

    @kb.add("escape")
    def _(event: Any) -> None:
        state.abort_event.set()

    return kb


def bottom_toolbar(state: RuntimeState) -> str:
    """底部状态栏——当前模式+模型"""
    mode = {"default": "default", "acceptEdits": "accept edits", "plan": "plan"}.get(
        state.permission_mode, state.permission_mode
    )
    return f" {mode} · {state.model} "
