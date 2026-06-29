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
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from . import prompt_ui
from .runtime import RuntimeState
from .slash import SlashContext, dispatch_slash

# 主线程回显 slash 命令结果用的 Rich console。Rich 在 write 时才解析 sys.stdout，
# 所以 patch_stdout 接管后，这里的 console.print 也会被安全地排到输入框上方
# （和 agent.py 里 worker 线程的 emit 走同一条 patch_stdout 路径）。
console = Console()


def run_on_main_terminal(
    func: Callable[[], Any], loop: asyncio.AbstractEventLoop
) -> Any:
    """把 func 调度到主线程的事件循环上、在 in_terminal 上下文里执行。

    worker 线程要问用户（确认编辑 / 批准命令 / 单选）时走这里。三个约束：
    1. 不能直接抢 stdin——必须让主线程的输入框先暂停、问完再恢复，in_terminal 负责。
    2. 不能用 prompt_toolkit.run_in_terminal(func)——它在「调用线程」里
       ensure_future(run()) 不传 loop，会退回 get_event_loop() 查当前线程的循环；
       worker 线程没有事件循环，直接 RuntimeError: There is no current event loop
       in thread 'worker_loop'。所以这里手写等价协程 _ask_on_main，交给
       run_coroutine_threadsafe 用显式 loop 在主线程上 ensure_future，绕开这条坑。
    3. func（如 typer.confirm）会做阻塞 stdin 读，必须丢进 executor 里跑，不能在
       事件循环线程上同步调。否则阻塞读会卡住主循环，patch_stdout 排队的
       "Command: / Run this command?" 提示刷不出来——用户看不到确认、turn 卡死，
       后续排队的输入也永远轮不到。丢进 executor 后主循环空闲、提示正常渲染。
    """

    # 注意：只是构造协程对象（async def 调用不碰事件循环，也不会 ensure_future），
    # 真正的 ensure_future 由 run_coroutine_threadsafe 在主线程用显式 loop 完成。
    async def _ask_on_main() -> Any:
        async with in_terminal():
            # 见约束 3：func 含阻塞 stdin 读，必须 run_in_executor，别在循环线程上跑。
            return await asyncio.get_running_loop().run_in_executor(None, func)

    return asyncio.run_coroutine_threadsafe(_ask_on_main(), loop).result()


class _ReplDispatcher:
    """REPL 输入分发 + worker 循环：主线程 submit，worker 线程 worker_loop。

    从 run_interactive_shell 闭包里抽出来，是为了可单测——worker_loop 的核心
    不变量「turn 进行中排队的输入，turn 结束后必须回灌并执行」可以拿 fake run_turn
    直接验证，不必依赖真实模型（模型会跑偏、耗时长，无法稳定复现这条路径）。
    """

    def __init__(
        self,
        state: RuntimeState,
        run_turn: Callable[[str], None],
        make_slash_context: Callable[[], SlashContext],
    ) -> None:
        self.state = state
        self.run_turn = run_turn
        self.make_slash_context = make_slash_context
        self.job_queue: queue.Queue[str] = queue.Queue()
        self.busy = threading.Event()

    def submit(self, text: str) -> None:
        # 用户输入和 cron 重放都走这里：slash 命中则主线程分派，否则丢给 worker。
        if text.startswith("/"):
            result = dispatch_slash(text, self.make_slash_context())
            if result.handled:
                if result.message:
                    # 用 console.print 渲染 Rich 标记（/help 等消息里的 [bold]...[/bold]），
                    # 不能用裸 print——否则标记原样泄漏成字面 [bold] 标签。
                    console.print(result.message)
                if result.should_query:
                    self.job_queue.put(result.prompt)
                return
        if self.busy.is_set():
            self.state.input_queue.put(text)
            print("[queued] turn 结束后自动处理")
        else:
            self.job_queue.put(text)

    def worker_loop(self) -> None:
        while True:
            text = self.job_queue.get()
            if text == "__EXIT__":
                break
            self.state.abort_event.clear()
            self.busy.set()
            try:
                self.run_turn(text)
            except Exception as exc:
                print(f"[error] {exc}")
            finally:
                self.busy.clear()
            # turn 结束后回灌排队期间攒下的输入——这是 [queued] 提示对用户的承诺。
            while not self.state.input_queue.empty():
                self.job_queue.put(self.state.input_queue.get())

    def stop(self) -> None:
        self.job_queue.put("__EXIT__")


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
    disp = _ReplDispatcher(state, run_turn, make_slash_context)
    worker = threading.Thread(target=disp.worker_loop, daemon=True)
    worker.start()

    session: PromptSession[str] = PromptSession(
        key_bindings=build_key_bindings(state),
        bottom_toolbar=lambda: bottom_toolbar(state),
    )

    async def _run() -> None:
        # get_running_loop()：拿到 prompt_async 真正在跑的那条事件循环。
        # 线程拆开后，worker 要问用户（确认编辑、批准命令）不能直接抢 stdin——
        # terminal_asker 把 func 经 run_on_main_terminal 调度到这条循环上：
        # in_terminal 暂停输入框、跑完 func 再恢复，worker 阻塞在 .result() 等结果。
        # 细节（为什么不能用裸 run_in_terminal）见 run_on_main_terminal 的 docstring。
        loop = asyncio.get_running_loop()

        def terminal_asker(func: Callable[[], Any]) -> Any:
            return run_on_main_terminal(func, loop)

        prompt_ui.set_terminal_asker(terminal_asker)

        async def _cron_pump() -> None:
            # 定期把到期的 cron job 重放进输入流，走和用户输入一样的 submit 路径。
            while True:
                await asyncio.sleep(0.5)
                if drain_pending is None:
                    continue
                for pp in drain_pending():
                    print(f"cron: 触发定时任务 → {pp}")
                    disp.submit(pp)

        # patch_stdout(raw=True)：worker 线程里 run_agent 的 console.print 会被安全地
        # 排到输入框上方。raw=True 关键——默认 raw=False 会把 vt100 转义序列里的 ESC
        # 字节替换成字面 '?'（Rich 的 [dim]/[bold yellow] 标记 + 数字高亮都产 ANSI），
        # 于是 compacted:/Command:/Diff/hook 这些行全变成 ?[2m... 乱码。raw=True 保留
        # ESC，真终端照常解释成颜色/样式，既不乱码又保住格式。
        with patch_stdout(raw=True):
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
                    disp.submit(text)
            finally:
                cron_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cron_task

    asyncio.run(_run())
    disp.stop()


def build_key_bindings(state: RuntimeState) -> KeyBindings:
    """v1先绑 ESC"""
    kb = KeyBindings()

    @kb.add("escape")
    def _(event: Any) -> None:
        state.abort_event.set()

    @kb.add("s-tab")
    def _(event: Any) -> None:
        new_mode = state.cycle_permission_mode()
        print(f"[mode -> {new_mode}]")

    return kb


def bottom_toolbar(state: RuntimeState) -> str:
    """底部状态栏——当前模式+模型"""
    mode = {"default": "default", "acceptEdits": "accept edits", "plan": "plan"}.get(
        state.permission_mode, state.permission_mode
    )
    return f" {mode} · {state.model} "
