"""回归测试：worker 线程经 run_on_main_terminal 调用确认函数，必须把执行
调度到主线程的事件循环上、安全返回结果，而不是在 worker 线程里抛
RuntimeError: There is no current event loop in thread 'worker_loop'。

保护的不变量（见 interactive.py 的 run_on_main_terminal 注释）：
worker 线程要问用户时，必须经 run_coroutine_threadsafe(coro, main_loop) 把
提问调度到主线程，in_terminal 负责暂停/恢复输入框。两个历史坑：

1. 早期用 prompt_toolkit.run_in_terminal(func)，它在「调用线程」里
   ensure_future(run()) 不传 loop → 退回 get_event_loop() 查当前线程 → worker
   没循环 → RuntimeError。在 REPL 里表现为：default 模式下任何 ask 行为工具
   （bash / 文件编辑 / web_*）从 worker 线程触发确认时，turn 直接崩成 [error]。

2. 修掉崩溃后又暴露：func（typer.confirm，含阻塞 stdin 读）若在事件循环线程上
   同步跑，会冻结主循环，patch_stdout 排队的 "Command: / Run this command?"
   提示刷不出来——用户看不到确认、turn 卡死，后续排队的输入永远轮不到。
   所以 func 必须丢进 executor，让主循环保持响应。
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import sys
import threading
import time

from agent_code.interactive import run_on_main_terminal


def _start_main_loop() -> tuple[dict, threading.Thread, threading.Event]:
    """启动一个后台主线程跑 asyncio 事件循环，返回共享 box + 线程 + 停止事件。

    box['loop'] 在循环就绪后填入，供 worker 线程调度协程用。
    """
    box: dict = {}
    stop = threading.Event()

    def main() -> None:
        async def _run() -> None:
            box["loop"] = asyncio.get_running_loop()
            box["ready"] = True
            while not stop.is_set():  # 轮询 threading.Event，主循环保持存活
                await asyncio.sleep(0.02)

        asyncio.run(_run())

    mt = threading.Thread(target=main, name="main", daemon=True)
    mt.start()
    while not box.get("ready"):
        time.sleep(0.01)
    return box, mt, stop


def test_run_on_main_terminal_returns_value_from_worker_thread() -> None:
    """worker 线程经 run_on_main_terminal 调 func：返回结果，不抛 RuntimeError。"""
    box, mt, stop = _start_main_loop()
    try:
        out: dict = {}

        def worker() -> None:
            try:
                out["v"] = run_on_main_terminal(lambda: 6 * 7, box["loop"])
            except Exception as exc:
                out["err"] = repr(exc)

        wt = threading.Thread(target=worker, name="worker_loop")
        wt.start()
        wt.join()

        assert "err" not in out, f"意外的异常: {out.get('err')}"
        assert out["v"] == 42
    finally:
        stop.set()
        mt.join(timeout=2)


def test_run_on_main_terminal_keeps_main_loop_responsive() -> None:
    """func 含阻塞读时，主循环必须保持响应（不卡）——否则确认提示刷不出来。

    旧实现同步调 func()，会冻结主循环：func 阻塞期间心跳几乎不推进。
    改成 run_in_executor 后，func 在线程池里阻塞，主循环持续推进心跳。
    """
    box: dict = {}
    stop = threading.Event()

    def main() -> None:
        async def _run() -> None:
            box["loop"] = asyncio.get_running_loop()
            ticks: list[float] = []
            box["ticks"] = ticks

            async def heartbeat() -> None:
                while not stop.is_set():
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.05)

            hb = asyncio.create_task(heartbeat())
            box["ready"] = True
            while not stop.is_set():
                await asyncio.sleep(0.02)
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

        asyncio.run(_run())

    mt = threading.Thread(target=main, name="main", daemon=True)
    mt.start()
    while not box.get("ready"):
        time.sleep(0.01)
    try:
        result: dict = {}

        def worker() -> None:
            def blocking_func() -> str:
                # 模拟 typer.confirm 的阻塞 stdin 读：睡 0.3s
                start = len(box["ticks"])
                time.sleep(0.3)
                box["delta"] = len(box["ticks"]) - start
                return "answered"

            try:
                result["v"] = run_on_main_terminal(blocking_func, box["loop"])
            except Exception as exc:
                result["err"] = repr(exc)

        wt = threading.Thread(target=worker, name="worker_loop")
        wt.start()
        wt.join()

        assert "err" not in result, result.get("err")
        assert result["v"] == "answered"
        # 关键：func 阻塞 0.3s 期间，主循环心跳必须推进多次。
        # 同步跑 func() 时主循环冻结，delta≈0；run_in_executor 后 delta≈6。
        assert (
            box["delta"] >= 3
        ), f"主循环在 func 阻塞期间被冻结：仅 {box['delta']} 个心跳（期望 >=3）"
    finally:
        stop.set()
        mt.join(timeout=2)


def test_run_on_main_terminal_writes_to_real_stdout_not_patch_stdout_proxy(
    monkeypatch,
) -> None:
    """fix：cross-thread 确认 func 的输出必须直写真实终端，绕过 patch_stdout 代理。

    根因（实跑复现）：patch_stdout 的 flush 线程靠 run_in_terminal(write_and_flush)
    把输出排到输入框上方，而 run_in_terminal 要排队等当前 in_terminal 释放。
    cross-thread 确认（typer.confirm）正好占着 in_terminal 阻塞读 stdin——于是
    确认提示（无换行符的 "Run this command? [y/N]:"）被代理缓冲、flush 线程又被
    挡，用户按回车前根本看不见，REPL 表现为"卡死，按一下回车才继续"。

    修法：run_on_main_terminal 在 in_terminal 里把 sys.stdout 临时指回真实终端
    （patch_stdout 接管前的原 stdout，= sys.__stdout__），func 直写、立即可见。
    in_terminal 已擦除输入框并关闭渲染，此刻直写真实终端是安全的。
    """
    real = io.StringIO()  # 真实终端替身
    proxy = io.StringIO()  # patch_stdout 代理替身（会缓冲）
    monkeypatch.setattr(sys, "stdout", proxy)
    monkeypatch.setattr(sys, "__stdout__", real)

    box, mt, stop = _start_main_loop()
    try:
        captured: dict = {}

        def worker() -> None:
            def func() -> bool:
                # 模拟 typer.confirm 的无换行提示
                print("Run this command? [y/N]:", end="")
                return True

            try:
                captured["v"] = run_on_main_terminal(func, box["loop"])
            except Exception as exc:  # noqa: BLE001
                captured["err"] = repr(exc)

        wt = threading.Thread(target=worker, name="worker_loop")
        wt.start()
        wt.join()

        assert "err" not in captured, captured.get("err")
        assert captured["v"] is True
        # 关键不变量：func 输出走真实终端，不进 patch_stdout 代理
        assert "Run this command?" in real.getvalue(), (
            "func 输出应直写真实终端（用户才看得见确认）"
        )
        assert "Run this command?" not in proxy.getvalue(), (
            "func 输出不应进 patch_stdout 代理（会被缓冲/被 in_terminal 挡住）"
        )
    finally:
        stop.set()
        mt.join(timeout=2)


def test_prompt_toolkit_run_in_terminal_is_unsafe_cross_thread() -> None:
    """锚定根因：裸的 prompt_toolkit.run_in_terminal(func) 在无线程循环里
    ensure_future → RuntimeError。把这条坑钉死，防止有人把它换回来。"""
    from prompt_toolkit.application.run_in_terminal import run_in_terminal

    err: dict = {}

    def worker() -> None:
        try:
            run_in_terminal(lambda: 1)
            err["ok"] = True
        except RuntimeError as exc:
            err["msg"] = str(exc)

    wt = threading.Thread(target=worker, name="worker_loop")
    wt.start()
    wt.join()

    # 线程无循环时必须抛 RuntimeError（这正是被修复的 bug）
    assert "msg" in err
    assert "no current event loop" in err["msg"]
