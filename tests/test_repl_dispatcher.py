"""回归测试：_ReplDispatcher 的排队-回灌机制。

保护的不变量：turn 进行中（busy）提交的输入进 input_queue 排队，turn 结束后
worker_loop 必须把它回灌到 job_queue 并真正执行。这是 REPL 里
"[queued] turn 结束后自动处理" 提示对用户的承诺——也是「第二个 prompt 在第一个
跑完后自动接着跑」的实现。

这条路径用真实模型无法稳定复现（模型会跑偏、单轮耗时长、不按时 final），
所以拿 fake run_turn 直接验证 worker_loop 的确定性逻辑。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_code.interactive import _ReplDispatcher
from agent_code.runtime import RuntimeState
from agent_code.slash import SlashContext


def _ctx() -> SlashContext:
    return SlashContext(
        cwd=Path("."),
        permission_mode="default",
        model="m",
        provider="p",
        session_id=None,
    )


def _wait_busy(disp: _ReplDispatcher, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if disp.busy.is_set():
            return
        time.sleep(0.005)
    raise AssertionError("worker 未进入 turn（busy 未置位）")


def test_queued_input_runs_after_busy_turn_ends() -> None:
    """P1 在跑（busy）时提交 P2 → 进 input_queue 排队；P1 结束后 P2 必须被执行。"""
    state = RuntimeState()
    ran: list[str] = []
    release_p1 = threading.Event()

    def fake_run_turn(text: str) -> None:
        ran.append(text)
        if text == "P1":
            release_p1.wait(timeout=5)  # 模拟 turn1 阻塞，保持 busy

    disp = _ReplDispatcher(state, fake_run_turn, _ctx)
    worker = threading.Thread(target=disp.worker_loop, daemon=True)
    worker.start()
    try:
        disp.submit("P1")
        _wait_busy(disp)

        # busy 期间提交 P2 → 必须排队，不能直接跑
        disp.submit("P2")
        assert state.input_queue.qsize() == 1, "P2 应进 input_queue 排队"
        assert ran == ["P1"], f"P2 不该在 turn1 期间执行: {ran}"

        # 放行 turn1 → 结束后回灌 P2 并执行
        release_p1.set()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if ran == ["P1", "P2"]:
                break
            time.sleep(0.005)
        assert ran == ["P1", "P2"], f"P2 未被回灌执行: {ran}"
    finally:
        disp.stop()
        worker.join(timeout=2)


def test_multiple_queued_inputs_run_in_order() -> None:
    """排了多条（P2、P3）→ turn1 结束后按顺序全部执行。"""
    state = RuntimeState()
    ran: list[str] = []
    release_p1 = threading.Event()

    def fake_run_turn(text: str) -> None:
        ran.append(text)
        if text == "P1":
            release_p1.wait(timeout=5)

    disp = _ReplDispatcher(state, fake_run_turn, _ctx)
    worker = threading.Thread(target=disp.worker_loop, daemon=True)
    worker.start()
    try:
        disp.submit("P1")
        _wait_busy(disp)
        disp.submit("P2")
        disp.submit("P3")
        assert state.input_queue.qsize() == 2
        release_p1.set()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if ran == ["P1", "P2", "P3"]:
                break
            time.sleep(0.005)
        assert ran == ["P1", "P2", "P3"], f"排队输入未按序执行: {ran}"
    finally:
        disp.stop()
        worker.join(timeout=2)


def test_slash_message_renders_rich_markup_not_literal(capsys) -> None:
    """slash 命令的回显里含 Rich 标记（/help 的 [bold]可用命令：[/bold]、[bold]/help[/bold]）。
    submit() 必须渲染标记，不能把 [bold] 原样泄漏到终端——之前 REPL 里 submit 用
    裸 print()，导致 /help 输出满屏字面 [bold] 标签（单次模式 cli.py 用 console.print
    是对的，REPL 这条路径漏了）。"""
    state = RuntimeState()
    disp = _ReplDispatcher(state, lambda _text: None, _ctx)
    disp.submit("/help")
    out = capsys.readouterr().out
    assert "可用命令" in out, f"/help 应输出命令列表: {out!r}"
    assert "[bold]" not in out, f"Rich 标记应被渲染，不应原样泄漏: {out!r}"


def test_busy_turn_exception_still_drains_queue() -> None:
    """turn1 抛异常也不影响回灌：P2 仍要执行，worker 不能死。"""
    state = RuntimeState()
    ran: list[str] = []
    release_p1 = threading.Event()

    def fake_run_turn(text: str) -> None:
        ran.append(text)
        if text == "P1":
            release_p1.wait(timeout=5)
            raise RuntimeError("boom")  # turn1 抛错

    disp = _ReplDispatcher(state, fake_run_turn, _ctx)
    worker = threading.Thread(target=disp.worker_loop, daemon=True)
    worker.start()
    try:
        disp.submit("P1")
        _wait_busy(disp)
        disp.submit("P2")
        release_p1.set()  # turn1 抛错 → finally busy.clear → 回灌 P2
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if ran == ["P1", "P2"]:
                break
            time.sleep(0.005)
        assert ran == ["P1", "P2"], f"turn1 抛错后 P2 未执行: {ran}"
        assert worker.is_alive(), "worker 应在 turn 异常后继续存活"
    finally:
        disp.stop()
        worker.join(timeout=2)
