"""回归测试：Stop hook（Day 8 v3）。

保护的不变量：
1. 模型返回纯文本（无 tool_use）自认答完时，若 Stop hook 退出码非 0 且 stdout/stderr
   有内容，harness 必须把内容当成"按这个继续"，注入一条合成 user 消息再跑一轮。
2. 续跑封顶 2 次——hook 恒 force 时不能死循环；初轮 + 2 次续跑后必须返回。
3. 续跑产生的 assistant(final) + 合成 user(continue) 都要落盘到 session，--resume 可还原。

与 PreToolUse/PostToolUse 不同，Stop 没有 tool_name：matcher 非 "*"/空 的 entry 一律跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_code.agent import run_agent
from agent_code.model import ModelResponse
from agent_code.session import Session
from agent_code.tools import default_tools


class FakeProvider:
    """按顺序回放预设的 ModelResponse，模拟 Provider.complete()。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._i = 0

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _write_stop_hook(tmp_path: Path) -> None:
    """在 tmp_path 下放一个条件型 Stop hook：final_text 不含 'test' 就 exit 1 +
    stderr 写 'add a unit test'。走真 subprocess，贴近实际 hook 执行路径。"""
    (tmp_path / "stop_hook.py").write_text(
        "import json, sys\n"
        "d = json.load(sys.stdin)\n"
        'if "test" not in d.get("final_text", ""):\n'
        '    sys.stderr.write("add a unit test")\n'
        "    sys.exit(1)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    # hooks.json 放 cwd（=tmp_path）；load_hooks 从 cwd 读。
    (tmp_path / "hooks.json").write_text(
        json.dumps(
            {"hooks": {"Stop": [{"matcher": "*", "run": "python3 stop_hook.py"}]}}
        ),
        encoding="utf-8",
    )


def test_stop_hook_forces_one_more_turn(tmp_path: Path) -> None:
    """第 1 轮 'done'（无 'test'）→ hook force 续跑；第 2 轮含 'test' → hook 放行 → 返回。"""
    _write_stop_hook(tmp_path)
    provider = FakeProvider(
        [
            ModelResponse(text="done"),
            ModelResponse(text="now with a test, done"),
        ]
    )

    result = run_agent(
        prompt="写个函数",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
    )

    assert result.final == "now with a test, done"
    # user(prompt) → assistant(done) → user(continue) → assistant(final)
    assert [m.role for m in result.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    cont = result.messages[2]
    assert cont.role == "user"
    assert "continue:" in (cont.content or "")
    assert "add a unit test" in (cont.content or "")


def test_stop_hook_caps_at_two_continuations(tmp_path: Path) -> None:
    """hook 恒 force（'done' 永不含 'test'）：初轮 + 2 次续跑后必须返回，不死循环。"""
    _write_stop_hook(tmp_path)
    provider = FakeProvider([ModelResponse(text="done") for _ in range(10)])

    result = run_agent(
        prompt="x",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
    )

    # 初轮 + 2 次续跑 = 3 次 complete() 调用
    assert provider._i == 3
    assert result.final == "done"
    cont = [
        m
        for m in result.messages
        if m.role == "user" and (m.content or "").startswith("continue:")
    ]
    assert len(cont) == 2


def test_stop_hook_continuation_persists_to_session(tmp_path: Path) -> None:
    """续跑的 assistant(final) + 合成 user(continue) 必须落盘，--resume 可还原。"""
    _write_stop_hook(tmp_path)
    provider = FakeProvider(
        [
            ModelResponse(text="done"),
            ModelResponse(text="now with a test, done"),
        ]
    )
    session = Session.create(tmp_path)

    run_agent(
        prompt="x",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        session=session,
    )

    roles = [m.role for m in session.history]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "continue:" in (session.history[2].content or "")


def test_stop_hook_emits_final_before_continue(tmp_path: Path) -> None:
    """与 day-08 §3.4 验证输出一致：Stop hook force 时，harness 必须先把模型这一轮的
    回答 emit 成 final:，再 emit continue:。否则模型答完即被 hook 推一轮，用户看不到
    模型实际说了什么——只有一句 continue: 凭空冒出来。

    回归保护：之前实现里 continue 分支直接 `continue`，跳过了 emit(f"final: {final}")，
    导致 trace 里 final: 永远晚于 continue:（甚至首答的 final 根本不显示）。"""
    _write_stop_hook(tmp_path)
    provider = FakeProvider(
        [
            ModelResponse(text="def reverse(s): return s[::-1]"),  # 无 'test' → force
            ModelResponse(text="ok with a test"),  # 含 'test' → 放行 → 返回
        ]
    )

    result = run_agent(
        prompt="写个函数",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
    )

    trace = result.trace
    finals = [i for i, t in enumerate(trace) if t.startswith("final:")]
    continues = [i for i, t in enumerate(trace) if t.startswith("continue:")]
    assert finals, f"trace 里没有任何 final: {trace}"
    assert continues, f"trace 里没有任何 continue: {trace}"
    # 关键不变量：第一个 final 必须在第一个 continue 之前
    assert finals[0] < continues[0], f"final: 应在 continue: 之前。trace={trace}"


def test_stop_hook_ignores_non_wildcard_matcher(tmp_path: Path) -> None:
    """Stop 没有 tool_name：matcher 非 "*"/空 的 entry 必须被跳过，不触发续跑。"""
    (tmp_path / "hooks.json").write_text(
        json.dumps(
            {"hooks": {"Stop": [{"matcher": "read_file", "run": "echo x; exit 1"}]}}
        ),
        encoding="utf-8",
    )
    provider = FakeProvider([ModelResponse(text="done")])

    result = run_agent(
        prompt="x",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
    )

    # matcher=read_file 不匹配 Stop（只认 "*"/空），直接返回，未续跑
    assert result.final == "done"
    assert provider._i == 1
    assert [m.role for m in result.messages] == ["user", "assistant"]
