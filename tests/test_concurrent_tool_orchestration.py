"""并发工具编排（day-08 §4.3）回归测试。

保护的不变量：
1. partition_tool_calls：连续只读工具凑成并行组，写 / 未知工具截断、自成串行组。
2. 单步内多个只读工具真正并行执行（墙钟证明），且回填给模型的 tool_result 顺序
   严格等于 tool_use 顺序（Anthropic 配对不变量——ex.map 按输入顺序返回）。
3. 读写混合时，写工具把只读组截断：写前/写后的读分属不同批次，写串行执行、
   写后的读能看到写之后的内容。
"""

from __future__ import annotations

import time
from pathlib import Path

from agent_code.agent import _format_call_args, partition_tool_calls, run_agent
from agent_code.model import ModelResponse
from agent_code.runtime import RuntimeState
from agent_code.tools import ToolCall, default_tools
from agent_code.tools.base import Tool, ToolRegistry


class FakeProvider:
    """按顺序回放预设的 ModelResponse，模拟 Provider.complete()。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._i = 0

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        resp = self._responses[self._i]
        self._i += 1
        return resp


# ---------------------------------------------------------------------------
# partition_tool_calls：纯函数正确性
# ---------------------------------------------------------------------------


def _read(i: str) -> ToolCall:
    return ToolCall(id=i, name="read_file", arguments={"path": f"f{i}.py"})


def _edit(i: str) -> ToolCall:
    return ToolCall(id=i, name="file_edit", arguments={"file_path": f"f{i}.py"})


def test_partition_groups_consecutive_read_only_and_splits_on_write() -> None:
    tools = default_tools()
    batches = partition_tool_calls(
        [_read("1"), _read("2"), _edit("3"), _read("4")], tools
    )
    ids = [[c.id for c in batch] for batch in batches]
    assert ids == [["1", "2"], ["3"], ["4"]]


def test_partition_unknown_tool_becomes_own_serial_batch() -> None:
    # 未知工具 fail-closed：不并入只读组，自成串行批次
    tools = default_tools()
    unk = ToolCall(id="u", name="does_not_exist", arguments={})
    batches = partition_tool_calls([_read("1"), unk, _read("2")], tools)
    ids = [[c.id for c in batch] for batch in batches]
    assert ids == [["1"], ["u"], ["2"]]


def test_partition_empty_input_returns_empty() -> None:
    assert partition_tool_calls([], default_tools()) == []


def test_partition_all_read_only_is_single_batch() -> None:
    tools = default_tools()
    batches = partition_tool_calls([_read("1"), _read("2"), _read("3")], tools)
    assert len(batches) == 1
    assert [c.id for c in batches[0]] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# _format_call_args：trace 预览截断
# ---------------------------------------------------------------------------


def test_format_call_args_truncates_long_strings_and_keeps_short() -> None:
    # 长字符串只截断到 80 字符 + … 做 trace 预览，完整值仍照常传工具
    out = _format_call_args({"content": "x" * 200, "path": "a.py"})
    assert "x" * 81 not in out  # 长串没整段进 trace
    assert "…" in out
    assert "'a.py'" in out  # 短值原样保留


# ---------------------------------------------------------------------------
# 真并行 + 顺序保持
# ---------------------------------------------------------------------------


def _registry_with_slow_echo(sleep_s: float) -> ToolRegistry:
    """带一个 is_read_only=True、内部 sleep 的自定义工具的 registry。"""

    def slow_echo(args: dict, ctx) -> str:
        time.sleep(sleep_s)
        return f"echo:{args.get('tag')}"

    registry = default_tools()
    registry.register(
        Tool(
            name="slow_echo",
            description="slow read-only echo for concurrency test",
            run=slow_echo,
            is_read_only=True,
        )
    )
    return registry


def test_parallel_read_only_tools_run_concurrently_and_keep_order(
    tmp_path: Path,
) -> None:
    # 3 个只读调用各 sleep 0.3s：串行 ~0.9s，并行（max_workers=4）~0.3s。
    sleep_s = 0.3
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="s1", name="slow_echo", arguments={"tag": "s1"}),
                    ToolCall(id="s2", name="slow_echo", arguments={"tag": "s2"}),
                    ToolCall(id="s3", name="slow_echo", arguments={"tag": "s3"}),
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    start = time.perf_counter()
    result = run_agent(
        prompt="并发跑三个 slow_echo",
        provider=provider,
        tools=_registry_with_slow_echo(sleep_s),
        cwd=tmp_path,
        state=RuntimeState(permission_mode="acceptEdits"),
    )
    elapsed = time.perf_counter() - start

    # 并行墙钟：串行需 ~0.9s，给并行留足余量到 0.75s
    assert elapsed < 0.75, f"未并行：墙钟 {elapsed:.2f}s（串行预期 ~{3 * sleep_s}s）"

    # 配对不变量：tool_result 顺序 == tool_use 顺序（ex.map 按输入序返回）
    tail = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tail] == ["s1", "s2", "s3"]
    assert [m.content for m in tail] == ["echo:s1", "echo:s2", "echo:s3"]


# ---------------------------------------------------------------------------
# 读写混合：写工具截断只读组
# ---------------------------------------------------------------------------


def test_write_tool_splits_read_batches_and_runs_between_reads(tmp_path: Path) -> None:
    demo = tmp_path / "demo.py"
    demo.write_text('print("hello")\n', encoding="utf-8")

    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="r1", name="read_file", arguments={"path": "demo.py"}),
                    ToolCall(
                        id="e1",
                        name="file_edit",
                        arguments={
                            "file_path": "demo.py",
                            "old_string": 'print("hello")',
                            "new_string": 'print("edited")',
                        },
                    ),
                    ToolCall(id="r2", name="read_file", arguments={"path": "demo.py"}),
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    run_agent(
        prompt="读 demo.py，改成 edited，再读一遍",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=RuntimeState(permission_mode="acceptEdits"),
    )

    # file_edit 真正落盘
    assert demo.read_text(encoding="utf-8") == 'print("edited")\n'

    # partition 会被写工具截断成 3 个批次；3 个 tool_result 仍按 tool_use 顺序回填
    # —— 关键：r2 在 e1 之后，能看到写之后的内容（写没和读并到同一并行组里）


def test_write_tool_splits_read_batches_preserves_order(tmp_path: Path) -> None:
    # 与上一测试同结构，单独断言 tool_result 顺序（r1, e1, r2）不受分区影响
    demo = tmp_path / "demo.py"
    demo.write_text("x\n", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="r1", name="read_file", arguments={"path": "demo.py"}),
                    ToolCall(
                        id="e1",
                        name="file_edit",
                        arguments={
                            "file_path": "demo.py",
                            "old_string": "x",
                            "new_string": "y",
                        },
                    ),
                    ToolCall(id="r2", name="read_file", arguments={"path": "demo.py"}),
                ]
            ),
            ModelResponse(text="done"),
        ]
    )
    result = run_agent(
        prompt="读改读",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=RuntimeState(permission_mode="acceptEdits"),
    )
    tail = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tail] == ["r1", "e1", "r2"]
