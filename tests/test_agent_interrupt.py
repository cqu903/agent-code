"""回归测试：ESC 半步中断。

保护的不变量：
1. provider.complete() 返回带 tool_use 的响应后、工具执行前，若 abort_event 已 set，
   run_agent 必须短路返回 final="interrupted"，且不真正执行任何工具。
2. 配对不变量——assistant 给了 N 个 tool_use，就必须紧跟 N 个 tool_result
   （按 tool_use_id 一一配对，is_error=True），否则下次请求会被 Anthropic API 拒。
   本项目用扁平 Message，多个 role="tool" 连续排列，_to_wire_messages 合并成一条 user 消息。
3. 中断时 assistant(tool_calls) + 全部 tool_result 都要落盘到 session，可被 --resume 还原。
"""

from __future__ import annotations

from pathlib import Path

from agent_code.agent import run_agent
from agent_code.model import ModelResponse
from agent_code.runtime import RuntimeState
from agent_code.session import Session
from agent_code.tools import ToolCall, default_tools


class FakeProvider:
    """按顺序回放预设的 ModelResponse，模拟 Provider.complete()。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._i = 0

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _two_tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(id="call_a", name="read_file", arguments={"path": "nope1.py"}),
            ToolCall(id="call_b", name="read_file", arguments={"path": "nope2.py"}),
        ]
    )


def test_interrupt_short_circuits_before_tool_execution(tmp_path: Path) -> None:
    provider = FakeProvider([_two_tool_response()])
    state = RuntimeState(permission_mode="default")
    # 模拟用户在 provider.complete() 阻塞期间按下 ESC：返回时 abort_event 已 set。
    state.abort_event.set()

    result = run_agent(
        prompt="读两个文件",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=state,
    )

    # 短路返回 interrupted，没有走到工具执行循环（否则 FakeProvider 耗尽会抛 IndexError）
    assert result.final == "interrupted"

    # 配对不变量：两个 tool_use 各有一个 is_error 的 tool_result，id 一一对应
    tail = result.messages[-2:]
    assert [m.role for m in tail] == ["tool", "tool"]
    assert all(m.is_error for m in tail)
    assert all(m.content == "Interrupted by user" for m in tail)
    assert {m.tool_call_id for m in tail} == {"call_a", "call_b"}

    # 紧邻 tool_result 之前的是带 tool_calls 的 assistant 消息
    assistant = result.messages[-3]
    assert assistant.role == "assistant"
    assert {c.id for c in (assistant.tool_calls or [])} == {"call_a", "call_b"}


def test_interrupt_persists_assistant_and_all_tool_results(tmp_path: Path) -> None:
    """session 落盘必须包含 assistant(tool_calls) + 全部 N 个 tool_result，
    这样 --resume 重建历史时配对不变量依然成立（不能只落最后两条）。"""
    provider = FakeProvider([_two_tool_response()])
    state = RuntimeState(permission_mode="default")
    state.abort_event.set()
    session = Session.create(tmp_path)

    run_agent(
        prompt="读两个文件",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=state,
        session=session,
    )

    roles = [m.role for m in session.history]
    # 顺序：user(prompt) → assistant(tool_calls) → tool → tool
    assert roles == ["user", "assistant", "tool", "tool"]
    tool_msgs = [m for m in session.history if m.role == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"call_a", "call_b"}
    assert all(m.is_error and m.content == "Interrupted by user" for m in tool_msgs)


def test_interrupt_without_tool_calls_persists_assistant_text(tmp_path: Path) -> None:
    """模型只回了文本（无 tool_calls）时按 ESC：文本仍落盘，final 标记 interrupted。"""
    provider = FakeProvider([ModelResponse(text="halfway answer")])
    state = RuntimeState(permission_mode="default")
    state.abort_event.set()
    session = Session.create(tmp_path)

    result = run_agent(
        prompt="随便聊聊",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=state,
        session=session,
    )

    assert result.final == "interrupted"
    # assistant 文本回复落盘，没有多余 tool 消息
    assert [m.role for m in session.history] == ["user", "assistant"]
    assert session.history[-1].content == "halfway answer"
