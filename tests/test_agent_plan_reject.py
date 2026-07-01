"""回归测试：plan 审批被用户拒绝（N）后的回灌消息。

保护的不变量（Day 8 plan 闭环）：
用户在审批面板选 N 时，回灌给模型的消息应当**只通知拒绝**，不得夹带
「重新提交 / 再调一次 exit_plan_mode / 重新呈现」之类的指令。

复现的 bug：两条拒绝路径的文案都命令模型立刻重交计划——
  - exit_plan_mode 工具路径：obs = "Plan not approved. Revise the plan and call exit_plan_mode again."
  - 文本检查点路径：       user msg = "Plan not approved. Revise the plan and present it again."
模型照做，下一轮再次触发 confirm_plan，审批面板立刻又弹一次——看起来像拒绝没被处理、
陷入反复弹窗。弱模型（默认 glm-5.2）甚至会用近乎不变的计划反复重交。

修复后：回灌消息只是通知「用户未批准」，是否重交交给模型自己判断（用户选项 3：只通知、不指挥）。
"""

from __future__ import annotations

from pathlib import Path

from agent_code import agent
from agent_code.agent import run_agent
from agent_code.model import ModelResponse
from agent_code.runtime import RuntimeState
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


# 禁止出现的「指挥模型重新提交」类指令词（小写子串匹配）。
# 修复后的纯通知消息 "The user did not approve the plan." 一个都不含。
_REJECT_DIRECTIVES = ("again", "revise", "present", "call exit_plan_mode")


def _assert_plain_notification(content: str | None) -> None:
    """回灌内容必须通知拒绝，且不含任何重新提交的指令。"""
    assert content is not None, "拒绝消息内容不应为 None"
    lowered = content.lower()
    assert "not approve" in lowered, f"拒绝消息应通知未批准，实际：{content!r}"
    for bad in _REJECT_DIRECTIVES:
        assert (
            bad not in lowered
        ), f"拒绝消息不得指挥模型重新提交（含 {bad!r}），实际：{content!r}"


def test_text_plan_rejection_notifies_without_command(
    tmp_path: Path, monkeypatch
) -> None:
    """plan 模式下模型把计划写成纯文本、用户拒绝：回灌的 user 消息只通知、不指挥。"""
    # 用户在审批面板选了 N
    monkeypatch.setattr(agent, "confirm_plan", lambda plan: False)
    provider = FakeProvider(
        [ModelResponse(text="计划：新建 day8_demo.py，放 fibonacci + 测试。")]
    )
    state = RuntimeState(permission_mode="plan")

    result = run_agent(
        prompt="写个 day8_demo.py",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=state,
        max_steps=1,  # 跑完「文本→拒绝→回灌」这一轮即止
    )

    # 最后一条是回灌的拒绝通知（user 角色）
    rejection = result.messages[-1]
    assert rejection.role == "user"
    _assert_plain_notification(rejection.content)
    # 拒绝不应翻模式：仍停留在 plan
    assert state.permission_mode == "plan"


def test_exit_plan_mode_rejection_notifies_without_command(
    tmp_path: Path, monkeypatch
) -> None:
    """模型调 exit_plan_mode 提交计划、用户拒绝：回灌的 tool_result 同样只通知、不指挥。"""
    monkeypatch.setattr(agent, "confirm_plan", lambda plan: False)
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="plan_1",
                        name="exit_plan_mode",
                        arguments={
                            "plan_summary": "计划：新建 day8_demo.py，放 fibonacci + 测试。"
                        },
                    )
                ]
            )
        ]
    )
    state = RuntimeState(permission_mode="plan")

    result = run_agent(
        prompt="写个 day8_demo.py",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=state,
        max_steps=1,
    )

    # 最后一条是拒绝的 tool_result（is_error=True）
    rejection = result.messages[-1]
    assert rejection.role == "tool"
    assert rejection.is_error
    assert rejection.tool_call_id == "plan_1"
    _assert_plain_notification(rejection.content)
    assert state.permission_mode == "plan"
