"""回归测试：acceptEdits 模式下 file_edit 必须真正写盘。

复现的 bug：agent.py 中 file_write/file_edit 安全校验分支里的 `continue`
缩进错误，导致每一次文件编辑都会在 is_dir 检查之后无条件 continue，
跳过 tools.run()——文件未被修改，agent 却因为收不到 observation 而幻觉成功。
"""

from __future__ import annotations

from pathlib import Path

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


def test_file_edit_writes_disk_in_accept_edits(tmp_path: Path) -> None:
    # 1. 准备一个待编辑文件
    demo = tmp_path / "demo.py"
    demo.write_text('print("hello")\n', encoding="utf-8")

    # 2. FakeProvider 回放模型的真实行为：read_file -> file_edit -> final
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "demo.py"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="file_edit",
                        arguments={
                            "file_path": "demo.py",
                            "old_string": 'print("hello")',
                            "new_string": 'print("hello day7")',
                        },
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    # 3. 运行 agent loop（acceptEdits 模式，无需确认 UI）
    run_agent(
        prompt="读 demo.py 再把 hello 改成 hello day7",
        provider=provider,
        tools=default_tools(),
        cwd=tmp_path,
        state=RuntimeState(permission_mode="acceptEdits"),
    )

    # 4. 断言文件确实被改写——bug 存在时这行会失败
    assert demo.read_text(encoding="utf-8") == 'print("hello day7")\n'
