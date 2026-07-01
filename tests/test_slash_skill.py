"""回归测试：/skill slash 命令必须把拼好的 skill prompt 交给 Agent Loop。

保护的不变量：/skill <name> <task> 命中后，SlashResult.should_query=True，且
SlashResult.prompt 必须非空、包含 skill 正文与用户 task——cli.py 是用
slash_result.prompt 调 run_once 的；prompt 为空会让 Agent Loop 发一条 content
为空的 user 消息，被 GLM 以 1213「未正常接收到prompt参数」拒掉。

之前的 bug：_cmd_skill 只设了 message=（终端回显用），漏了 prompt=，导致
slash_result.prompt 取默认空串。
"""

from __future__ import annotations

from pathlib import Path

from agent_code.slash import SlashContext, dispatch_slash


def _make_skill(cwd: Path, name: str = "debug-test") -> None:
    skill_dir = cwd / ".agent" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: 调试失败的 Python 测试\n"
        "---\n"
        "# Debug Test\n"
        "失败时按步骤排查。\n",
        encoding="utf-8",
    )


def _ctx(cwd: Path) -> SlashContext:
    return SlashContext(
        cwd=cwd,
        permission_mode="default",
        model="m",
        provider="p",
        session_id=None,
    )


def test_skill_sets_nonempty_prompt_for_agent_loop(tmp_path: Path) -> None:
    """/skill 命中 → should_query=True 且 prompt 含 skill 正文与 task，不能是空串。"""
    _make_skill(tmp_path)
    result = dispatch_slash("/skill debug-test 检查最可疑的断言", _ctx(tmp_path))

    assert result.handled is True
    assert result.should_query is True, "/skill 必须触发 Agent Loop"
    # 回归点：prompt 不能是空串（cli.py 用它调 run_once）
    assert result.prompt, "slash_result.prompt 为空 → Agent Loop 拿到空 prompt"
    assert "Debug Test" in result.prompt, f"prompt 应含 skill 正文: {result.prompt!r}"
    assert (
        "检查最可疑的断言" in result.prompt
    ), f"prompt 应含用户 task: {result.prompt!r}"


def test_skill_task_keeps_spaces_between_words(tmp_path: Path) -> None:
    """多词 task 必须用空格连接，不能粘成一个词。

    回归点：_cmd_skill 曾用 "".join(_args[1:])，英文多词 task 经 shlex.split
    拆成多个 token 后会被粘成 checkthefailingtest。中文无空格不受影响，所以
    这个 bug 只在多词 task 上暴露——现有用中文 task 的测试抓不到。
    """
    _make_skill(tmp_path)
    result = dispatch_slash("/skill debug-test check the failing test", _ctx(tmp_path))
    assert result.should_query is True
    assert "check the failing test" in result.prompt
    assert "checkthefailingtest" not in result.prompt
