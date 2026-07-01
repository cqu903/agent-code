"""day-09 v3 工具白名单的回归测试。

覆盖 /skill 本轮 allowed_tools 的两道闸 + slash 携带，全程不依赖真实模型：
- §3.5 第一道闸：ToolRegistry.filtered 收敛给模型的工具 schema。
- §3.6 第二道闸：permissions.decide_permission 兜底 deny 越界工具。
- §3.2 SlashResult 把 SKILL.md frontmatter 里的 allowed_tools 透传出来。
"""

from __future__ import annotations

from pathlib import Path

from agent_code.permissions import PermissionRequest, decide_permission
from agent_code.slash import SlashContext, dispatch_slash
from agent_code.tools import default_tools

# ---------------------------------------------------------------- §3.6 权限兜底


def test_decide_permission_denies_tool_outside_skill_whitelist(tmp_path: Path) -> None:
    """白名单内的工具才放行：file_edit 不在 [read_file, grep] → deny。"""
    req = PermissionRequest(
        tool_name="file_edit",
        args={},
        mode="default",
        cwd=tmp_path,
        allowed_tools=["read_file", "grep"],
    )
    decision = decide_permission(req)
    assert decision.behavior == "deny"
    assert "file_edit" in (decision.message or "")


def test_decide_permission_allows_readonly_inside_whitelist(tmp_path: Path) -> None:
    """白名单内的只读工具不被误伤：read_file ∈ [read_file, grep] → allow。"""
    req = PermissionRequest(
        tool_name="read_file",
        args={},
        mode="default",
        cwd=tmp_path,
        allowed_tools=["read_file", "grep"],
    )
    assert decide_permission(req).behavior == "allow"


def test_decide_permission_none_whitelist_does_not_constrain(tmp_path: Path) -> None:
    """allowed_tools=None 表示不收敛：普通对话里 file_edit 仍走默认 ask，不被 deny。"""
    req = PermissionRequest(
        tool_name="file_edit",
        args={},
        mode="default",
        cwd=tmp_path,
        allowed_tools=None,
    )
    assert decide_permission(req).behavior == "ask"


def test_decide_permission_empty_whitelist_denies_everything(tmp_path: Path) -> None:
    """allowed_tools=[] 是纯文本 skill：本轮禁止任何工具，连只读的 read_file 也 deny。"""
    req = PermissionRequest(
        tool_name="read_file",
        args={},
        mode="default",
        cwd=tmp_path,
        allowed_tools=[],
    )
    assert decide_permission(req).behavior == "deny"


# -------------------------------------------------------------- §3.5 工具池过滤


def test_tool_registry_filtered_none_returns_self() -> None:
    """None=不收敛，直接返回同一个注册表。"""
    tools = default_tools()
    assert tools.filtered(None) is tools


def test_tool_registry_filtered_subset() -> None:
    """[read_file, grep] 只保留这两个工具的 schema，其余（file_edit/bash/...）不可见。"""
    tools = default_tools()
    visible = tools.filtered(["read_file", "grep"])
    names = {t.name for t in visible.list()}
    assert names == {"read_file", "grep"}


def test_tool_registry_filtered_empty() -> None:
    """[]=纯文本 skill，给模型看的工具面为空。"""
    tools = default_tools()
    assert tools.filtered([]).list() == []


# ------------------------------------------------------------- §3.2 slash 携带


def _make_skill_with_tools(cwd: Path) -> None:
    skill_dir = cwd / ".agent" / "skills" / "debug-test"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: debug-test\n"
        "description: 调试失败测试\n"
        'allowed_tools: ["read_file", "grep"]\n'
        "---\n"
        "# Debug Test\n",
        encoding="utf-8",
    )


def test_skill_slash_result_carries_allowed_tools(tmp_path: Path) -> None:
    """/skill 命中后，SlashResult.allowed_tools 必须等于 SKILL.md frontmatter 里的声明。"""
    _make_skill_with_tools(tmp_path)
    ctx = SlashContext(
        cwd=tmp_path,
        permission_mode="default",
        model="m",
        provider="p",
        session_id=None,
    )
    result = dispatch_slash("/skill debug-test 检查最可疑的断言", ctx)
    assert result.should_query is True
    assert result.allowed_tools == ["read_file", "grep"]
