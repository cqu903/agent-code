"""is_read_only 标记回归测试。

并发工具编排：只读工具标记 is_read_only=True（可并行），
写类 / 交互类 / 网络类保持 False（串行）。

EXPECTED 是真值来源快照——新增 / 删除 / 重命名工具时本测试会显式失败，
强制作者复核该工具是否只读，避免白名单与真实注册名漂移、或新工具被遗漏。
"""

from __future__ import annotations

from agent_code.tools import default_tools

# name -> 是否只读。覆盖 default_tools() 中所有已注册工具。
EXPECTED: dict[str, bool] = {
    # 只读：无副作用，可并行
    "read_file": True,
    "list_files": True,
    "glob": True,
    "grep": True,
    "project_tree": True,
    "git_status": True,
    "git_diff": True,
    "system_date": True,
    "echo": True,
    "memory_recall": True,
    "cron_list": True,
    "todo_read": True,
    "skill_list": True,
    "skill_load": True,
    # 写类：有副作用，串行
    "file_write": False,
    "file_edit": False,
    "bash": False,
    "memory_write": False,
    "todo_write": False,
    "cron_create": False,
    "cron_cancel": False,
    # 计划模式：翻转权限模式 / 阻塞等批准，串行
    "enter_plan_mode": False,
    "exit_plan_mode": False,
    # 交互类 / 网络类：默认串行（前者需用户确认，后者有外部副作用）
    "ask_user_question": False,
    "web_fetch": False,
    "web_search": False,
}


def test_is_read_only_flags_match_snapshot() -> None:
    flagged = {t.name: t.is_read_only for t in default_tools().list()}
    only_snapshot = set(EXPECTED) - set(flagged)
    only_registered = set(flagged) - set(EXPECTED)
    assert not only_snapshot and not only_registered, (
        "工具集与快照不一致，新增 / 删除 / 重命名工具时请同步更新 EXPECTED："
        f" 仅快照={sorted(only_snapshot)}，仅注册={sorted(only_registered)}"
    )
    wrong = {n: flagged[n] for n in EXPECTED if flagged[n] != EXPECTED[n]}
    assert not wrong, f"is_read_only 标记与快照不符: {wrong}"


def test_read_only_subset_is_non_empty_and_flagged_true() -> None:
    # 直接锁定白名单语义：这些名字必须存在且为 True。
    flagged = {t.name: t.is_read_only for t in default_tools().list()}
    read_only = {n for n, ro in EXPECTED.items() if ro}
    assert read_only <= flagged.keys()
    assert all(flagged[n] for n in read_only)
