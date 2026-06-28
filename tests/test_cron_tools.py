"""cron_tools 回归测试：

1. 三个 cron 工具通过 @register_tool 自动并入 default_tools() 注册表；
2. create / list / cancel 的 happy path 与错误分支行为正确。

工具函数在 REPL 外（_scheduler 为 None）会临时构造一个 CronScheduler，
持久化到 ctx.cwd/.agent/cron.json，因此每个 tmp_path 互不影响。
"""

from __future__ import annotations

from pathlib import Path

import agent_code.cron_tools as cron_tools
from agent_code.tools import ToolContext, default_tools


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def test_cron_tools_registered() -> None:
    # @register_tool 装饰 + tools.py 末尾导入 cron_tools，三者应自动出现在默认注册表
    names = {t.name for t in default_tools().list()}
    assert {"cron_create", "cron_list", "cron_cancel"} <= names

    tool = next(t for t in default_tools().list() if t.name == "cron_create")
    assert tool.parameters["required"] == ["slash", "every_seconds"]
    assert set(tool.parameters["properties"]) == {"slash", "every_seconds", "label"}


def test_cron_create_list_cancel(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    created = cron_tools.cron_create(
        {"slash": "/check", "every_seconds": 60, "label": "pr"}, ctx
    )
    assert "Cron job created" in created
    jid = created.split("created: ")[1].split(" - ")[0]

    listed = cron_tools.cron_list({}, ctx)
    assert "/check" in listed and "60s" in listed and jid in listed

    cancelled = cron_tools.cron_cancel({"id": jid}, ctx)
    assert "cancelled" in cancelled
    assert cron_tools.cron_list({}, ctx) == "(no cron jobs)"


def test_cron_create_validates_args(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert "missing required argument: 'slash'" in cron_tools.cron_create(
        {"every_seconds": 30}, ctx
    )
    assert "every_seconds must be positive" in cron_tools.cron_create(
        {"slash": "/x", "every_seconds": 0}, ctx
    )


def test_cron_cancel_unknown_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert "job not found" in cron_tools.cron_cancel({"id": "deadbeef"}, ctx)
    assert "missing required argument" in cron_tools.cron_cancel({}, ctx)
