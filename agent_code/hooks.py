from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# hooks.json 放在cwd下，是最顶级的hook配置
HOOKS_FILE = "hooks.json"


def load_hooks(cwd: Path) -> dict[str, list[dict[str, Any]]]:
    """加载hooks.json。文件不存在返回空dict"""
    file_path = cwd / HOOKS_FILE
    if not file_path.exists():
        return {}
    try:
        with open(file_path, encoding="utf-8") as hooks_file:
            data = json.load(hooks_file)
            # 主线使用 {"hooks": {"PostToolUse": [...]}}。
            # 也兼容直接写 {"PostToolUse": [...]}，方便手工调试。
            return data.get("hooks", data)
    except (json.JSONDecodeError, OSError) as exc:
        # 错误的JSON不应该阻塞Agent启动，打印警告即可
        print(f"[hook warning] failed to load {file_path}: {exc}")
        return {}


def _matches(tool_name: str, matcher: str) -> bool:
    """
    matcher 和 tool_name 的匹配规则：
    - "*" 匹配所有工具
    - 单值精确匹配
    - "a|b" 多值匹配
    不做正则和支持通配符，保持简单可预期。
    """
    if matcher == "*":
        return True
    if "|" in matcher:
        return tool_name in matcher.split("|")
    return matcher == tool_name


def _run_hook_command(
    command: str, input_data: dict[str, Any], cwd: Path, timeout: int = 30
) -> tuple[bool, str]:
    """执行一个hook command，subprocess跑，stdin传json，timeout默认30s。
    返回(success,output)二元组——success为True表示退出码0"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            input=json.dumps(input_data, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "hook timed out"
    except Exception as exc:
        return False, f"hook execution error: {exc}"


def _collect_commands(entry: dict[str, Any]) -> list[str]:
    """从 hook entry 收集要执行的命令列表。
    支持两种写法：顶层 "run" 单命令，或 "hooks": [{"type":"command","command":...}] 多命令。
    run_hooks / run_hooks_raw 共用，避免两处解析逻辑漂移。"""
    if "run" in entry:
        return [entry["run"]]
    commands: list[str] = []
    for h in entry.get("hooks", []):
        if isinstance(h, dict) and h.get("type") == "command":
            cmd = h.get("command", "")
            if cmd:
                commands.append(cmd)
    return commands


def run_hooks(
    event: str,
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: Path,
    tool_result: str = "",
) -> list[dict[str, Any]]:
    """在给定 event 下执行所有匹配 tool_name 的 hooks。
    返回一个 list，每个元素是 {"event": ..., "tool": ..., "success": bool, "output": str}。
    空 list 表示没有匹配到 hook。

    这是 harness 的 hook dispatch 入口——agent.py 在工具前后调用本函数。"""
    config = load_hooks(cwd)
    entries = config.get(event, [])
    results: list[dict[str, Any]] = []
    for entry in entries:
        matcher = entry.get("matcher", "*")
        if not _matches(tool_name, matcher):
            continue
        # 支持两种格式： run 单命令，或 hooks[].command 多命令（解析逻辑见 _collect_commands）
        for cmd in _collect_commands(entry):
            input_data = {
                "event": event,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": tool_result,
                "cwd": str(cwd),
            }
            success, output = _run_hook_command(cmd, input_data, cwd)
            results.append(
                {
                    "event": event,
                    "tool": tool_name,
                    "command": cmd,
                    "success": success,
                    "output": output,
                }
            )
    return results


def run_hooks_raw(
    event: str, payload: dict[str, Any], cwd: Path
) -> list[dict[str, Any]]:
    """跑没有工具上下文的 hook（如 Stop）。整个 payload 原样作为 stdin JSON 传给 hook。

    与 run_hooks 的区别：Stop 这类事件没有 tool_name，所以只认 matcher 为 "*"/空 的
    entry，其余一律跳过（_matches 需要 tool_name，这里用不上）。
    返回 [{"event","command","success","output"}]——无 "tool" 字段。

    agent.py 在模型给出无 tool_use 的最终回答后调用本函数：任一 hook 退出码非 0 且
    stdout/stderr 有内容，harness 就把该内容当"按这个继续"，注入合成 user 消息续跑一轮。
    """
    config = load_hooks(cwd)
    results: list[dict[str, Any]] = []
    for entry in config.get(event, []):
        if entry.get("matcher", "*") not in ("*", ""):
            continue
        for cmd in _collect_commands(entry):
            success, output = _run_hook_command(cmd, payload, cwd)
            results.append(
                {
                    "event": event,
                    "command": cmd,
                    "success": success,
                    "output": output,
                }
            )
    return results
