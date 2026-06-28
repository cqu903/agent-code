"""文件系统类工具：读、列、glob、grep、目录树、写、编辑。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import ToolContext, register_tool
from ..file_history import backup
from ..fs_safety import (
    apply_single_replace,
    ensure_text_file,
    ensure_whithin_size,
    resolve_in_cwd,
    should_skip,
    truncate_output,
)


@register_tool(
    name="read_file",
    description="Read a text file inside the project. Path is relative to cwd.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside cwd.",
            },
        },
        "required": ["path"],
    },
)
def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    # 模型给出相对路径，fs_safety将路径锁回cwd内，探测二进制再卡大小上限
    path_str = args.get("path", "")
    if not path_str:
        return "error: missing required argument 'path'"
    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
        ensure_text_file(path)
        ensure_whithin_size(path)
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        return f"error: {exc}"
    # 记录”模型看到的版本。“
    ctx.read_state.record(path, text)
    return truncate_output(text)


@register_tool(
    name="list_files",
    description="List files and directories at a path inside cwd.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path; defaults to '.'.",
                "default": ".",
            },
        },
        "required": [],
    },
)
def list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", ".")
    try:
        base = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as exc:
        return f"error: {exc}"
    if not base.is_dir():
        return f"error: {path_str} is not a directory"
    entries: list[str] = []
    for child in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        rel = child.relative_to(ctx.cwd)
        if should_skip(rel, ctx.skip_policy):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    return truncate_output("\n".join(entries) or "(empty)")


@register_tool(
    name="glob",
    description="Find files by glob pattern, e.g. '**/*.py'.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
        },
        "required": ["pattern"],
    },
)
def glob(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "error: missing required argument 'pattern'"
    matches: list[Path] = []
    try:
        for path in ctx.cwd.rglob(pattern):
            rel = path.relative_to(ctx.cwd)
            if should_skip(rel, ctx.skip_policy):
                continue
            matches.append(path)
    except NotImplementedError as exc:
        return f"error: {exc}"
    # 按照修改时间排序，让最近改过的文件排在最前面
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    matches = matches[:200]

    lines = [str(p.relative_to(ctx.cwd)) for p in matches]
    return truncate_output("\n".join(lines) or "(no matches)")


@register_tool(
    name="grep",
    description="Search file contents with a regular expression (ripgrep if available).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression."},
            "path": {
                "type": "string",
                "description": "Relative path; defaults to '.'.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob filter, e.g. '*.py'.",
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive match.",
                "default": False,
            },
        },
        "required": ["pattern"],
    },
)
def grep(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "error: missing required argument 'pattern'"
    path_arg = args.get("path", ".")
    glob_arg = args.get("glob")
    ignore_case = bool(args.get("ignore_case", False))

    try:
        base = resolve_in_cwd(ctx.cwd, path_arg)
    except ValueError as exc:
        return f"error: {exc}"

    # 系统安装了ripgrep就走它，否则退化为纯python
    if shutil.which("rg"):
        return _grep_ripgrep(pattern, base, glob_arg, ignore_case, ctx)
    return _grep_python(pattern, base, glob_arg, ignore_case, ctx)


def _grep_ripgrep(
    pattern: str, base: Path, glob_arg: str | None, ignore_case: bool, ctx: ToolContext
) -> str:
    # ripgrep 自带 .gitignore 解析和 VCS 目录跳过，我们只需要追加自定义 skip。
    args: list[str] = ["rg", "--line-number", "--no-heading", "--max-columns", "500"]
    if ignore_case:
        args.append("-i")
    for name in ctx.skip_policy.skip_dirs:
        args.extend(["--glob", f"!{name}/**"])
    if glob_arg:
        args.extend(["--glob", glob_arg])
    args.append(pattern)
    # rg 必须收绝对路径才能让 --glob 的相对规则可预测；
    # 但输出给模型前要把每行的绝对前缀切回相对路径，省 token、和 _grep_python 保持一致。
    args.append(str(base))
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"error: {exc}"

    # ripgrep 没匹配会返回 exit code 1，这不是错；真错才看 stderr。
    if proc.returncode not in (0, 1):
        return f"error: rg: {proc.stderr.strip() or proc.returncode}"
    return truncate_output(
        _relativize_rg_output(proc.stdout, ctx.cwd) or "(no matches)"
    )


def _relativize_rg_output(stdout: str, cwd: Path) -> str:
    # rg 每行形如 "/abs/path:lineno:content"。命中 cwd 前缀的就切成相对路径，
    # 不命中（罕见）原样保留，避免吞掉模型可能想看到的诊断信息。
    cwd_prefix = f"{cwd}/"
    lines = [
        line[len(cwd_prefix) :] if line.startswith(cwd_prefix) else line
        for line in stdout.splitlines()
    ]
    return "\n".join(lines).strip()


def _grep_python(
    pattern: str, base: Path, glob_arg: str | None, ignore_case: bool, ctx: ToolContext
) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"error: invalid regex: {exc}"
    if base.is_file():
        candidates: list[Path] = [base]
    else:
        candidates = []
        try:
            for path in base.rglob(glob_arg or "*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(ctx.cwd)
                if should_skip(rel, ctx.skip_policy):
                    continue
                candidates.append(path)
        except NotImplementedError as exc:
            return f"error: {exc}"
    hits: list[str] = []
    for path in candidates:
        try:
            ensure_text_file(path)
        except ValueError:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(ctx.cwd)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{rel}:{lineno}:{line}")
    return truncate_output("\n".join(hits) or "(no matches)")


@register_tool(
    name="project_tree",
    description="Show the project directory tree from cwd.",
    parameters={
        "type": "object",
        "properties": {
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursion depth.",
                "default": 3,
            },
        },
        "required": [],
    },
)
def project_tree(args: dict[str, Any], ctx: ToolContext) -> str:
    max_depth = int(args.get("max_depth", 5))
    max_nodes = 200
    lines: list[str] = [f"{ctx.cwd.name}/"]
    nodes = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal nodes
        if depth > max_depth:
            return
        children = sorted(
            (
                c
                for c in directory.iterdir()
                if not should_skip(c.relative_to(ctx.cwd), ctx.skip_policy)
            ),
            key=lambda p: (not p.is_dir(), p.name),
        )
        for child in children:
            if nodes >= max_nodes:
                if nodes == max_nodes:
                    lines.append("  " * depth + "...[truncated]")
                    nodes += 1
                return
            suffix = "/" if child.is_dir() else ""
            lines.append(" " * depth + child.name + suffix)
            nodes += 1
            if child.is_dir():
                walk(child, depth + 1)

    walk(ctx.cwd, 1)
    return truncate_output("\n".join(lines))


@register_tool(
    name="file_write",
    description="Write or overwrite a file. Path is relative to cwd.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path inside cwd.",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
            },
        },
        "required": ["file_path", "content"],
    },
)
def file_write(args: dict[str, Any], ctx: ToolContext) -> str:
    """整文件覆盖写入，前置校验由agent.py拦截块完成"""
    path_str = args.get("file_path", "")
    content = args.get("content", "")
    if not path_str:
        return "error: missing required argument 'file_path'"
    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as exc:
        return f"error: {exc}"

    if path.exists():
        try:
            old = path.read_text(encoding="utf-8")
            backup(ctx.cwd, path, old)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    # 写盘后刷新 read_state，让下一次编辑基于最新内容
    ctx.read_state.record(path, content)
    return f"Wrote {len(content)} chars to {path_str}"


@register_tool(
    name="file_edit",
    description=(
        "Edit a file by replacing old_string with new_string. "
        "old_string must be unique in the file (or use replace_all=True). "
        "You must read the file before editing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path inside cwd.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to replace. Must match including whitespace and indentation.",
            },
            "new_string": {
                "type": "string",
                "description": "String to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences. Default false.",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
)
def file_edit(args: dict[str, Any], ctx: ToolContext) -> str:
    """字符串替换编辑。前置校验在 agent.py 拦截块里完成。"""
    path_str = args.get("file_path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))
    if not path_str:
        return "error: missing required argument 'file_path'"
    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as exc:
        return f"error: {exc}"

    try:
        content = path.read_text(encoding="utf-8")
        backup(ctx.cwd, path, content)
    except (FileNotFoundError, IsADirectoryError) as exc:
        return f"error: {exc}"

    # 防 race：agent.py 已经做过一次 apply_single_replace 算 diff，
    # 如果 confirm 那一刻到现在 old_content 又被外部改过，这里会再兜一次。
    new_content, err = apply_single_replace(
        content, old_string, new_string, replace_all
    )
    if err:
        return err

    path.write_text(new_content, encoding="utf-8")
    ctx.read_state.record(path, new_content)
    return f"Edited {path_str}: replaced {len(old_string)} chars with {len(new_string)} chars"
