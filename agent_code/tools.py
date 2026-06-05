from __future__ import annotations

import os
import re
import shutil
import time
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime

from rich.console import Console
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from urllib.parse import urlparse
import html2text
import httpx

from .fs_safety import (
    ReadFileState,
    SkipPolicy,
    ensure_text_file,
    ensure_whithin_size,
    resolve_in_cwd,
    should_skip,
    truncate_output,
)


@dataclass
class ToolContext:
    # 工具运行时上下文，装cwd，skip规则，ReadFileState
    cwd: Path
    skip_policy: SkipPolicy = field(default_factory=SkipPolicy.default)
    read_state: ReadFileState = field(default_factory=ReadFileState)


ToolFunc = Callable[[dict[str, Any], ToolContext], str]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


def echo(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", ""))


def system_date(args: dict[str, Any], ctx: ToolContext) -> str:
    tz_time = args.get("timezone")
    if tz_time:
        try:
            tz = ZoneInfo(tz_time)
        except ZoneInfoNotFoundError:
            return f"unknown timezone: {tz_time}"
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


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
        rel = Path.relative_to(ctx.cwd)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{rel}:{lineno}:{line}")
    return truncate_output("\n".join(hits) or "(no matches)")


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


WEB_USER_AGENT = "agent-code/0.1 (+https://example.com/agent-code)"
WEB_FETCH_MAX_BYTES = 10 * 1024 * 1024
WEB_FETCH_MAX_CHARS = 20_000
WEB_URL_MAX_LENGTH = 2000
WEB_FETCH_TIMEOUT_S = 30.0
WEB_SEARCH_TIMEOUT_S = 15.0
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
WEB_SEARCH_RETRIES = 3


def _web_http_client(timeout: float) -> httpx.Client:
    # trust_env：读取 HTTPS_PROXY / ALL_PROXY（可在 agent_code/.env 中配置）
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
    )


def _vaidate_url(url: str) -> None:
    if len(url) > WEB_URL_MAX_LENGTH:
        raise ValueError(f"url too long: {len(url)}>{WEB_URL_MAX_LENGTH}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme: {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise ValueError("url with credentials is not allowed")
    if not parsed.hostname or "." not in parsed.hostname:
        raise ValueError(f"invalid hostname: {parsed.hostname}")


def _html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0  # 关掉硬换行，保留模型上下文里更长的段落。
    converter.ignore_images = True
    converter.ignore_emphasis = False
    return converter.handle(html).strip()


def web_fetch(args: dict[str, Any], ctx: ToolContext) -> str:
    url = args.get("url", "")
    if not url:
        return "error: missing required argument 'url'"
    try:
        _vaidate_url(url)
    except ValueError as exc:
        return f"error: {exc}"

    headers = {
        "User-Agent": WEB_USER_AGENT,
        "Accept": "text/html,text/*;q=0.9,*/*;q=0.5",
    }
    try:
        with _web_http_client(WEB_FETCH_TIMEOUT_S) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"error: {exc}"
    if len(resp.content) > WEB_FETCH_MAX_BYTES:
        return f"error: response too large: {len(resp.content)}>{WEB_FETCH_MAX_BYTES}"
    content_type = resp.headers.get("content-type", "").lower()
    if "text/html" in content_type or "application/xhtml" in content_type:
        body = _html_to_markdown(resp.text)
    elif (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
    ):
        body = resp.text
    else:
        return f"error: unsupported content-type: {content_type or '(none)'}"

    return truncate_output(body, max_chars=WEB_FETCH_MAX_CHARS)


def _tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set")

    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None
    with _web_http_client(WEB_SEARCH_TIMEOUT_S) as client:
        for attempt in range(WEB_SEARCH_RETRIES):
            try:
                resp = client.post(TAVILY_SEARCH_URL, json=payload, headers=headers)
                resp.raise_for_status()
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 >= WEB_SEARCH_RETRIES:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:
            assert last_exc is not None
            raise last_exc

    results: list[dict[str, str]] = []
    for item in resp.json().get("results", []):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results


def web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = args.get("query", "")
    if not query:
        return "error: missing required argument 'query'"
    max_results = max(1, min(int(args.get("max_results", 5)), 10))
    try:
        results = _tavily_search(query, max_results=max_results)
    except ValueError as exc:
        return f"error: {exc}"
    except httpx.HTTPError as exc:
        msg = f"error: Tavily search failed ({TAVILY_SEARCH_URL}): {exc}"
        err = str(exc)
        if "SSL" in err or "EOF" in err or "connect" in err.lower():
            proxy = (
                os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("ALL_PROXY")
                or os.environ.get("all_proxy")
            )
            if proxy:
                msg += f" (已检测到代理 {proxy!r}，若仍失败请检查代理是否可用)"
            else:
                msg += (
                    "。多为网络无法直连 api.tavily.com："
                    "在 agent_code/.env 添加 HTTPS_PROXY=http://127.0.0.1:端口"
                    "（Clash 等本地代理端口），或切换网络后重试"
                )
        return msg
    if not results:
        return "(no results)"
    lines = [f"- {r['title']}\n  {r['url']}" for r in results]
    return truncate_output("\n".join(lines))


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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    # 写盘后刷新 read_state，让下一次编辑基于最新内容
    ctx.read_state.record(path, content)
    return f"Wrote {len(content)} chars to {path_str}"


class ToolRegistry:
    def __init__(self) -> None:
        # 注册表是工具名和python函数之间的harness边界
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        # 未知工具也返回observation，不让Agent Loop崩溃
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=call.id,
            content=tool.run(call.arguments, ctx),
        )

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def print_help(self, console: Console | None = None) -> None:
        out = console or Console()
        out.print("[bold]Available tools:[/bold]")
        if not self._tools:
            out.print("[dim](none)[/dim]")
            return
        for name in sorted(self._tools):
            tool = self._tools[name]
            out.print(f"  [cyan]{name}[/cyan] — {tool.description}")


def default_tools() -> ToolRegistry:
    # 后续会加入文件工具和bash工具
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Return the input text",
            run=echo,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to return."}
                },
                "required": ["text"],
            },
        )
    )

    registry.register(
        Tool(
            name="system_date",
            description="Return the current system date and time.",
            run=system_date,
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone name (e.g. UTC, Asia/Shanghai, America/New_York). Defaults to the system local timezone.",
                    }
                },
                "required": [],
            },
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a text file inside the project. Path is relative to cwd.",
            run=read_file,
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
    )
    registry.register(
        Tool(
            name="list_files",
            description="List files and directories at a path inside cwd.",
            run=list_files,
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
    )
    registry.register(
        Tool(
            name="glob",
            description="Find files by glob pattern, e.g. '**/*.py'.",
            run=glob,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern."},
                },
                "required": ["pattern"],
            },
        )
    )
    registry.register(
        Tool(
            name="grep",
            description="Search file contents with a regular expression (ripgrep if available).",
            run=grep,
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
    )
    registry.register(
        Tool(
            name="project_tree",
            description="Show the project directory tree from cwd.",
            run=project_tree,
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
    )
    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch a URL and return its content as markdown (or raw text).",
            run=web_fetch,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."},
                },
                "required": ["url"],
            },
        )
    )
    registry.register(
        Tool(
            name="web_search",
            description="Search the web (Tavily) and return top results.",
            run=web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    )
    registry.register(
        Tool(
            name="file_write",
            description="Write or overwrite a file. Path is relative to cwd.",
            run=file_write,
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
    )
    return registry
