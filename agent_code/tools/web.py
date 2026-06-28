"""网络类工具：web_fetch、web_search（Tavily）。"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

import html2text
import httpx

from .base import ToolContext, register_tool
from ..fs_safety import truncate_output

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


@register_tool(
    name="web_fetch",
    description="Fetch a URL and return its content as markdown (or raw text).",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
        },
        "required": ["url"],
    },
)
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


@register_tool(
    name="web_search",
    description="Search the web (Tavily) and return top results.",
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
