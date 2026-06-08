from __future__ import annotations

from pathlib import Path

# Agent.md大小上限：超过就截断保护，避免一份超长规则文件挤掉别的上下文
_MAX_AGENT_MD_BYTES = 50 * 1024


def load_agent_md(cwd: Path) -> str | None:
    """
    读取cwd下的Agent.md，包装成<project-rules>块
    文件不存在返回None，不是错误，只是没有配置
    """
    agent_md = cwd / "AGENT.md"
    if not agent_md.exists():
        return None
    content = agent_md.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return None
    if len(content.encode("utf-8")) > _MAX_AGENT_MD_BYTES:
        truncated = content.encode("utf-8")[:_MAX_AGENT_MD_BYTES].decode(
            "utf-8", errors="replace"
        )
        content = truncated + "\n\n[... AGENT.md truncated ad 50 KB...]"
    return f"<project-rules>\n{content}\n</project-rules>"
