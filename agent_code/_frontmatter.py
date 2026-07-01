from __future__ import annotations


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    把这种格式的文本：
    ---
    name: my-skill
    description: 做某件事
    allowed_tools: ["read", "write"]
    ---
    这里是正文 body…

    解析成 ({"name": "my-skill", "description": "做某件事", "allowed_tools": "[\"read\", \"write\"]"}, "这里是正文 body…")
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    raw_frontmatter = text[4:end]
    body = text[end + len("\n---\n") :]
    fields: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields, body.strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
