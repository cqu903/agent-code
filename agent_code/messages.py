from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .tools import ToolCall

ProviderFormat = Literal["anthropic", "openai"]


def sanitize_text(text: str) -> str:
    """Remove lone UTF-16 surrogates that break JSON / UTF-8 serialization."""
    return "".join(
        "\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text
    )


def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    is_error: bool = False
    provider_data: dict[str, Any] | None = None


def message_to_record(msg: Message) -> dict[str, Any]:
    """把 Message 转成 JSONL 可落盘的 dict（不含 timestamp）。

    字段设计兼容 Anthropic / OpenAI 两种 wire format：
    - 通用字段：role, content, tool_calls, tool_call_id, is_error
    - provider_data.format 标记原始 API 形态（anthropic content blocks / openai tool_calls wire）
    """
    record: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        record["content"] = sanitize_text(msg.content)
    if msg.tool_calls:
        record["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        record["tool_call_id"] = msg.tool_call_id
    if msg.is_error:
        record["is_error"] = True
    if msg.provider_data is not None:
        record["provider_data"] = sanitize_value(msg.provider_data)
    return record


def message_from_record(data: dict[str, Any]) -> Message | None:
    """从 JSONL dict 还原 Message。兼容旧版仅含 role/content 的记录。"""
    role = data.get("role")
    if role not in ("user", "assistant", "tool"):
        return None

    tool_calls: list[ToolCall] | None = None
    raw_calls = data.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        parsed: list[ToolCall] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            arguments = call.get("arguments")
            parsed.append(
                ToolCall(
                    id=str(call.get("id", "")),
                    name=str(call.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        if parsed:
            tool_calls = parsed

    provider_data = data.get("provider_data")
    if provider_data is not None and not isinstance(provider_data, dict):
        provider_data = None
    elif provider_data is not None:
        provider_data = sanitize_value(provider_data)

    content = data.get("content")
    if isinstance(content, str):
        content = sanitize_text(content)

    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
        is_error=bool(data.get("is_error", False)),
        provider_data=provider_data,
    )
