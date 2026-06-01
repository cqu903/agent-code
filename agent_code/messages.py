from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .tools import ToolCall


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    content: str | None = None
    is_error: bool = False
    provider_data: dict[str, Any] | None = None
