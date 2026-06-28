from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from .messages import Message, sanitize_value
from .tools import Tool, ToolCall


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    provider_data: dict[str, Any] | None = None
    stop_reason: str = "end_turn"


class Provider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        system: str | None = None,
    ) -> ModelResponse: ...


class AnthropicProvider:
    def __init__(
        self,
        model: str = "glm-5.2",
        max_tokens: int = 8192,
        base_url: str | None = None,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not api_key:
            raise RuntimeError(
                "请先设置 ANTHROPIC_AUTH_TOKEN，例如：ANTHROPIC_AUTH_TOKEN='sk-...'"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "ANTHROPIC_BASE_URL",
            "https://api.deepseek.com/anthropic",
        )
        self.client = Anthropic(api_key=api_key, base_url=self.base_url)

    def _to_anthropic_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_tool_input(self, value: object) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _content_block_to_dict(self, block: Any) -> dict[str, Any]:
        if hasattr(block, "model_dump"):
            return block.model_dump(exclude_none=True)
        if hasattr(block, "dict"):
            return block.dict(exclude_none=True)
        data = {"type": block.type}
        for name in {"text", "id", "name", "input", "thinking", "signature"}:
            if hasattr(block, name):
                data[name] = getattr(block, name)
        return data

    def _build_assistant_wire(self, message: Message) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for call in message.tool_calls or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return {"role": "assistant", "content": content}

    def _to_wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "user":
                wire.append({"role": "user", "content": message.content or ""})
                index += 1
                continue

            if message.role == "assistant":
                provider_data = message.provider_data
                if (
                    provider_data
                    and provider_data.get("format", "anthropic") == "anthropic"
                    and "content" in provider_data
                ):
                    wire.append(
                        {
                            "role": "assistant",
                            "content": provider_data["content"],
                        }
                    )
                else:
                    wire.append(self._build_assistant_wire(message))
                index += 1
                continue

            if message.role == "tool":
                tool_results: list[dict[str, Any]] = []
                while index < len(messages) and messages[index].role == "tool":
                    tool_message = messages[index]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_message.tool_call_id,
                            "content": tool_message.content or "",
                            "is_error": tool_message.is_error,
                        }
                    )
                    index += 1
                wire.append({"role": "user", "content": tool_results})
                continue

            index += 1
        return wire

    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        system: str | None = None,
    ) -> ModelResponse:
        wire_messages = self._to_wire_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire_messages,
        }
        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        kwargs = sanitize_value(kwargs)

        response = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            assistant_content.append(self._content_block_to_dict(block))

            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=self._parse_tool_input(block.input),
                    )
                )

        provider_data = (
            {"format": "anthropic", "content": assistant_content}
            if assistant_content
            else None
        )
        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            provider_data=provider_data,
            stop_reason=response.stop_reason or "end_turn",
        )
