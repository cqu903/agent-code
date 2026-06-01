from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from .llm_log import LLMLogger
from .tools import Tool, ToolCall


@dataclass
class ModelResponse:
    # 一次模型响应可以是最终文本，也可以是工具调用。
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str = "end_turn"


class Provider(Protocol):
    def complete(
        self, messages: list[dict[str, str]], tools: list[Tool] | None = None
    ) -> ModelResponse: ...


class AnthropicProvider:
    def __init__(
        self,
        model: str = "glm-5.1",
        max_tokens: int = 8192,
        base_url: str | None = None,
        llm_logger: LLMLogger | None = None,
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
        self.llm_logger = llm_logger
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

    def complete(
        self, messages: list[dict[str, str]], tools: list[Tool] | None = None
    ) -> ModelResponse:
        # 先准备一次模型请求的基础参数，messages是Agent Loop累积出来的上下文
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        # 如果registry里有工具，就把Tool翻译成Anthropic的tools格式
        # 这里只是告诉模型有哪些工具
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        turn = 0
        if self.llm_logger:
            turn = self.llm_logger.next_turn()
            self.llm_logger.log_request(turn, kwargs)

        started = time.monotonic()
        response = self.client.messages.create(**kwargs)
        duration_ms = int((time.monotonic() - started) * 1000)

        if self.llm_logger:
            self.llm_logger.log_response(
                turn,
                response.model_dump(exclude_none=True),
                duration_ms=duration_ms,
            )

        # Claude/DeepSeek 可能同时返回 text block 和 tool_use block。
        # text_parts 收集普通回答；tool_calls 收集"模型想调用工具"的请求
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            # 原样保存 assistant content，后面 agent.py 会把它放回 messages。
            # 这能保留 thinking / signature 等额外 block，避免下一轮请求丢上下文。
            assistant_content.append(self._content_block_to_dict(block))

            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # provider只负责把外部协议翻译成内部的ToolCall
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=self._parse_tool_input(block.input),
                    )
                )
        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            assistant_content=assistant_content or None,
            stop_reason=response.stop_reason or "end_turn",
        )
