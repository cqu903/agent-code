from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import os
from anthropic import Anthropic


@dataclass
class ToolCall:
    # 模型请求 harness 执行这个工具。
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    # harness 把工具观察结果交回模型。
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    # 一次模型响应可以是最终文本，也可以是工具调用。
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str = "end_turn"


class Provider(Protocol):
    def complete(
        self, messages: list[dict[str, str]], tools: list[Any]
    ) -> ModelResponse: ...


class MockProvider:
    def complete(
        self, messages: list[dict[str, str]], tools: list[Any]
    ) -> ModelResponse:
        last = messages[-1]

        if last["role"] == "user":
            # 第一轮不直接回答，而是请求 harness 执行工具。需要先判断调用的工具
            user_content = last["content"]
            if user_content.startswith("用 echo 工具说"):
                text = (
                    user_content.replace("用 echo 工具说", "").strip() or user_content
                )
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_echo_1", name="echo", arguments={"text": text}
                        )
                    ],
                    stop_reason="tool_use",
                )
            elif user_content.startswith("用 uppercase 工具说"):
                text = (
                    user_content.replace("用 uppercase 工具说", "").strip()
                    or user_content
                )
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_uppercase_1",
                            name="uppercase",
                            arguments={"text": text},
                        )
                    ],
                    stop_reason="tool_use",
                )
            else:
                return ModelResponse(
                    text="请使用「用 echo 工具说 …」或「用 uppercase 工具说 …」。"
                )

        elif last["role"] == "tool":
            # 第二轮把工具观察结果变成最终回答。
            return ModelResponse(text=f"工具返回：{last['content']}")
        return ModelResponse(text="unsupported message role")


class AnthropicProvider:
    def __init__(
        self,
        model: str = "glm-5.1",
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

    def complete(
        self, messages: list[dict[str, str]], tools: list[Any]
    ) -> ModelResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return ModelResponse(
            text="\n".join(text_parts) or None, stop_reason=response.stop_reason
        )
