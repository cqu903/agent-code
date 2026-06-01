from __future__ import annotations

from dataclasses import dataclass

from .messages import Message
from .model import Provider
from .tools import ToolRegistry


@dataclass
class AgentResult:
    final: str
    trace: list[str]
    messages: list[Message]


def run_agent(
    prompt: str, provider: Provider, tools: ToolRegistry, max_steps: int = 99
) -> AgentResult:
    messages: list[Message] = [Message(role="user", text=prompt)]
    trace: list[str] = []

    for _step in range(max_steps):
        response = provider.complete(messages, tools=tools.list())
        messages.append(
            Message(
                role="assistant",
                text=response.text,
                tool_calls=response.tool_calls,
                provider_data=response.provider_data,
            )
        )

        if not response.tool_calls:
            final = response.text or ""
            trace.append(f"final: {final}")
            return AgentResult(final=final, trace=trace, messages=messages)

        for call in response.tool_calls:
            trace.append(f"tool_call: {call.name} {call.arguments}")
            result = tools.run(call)
            trace.append(f"observation: {result.content}")
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=result.tool_call_id,
                    content=result.content,
                    is_error=result.is_error,
                )
            )

    final = f"reached max_steps={max_steps}"
    trace.append(f"final: {final}")
    return AgentResult(final=final, trace=trace, messages=messages)
