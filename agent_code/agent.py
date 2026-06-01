from __future__ import annotations

from dataclasses import dataclass

from .model import Provider
from .tools import ToolRegistry


@dataclass
class AgentResult:
    final: str
    trace: list[str]


def run_agent(prompt: str, provider: Provider, tools: ToolRegistry) -> AgentResult:
    # messages 是每一轮都要交回provider的上下文
    messages = [{"role": "user", "content": prompt}]
    trace: list[str] = []

    response = provider.complete(messages, tools)

    for call in response.tool_calls or []:
        trace.append(f"tool_call: {call.name} {call.arguments}")

        result = tools.run(call)
        trace.append(f"observation: {result.content}")
        # 工具结果会成为下一轮模型调用的observation
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            }
        )
        response = provider.complete(messages, tools)

    final = response.text or ""
    trace.append(f"final: {final}")
    return AgentResult(final=final, trace=trace)
