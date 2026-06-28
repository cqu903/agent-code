"""tools 包：基础设施 + 所有具体工具模块的统一入口。

re-export base.py 里的基础设施名字，让旧的 `from agent_code.tools import X` 零改动；
顶部 import 各工具模块以触发 @register_tool，把工具并入 _REGISTERED_TOOLS。
"""

from __future__ import annotations

from .base import (  # re-export，消费者零改动
    Tool,
    ToolCall,
    ToolContext,
    ToolFunc,
    ToolRegistry,
    ToolResult,
    default_tools,
    register_tool,
)

# 顶部 import 各工具模块：触发各自的 @register_tool，把工具并入 _REGISTERED_TOOLS。
# 这是 Python 惯用的"顶部 import"，不再是 default_tools() 里的延迟导入特例。
from . import bash_tool, cron, fs, git, memory_tool, misc, web  # noqa: F401
