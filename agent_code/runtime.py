"""agent_code/runtime.py — 跨主线程 / worker 线程共享的运行态。"""

from __future__ import annotations
import queue
import threading
from dataclasses import dataclass, field


@dataclass
class RuntimeState:
    # 主线程（输入/键位/状态栏）写这些；worker 线程（Agent Loop）读这些。
    permission_mode: str = "default"  # default | acceptEdits | plan，shift+tab 改它
    model: str = "deepseek-v4-pro"  # /model 改它，下一轮 turn 生效
    provider: str = "anthropic"
    abort_event: threading.Event = field(default_factory=threading.Event)
    input_queue: queue.Queue[str] = field(default_factory=queue.Queue)
