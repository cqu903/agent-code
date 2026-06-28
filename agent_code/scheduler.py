from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any


class CronJob:
    """一个定时任务。id 是 12 位 hex，slash 是到点要重放的命令/prompt。"""

    def __init__(
        self,
        job_id: str,
        slash: str,
        every_seconds: int,
        label: str = "",
        last_run_at: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.id = job_id
        self.slash = slash  # 到点要重放的 slash 或 prompt
        self.every_seconds = every_seconds  # 执行间隔（秒）
        self.label = label  # 人类可读标签
        self.last_run_at = last_run_at
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()


# cron 持久化文件，放在 .agent/ 下和 sessions/memory 同级
_CRON_FILE = ".agent/cron.json"


def _cron_path(cwd: Path) -> Path:
    """返回 .agent/cron.json 路径，自动创建 .agent/ 目录。"""
    agent_dir = cwd / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir / "cron.json"


def _load_jobs(cwd: Path) -> list[CronJob]:
    """从 .agent/cron.json 加载持久化 job 列表。文件不存在或损坏返回 []。"""
    fpath = _cron_path(cwd)
    if not fpath.exists():
        return []
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    jobs: list[CronJob] = []
    for item in data.get("jobs", []):
        try:
            jobs.append(
                CronJob(
                    job_id=item["id"],
                    slash=item["slash"],
                    every_seconds=item["every_seconds"],
                    label=item.get("label", ""),
                    last_run_at=item.get("last_run_at"),
                    created_at=item.get("created_at"),
                )
            )
        except (KeyError, TypeError):
            continue  # 跳过损坏的job，不让坏数据阻塞调度器
    return jobs


def _save_jobs(cwd: Path, jobs: list[CronJob]) -> None:
    """把当前job持久化"""
    fpath = _cron_path(cwd)
    data = {
        "jobs": [
            {
                "id": j.id,
                "slash": j.slash,
                "every_seconds": j.every_seconds,
                "label": j.label,
                "last_run_at": j.last_run_at,
                "created_at": j.created_at,
            }
            for j in jobs
        ]
    }
    fpath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class CronScheduler:
    """REPL 内的 cron 调度器。维护 job 列表 + 后台 daemon thread + pending queue。

    harness 边界：
    - 后台线程只负责到点把 prompt 放进 pending queue，绝不直接调用 run_agent
    - REPL 主循环在每次 run_once 返回后 drain queue，把待处理 prompt 作为新一轮用户输入
    - 调度器只在 REPL 模式激活；一次性模式不创建后台线程
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self._jobs: list[CronJob] = _load_jobs(cwd)
        self._pending: Queue[str] = Queue()  # 到点排队的prompt
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # Job管理
    def add_job(self, slash: str, every_seconds: int, label: str = "") -> CronJob:
        """添加一个 cron job，id自动生成"""
        jid = uuid.uuid4().hex[:12]
        job = CronJob(
            job_id=jid,
            slash=slash,
            every_seconds=every_seconds,
            label=label,
        )
        with self._lock:
            self._jobs.append(job)
            _save_jobs(self.cwd, self._jobs)
        return job

    def list_jobs(self) -> list[CronJob]:
        with self._lock:
            return list(self._jobs)

    def cancel_job(self, jid: str) -> bool:
        """按 id 取消一个job，返回True表示找到并删除"""
        with self._lock:
            for i, j in enumerate(self._jobs):
                if j.id == jid:
                    self._jobs.pop(i)
                    _save_jobs(self.cwd, self._jobs)
                    return True
        return False

    # pending queue
    def drain_pending(self) -> list[str]:
        """去除当前pending queue 里面所有等待重放的prompt，调用一次就排空"""
        items: list[str] = []
        while not self._pending.empty():
            try:
                items.append(self._pending.get_nowait())
            except Exception:
                break
        return items

    # 后台调度循环
    def _loop(self) -> None:
        """后台线程主循环。每 1 秒 tick 一次：
        - 检查哪些 job 的上次执行时间距今超过 every_seconds
        - 到点的 job 把 slash 放进 pending queue
        - 更新 last_run_at，写回持久化文件
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(1.0)
            if self._stop_event.is_set():
                break
            now_ts = datetime.now(timezone.utc).timestamp()
            dirty = False
            with self._lock:
                for job in self._jobs:
                    baseline = job.last_run_at or job.created_at
                    last_ts = 0.0
                    if baseline:
                        try:
                            last_dt = datetime.fromisoformat(baseline)
                            if last_dt.tzinfo is None:
                                last_dt = last_dt.replace(tzinfo=timezone.utc)
                            last_ts = last_dt.timestamp()
                        except ValueError:
                            pass
                    if now_ts - last_ts >= job.every_seconds:
                        self._pending.put(job.slash)
                        job.last_run_at = datetime.now(timezone.utc).isoformat()
                        dirty = True
                if dirty:
                    _save_jobs(self.cwd, self._jobs)

    def start(self) -> None:
        """启动后台调度线程（daemon线程）"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止调度器。不 join——daemon thread 会在主线程退出时回收。"""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
