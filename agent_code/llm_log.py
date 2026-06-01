from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LLMLogger:
    log_dir: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    _turn: int = field(default=0, init=False, repr=False)
    _path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y-%m-%d_%H%M%S")
        self._path = self.log_dir / f"{stamp}_session_{self.session_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def next_turn(self) -> int:
        self._turn += 1
        return self._turn

    def log_request(self, turn: int, payload: dict[str, Any]) -> None:
        self._append("request", turn, payload)

    def log_response(
        self, turn: int, payload: dict[str, Any], *, duration_ms: int
    ) -> None:
        self._append("response", turn, payload, duration_ms=duration_ms)

    def _append(
        self,
        direction: str,
        turn: int,
        payload: dict[str, Any],
        *,
        duration_ms: int | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "turn": turn,
            "direction": direction,
            "payload": payload,
        }
        if duration_ms is not None:
            record["duration_ms"] = duration_ms
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_llm_logger(log_dir: Path | None) -> LLMLogger | None:
    if log_dir is None:
        return None
    return LLMLogger(log_dir=log_dir.resolve())
