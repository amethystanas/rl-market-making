"""Structured event log (JSON Lines) written by the trading loop and read by the dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventLog:
    """Appends one JSON object per line. The file is reset when a new session starts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, event_type: str, **fields) -> None:
        event = {"type": event_type, "time": time.time(), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=float) + "\n")


class NullEventLog:
    def write(self, event_type: str, **fields) -> None:
        pass
