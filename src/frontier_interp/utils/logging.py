
"""Simple logging utilities.

The goal is to keep runtime logging easy to read in terminals and also to persist a
plain-text log file per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from frontier_interp.utils.time import format_seconds


class RunLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_progress(self, prefix: str, progress_status: dict):
        self.log(
            f"{prefix}[progress] {progress_status['completed']}/{progress_status['total']} "
            f"| elapsed {format_seconds(progress_status['elapsed'])} "
            f"| eta {format_seconds(progress_status['eta'])}"
        )
