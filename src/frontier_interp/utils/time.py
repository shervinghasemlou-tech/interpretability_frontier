
"""Timing and ETA utilities."""

from __future__ import annotations

import time


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressTracker:
    """A minimal global progress tracker with ETA support."""

    def __init__(self, total_units: int):
        self.total_units = total_units
        self.start_time = time.time()
        self.completed_units = 0

    def update(self, n: int = 1):
        self.completed_units += n

    def status(self) -> dict:
        elapsed = time.time() - self.start_time
        rate = self.completed_units / max(elapsed, 1e-8)
        remaining = self.total_units - self.completed_units
        eta = remaining / max(rate, 1e-8)
        return {
            "completed": self.completed_units,
            "total": self.total_units,
            "elapsed": elapsed,
            "eta": eta,
        }
