
"""Dataset example datatypes used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Example:
    """A single normalized example.

    The code normalizes very different datasets into a single intermediate format.
    Some fields are optional because not every dataset is multiple-choice or chat.
    """

    text: str
    family: str
    source: str
    task_type: str
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    metadata: dict = field(default_factory=dict)
