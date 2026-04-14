"""Dataset example datatypes used throughout the pipeline.

The repository normalizes heterogeneous open datasets into one intermediate form.
That keeps the runner simple while still supporting:
- free-form language modeling prompts,
- multiple-choice evaluation,
- reasoning datasets with long-form answers,
- instruction/chat prompts,
- and handwritten diagnostic suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    """A single normalized example.

    Attributes
    ----------
    text:
        The prompt text fed to the frozen target model and the interpreter.
    family:
        A high-level prompt family used for grouped analysis and plots.
    source:
        Dataset registry key or handcrafted-suite identifier.
    task_type:
        One of ``prompt_suite``, ``lm_continuation``, ``multiple_choice``,
        ``reasoning``, ``instruction`` or ``chat``.
    choices:
        Optional answer choices for restricted-choice behavioral evaluation.
    answer:
        Optional gold answer string for multiple-choice or supervised datasets.
    metadata:
        Arbitrary metadata propagated through the pipeline for artifact exports.
    """

    text: str
    family: str
    source: str
    task_type: str
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
