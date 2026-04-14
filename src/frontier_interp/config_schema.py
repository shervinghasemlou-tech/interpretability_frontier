
"""Typed configuration loading for all experiments.

The repository intentionally centralizes nearly all experiment behavior in YAML.
The CLI takes a single ``--config`` argument, loads it into these dataclasses,
and passes the structured config through the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ModelSpec:
    registry_key: str
    enabled: bool = True
    revision: Optional[str] = None
    torch_dtype: Optional[str] = None
    attn_implementation: str = "eager"


@dataclass
class DatasetSpec:
    registry_key: str
    enabled: bool = True
    split: Optional[str] = None
    num_samples: int = 64
    family: Optional[str] = None
    subset: Optional[str] = None
    add_choices: bool = True


@dataclass
class SweepSpec:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    interpreter_sizes: List[int] = field(default_factory=lambda: [64, 128])
    limit_layers: int = 4
    limit_heads: int = 8
    mechanism_types: List[str] = field(default_factory=lambda: ["attention_probs"])
    interpreter_arches: List[str] = field(default_factory=lambda: ["transformer"])
    controls: List[str] = field(default_factory=list)


@dataclass
class TrainingSpec:
    behavior_steps: int = 800
    mechanism_steps: int = 300
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_prompt_len: int = 96
    interpreter_layers: int = 2
    interpreter_heads: int = 4
    dropout: float = 0.0


@dataclass
class OutputSpec:
    root_dir: str = "outputs"
    run_name: str = "debug_run"
    save_checkpoints: bool = True
    save_plots: bool = True
    save_per_example_artifacts: bool = False


@dataclass
class StatsSpec:
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    fdr_method: str = "bh"
    primary_metrics: List[str] = field(
        default_factory=lambda: ["mean_gap", "p90_gap", "frac_hard_heads"]
    )


@dataclass
class RuntimeSpec:
    device: str = "auto"
    allow_fp16: bool = True
    trust_remote_code: bool = True
    compile_interpreters: bool = False


@dataclass
class ExperimentConfig:
    description: str
    models: List[ModelSpec]
    datasets: List[DatasetSpec]
    sweep: SweepSpec = field(default_factory=SweepSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    outputs: OutputSpec = field(default_factory=OutputSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)


def _load_dataclass(cls, data: Dict[str, Any]):
    """Recursively materialize a dataclass from a YAML-derived dictionary."""
    kwargs = {}
    for name, field_info in cls.__dataclass_fields__.items():
        if name not in data:
            continue
        value = data[name]
        field_type = field_info.type

        if name == "models":
            kwargs[name] = [ModelSpec(**x) for x in value]
        elif name == "datasets":
            kwargs[name] = [DatasetSpec(**x) for x in value]
        elif name == "sweep":
            kwargs[name] = SweepSpec(**value)
        elif name == "training":
            kwargs[name] = TrainingSpec(**value)
        elif name == "outputs":
            kwargs[name] = OutputSpec(**value)
        elif name == "stats":
            kwargs[name] = StatsSpec(**value)
        elif name == "runtime":
            kwargs[name] = RuntimeSpec(**value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str) -> ExperimentConfig:
    """Load a YAML config file into :class:`ExperimentConfig`."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _load_dataclass(ExperimentConfig, raw)
