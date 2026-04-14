
"""Dataset registry.

Each dataset entry describes how it should be interpreted by the experiment code.
The loaders themselves live in :mod:`frontier_interp.data.dataset_factory`.
"""

from __future__ import annotations

from typing import Dict, Any


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "handcrafted_diagnostics": {
        "loader": "handcrafted",
        "kind": "prompt_suite",
        "default_family": "diagnostic",
    },
    "wikitext103_validation": {
        "loader": "hf_wikitext",
        "dataset_name": "Salesforce/wikitext",
        "subset": "wikitext-103-v1",
        "default_split": "validation",
        "kind": "lm_continuation",
    },
    "hellaswag_validation": {
        "loader": "hf_hellaswag",
        "dataset_name": "Rowan/hellaswag",
        "default_split": "validation",
        "kind": "multiple_choice",
    },
    "piqa_validation": {
        "loader": "hf_piqa",
        "dataset_name": "ybisk/piqa",
        "default_split": "validation",
        "kind": "multiple_choice",
    },
    "arc_easy_validation": {
        "loader": "hf_arc",
        "dataset_name": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "default_split": "validation",
        "kind": "multiple_choice",
    },
    "arc_challenge_validation": {
        "loader": "hf_arc",
        "dataset_name": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "default_split": "validation",
        "kind": "multiple_choice",
    },
    "gsm8k_test": {
        "loader": "hf_gsm8k",
        "dataset_name": "openai/gsm8k",
        "subset": "main",
        "default_split": "test",
        "kind": "reasoning",
    },
    "alpaca_train": {
        "loader": "hf_alpaca",
        "dataset_name": "tatsu-lab/alpaca",
        "default_split": "train",
        "kind": "instruction",
    },
    "ultrachat_train_sft": {
        "loader": "hf_ultrachat",
        "dataset_name": "HuggingFaceH4/ultrachat_200k",
        "subset": "default",
        "default_split": "train_sft",
        "kind": "chat",
    },
}


def resolve_dataset_spec(registry_key: str) -> Dict[str, Any]:
    if registry_key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset registry key: {registry_key}")
    return DATASET_REGISTRY[registry_key]
