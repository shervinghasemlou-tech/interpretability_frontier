
"""Curated model registry for sub-1B open-weight language models.

The registry is intentionally conservative and focuses on model families that are
useful for a paper about mechanistic compressibility:
- base / instruct or base / chat pairs,
- compact modern architectures,
- at least one family designed for interpretability research.
"""

from __future__ import annotations

from typing import Dict, Any


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen25_05b_base": {
        "hf_name": "Qwen/Qwen2.5-0.5B",
        "family": "qwen2.5",
        "variant": "base",
        "param_bucket": "0.5B",
    },
    "qwen25_05b_instruct": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "qwen2.5",
        "variant": "instruct",
        "param_bucket": "0.5B",
    },
    "smollm2_360m_base": {
        "hf_name": "HuggingFaceTB/SmolLM2-360M",
        "family": "smollm2",
        "variant": "base",
        "param_bucket": "360M",
    },
    "smollm2_360m_instruct": {
        "hf_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "family": "smollm2",
        "variant": "instruct",
        "param_bucket": "360M",
    },
    "openelm_450m_base": {
        "hf_name": "apple/OpenELM-450M",
        "family": "openelm",
        "variant": "base",
        "param_bucket": "450M",
    },
    "openelm_450m_instruct": {
        "hf_name": "apple/OpenELM-450M-Instruct",
        "family": "openelm",
        "variant": "instruct",
        "param_bucket": "450M",
    },
    "danube3_500m_base": {
        "hf_name": "h2oai/h2o-danube3-500m-base",
        "family": "danube3",
        "variant": "base",
        "param_bucket": "500M",
    },
    "danube3_500m_chat": {
        "hf_name": "h2oai/h2o-danube3-500m-chat",
        "family": "danube3",
        "variant": "chat",
        "param_bucket": "500M",
    },
    "pythia_410m": {
        "hf_name": "EleutherAI/pythia-410m",
        "family": "pythia",
        "variant": "base",
        "param_bucket": "410M",
    },
    "pythia_410m_deduped": {
        "hf_name": "EleutherAI/pythia-410m-deduped",
        "family": "pythia",
        "variant": "deduped",
        "param_bucket": "410M",
    },
}


def resolve_model_spec(registry_key: str) -> Dict[str, Any]:
    if registry_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model registry key: {registry_key}")
    return MODEL_REGISTRY[registry_key]
