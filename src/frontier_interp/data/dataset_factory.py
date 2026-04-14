"""Dataset loading and normalization.

This module keeps the repository dataset-agnostic. The experiment code consumes a
list of normalized :class:`Example` objects, while dataset-specific quirks stay
isolated here.
"""

from __future__ import annotations

from typing import List

from datasets import load_dataset

from frontier_interp.data.base import Example
from frontier_interp.data.prompts import HANDCRAFTED_PROMPTS
from frontier_interp.registries.datasets import resolve_dataset_spec



def _trim(text: str) -> str:
    return " ".join(str(text).split())



def load_examples_from_spec(dataset_spec) -> List[Example]:
    registry = resolve_dataset_spec(dataset_spec.registry_key)
    loader = registry["loader"]

    if loader == "handcrafted":
        return _load_handcrafted(dataset_spec)
    if loader == "hf_wikitext":
        return _load_wikitext(dataset_spec, registry)
    if loader == "hf_hellaswag":
        return _load_hellaswag(dataset_spec, registry)
    if loader == "hf_piqa":
        return _load_piqa(dataset_spec, registry)
    if loader == "hf_arc":
        return _load_arc(dataset_spec, registry)
    if loader == "hf_gsm8k":
        return _load_gsm8k(dataset_spec, registry)
    if loader == "hf_alpaca":
        return _load_alpaca(dataset_spec, registry)
    if loader == "hf_ultrachat":
        return _load_ultrachat(dataset_spec, registry)

    raise ValueError(f"Unsupported loader: {loader}")



def _load_handcrafted(dataset_spec) -> List[Example]:
    examples: List[Example] = []
    for family, prompts in HANDCRAFTED_PROMPTS.items():
        for p in prompts[: dataset_spec.num_samples]:
            examples.append(Example(text=p, family=family, source="handcrafted_diagnostics", task_type="prompt_suite"))
    return examples



def _load_wikitext(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        txt = _trim(row.get("text", ""))
        if len(txt) > 20:
            out.append(Example(text=txt, family="wikitext", source=dataset_spec.registry_key, task_type="lm_continuation"))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_hellaswag(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        ctx = _trim(row["ctx"])
        choices = [_trim(x) for x in row["endings"]]
        answer = choices[int(row["label"])] if str(row["label"]).isdigit() else None
        out.append(Example(text=ctx, family="hellaswag", source=dataset_spec.registry_key, task_type="multiple_choice", choices=choices, answer=answer))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_piqa(dataset_spec, registry) -> List[Example]:
    split = dataset_spec.split or registry["default_split"]

    # Some PIQA mirrors on HF still require legacy dataset scripts, which newer
    # `datasets` versions reject. Fall back to script-free parquet mirrors.
    candidate_names = [registry["dataset_name"], "nthngdy/piqa", "gimmaru/piqa"]
    ds = None
    first_error = None
    for dataset_name in candidate_names:
        try:
            ds = load_dataset(dataset_name, split=split)
            break
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if "Dataset scripts are no longer supported" in str(exc):
                continue
            raise

    if ds is None:
        raise RuntimeError(
            "Failed to load PIQA from all known mirrors "
            f"({candidate_names}). First error: {first_error}"
        )

    out = []
    for row in ds:
        prompt = _trim(row["goal"])
        choices = [_trim(row["sol1"]), _trim(row["sol2"])]
        answer = choices[int(row["label"])]
        out.append(Example(text=prompt, family="piqa", source=dataset_spec.registry_key, task_type="multiple_choice", choices=choices, answer=answer))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_arc(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        q = _trim(row["question"])
        choices = [_trim(x) for x in row["choices"]["text"]]
        answer_key = row.get("answerKey")
        answer = None
        if answer_key is not None:
            labels = list(row["choices"]["label"])
            if answer_key in labels:
                answer = choices[labels.index(answer_key)]
        out.append(Example(text=q, family=registry["subset"].lower(), source=dataset_spec.registry_key, task_type="multiple_choice", choices=choices, answer=answer))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_gsm8k(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        q = _trim(row["question"])
        a = _trim(row["answer"])
        out.append(Example(text=q, family="gsm8k", source=dataset_spec.registry_key, task_type="reasoning", answer=a))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_alpaca(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        prompt = _trim(row["instruction"])
        if row.get("input"):
            prompt = prompt + "\nInput: " + _trim(row["input"])
        out.append(Example(text=prompt, family="alpaca", source=dataset_spec.registry_key, task_type="instruction", answer=_trim(row.get("output", ""))))
        if len(out) >= dataset_spec.num_samples:
            break
    return out



def _load_ultrachat(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    out = []
    for row in ds:
        msgs = row.get("messages") or []
        if not msgs:
            continue
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        assistant_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if not user_msgs:
            continue
        prompt = _trim(user_msgs[0])
        answer = _trim(assistant_msgs[0]) if assistant_msgs else None
        out.append(Example(text=prompt, family="ultrachat", source=dataset_spec.registry_key, task_type="chat", answer=answer))
        if len(out) >= dataset_spec.num_samples:
            break
    return out
