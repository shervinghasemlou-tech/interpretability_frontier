"""Dataset loading and normalization.

This module keeps the repository dataset-agnostic. The experiment code consumes a
list of normalized :class:`Example` objects, while dataset-specific quirks stay
isolated here.

Key behavior
------------
- ``num_samples`` limits the number of normalized examples when set to an int.
- ``use_full_split=True`` or ``num_samples=None`` means: iterate the entire
  requested split and keep all normalized examples.
"""

from __future__ import annotations

from typing import List, Optional

from datasets import load_dataset

from frontier_interp.data.base import Example
from frontier_interp.data.prompts import HANDCRAFTED_PROMPTS
from frontier_interp.registries.datasets import resolve_dataset_spec


def _trim(text: str) -> str:
    return " ".join(str(text).split())


def _sample_cap(dataset_spec) -> Optional[int]:
    if getattr(dataset_spec, "use_full_split", False):
        return None
    return getattr(dataset_spec, "num_samples", None)


def _should_stop(count: int, cap: Optional[int]) -> bool:
    return cap is not None and count >= cap


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
    cap = _sample_cap(dataset_spec)
    examples: List[Example] = []
    for family, prompts in HANDCRAFTED_PROMPTS.items():
        for p in prompts:
            examples.append(Example(text=p, family=family, source="handcrafted_diagnostics", task_type="prompt_suite"))
            if _should_stop(len(examples), cap):
                return examples
    return examples


def _load_wikitext(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
    out = []
    for row in ds:
        txt = _trim(row.get("text", ""))
        if len(txt) > 20:
            out.append(Example(text=txt, family="wikitext", source=dataset_spec.registry_key, task_type="lm_continuation"))
        if _should_stop(len(out), cap):
            break
    return out


def _load_hellaswag(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
    out = []
    for row in ds:
        ctx = _trim(row["ctx"])
        choices = [_trim(x) for x in row["endings"]]
        answer = choices[int(row["label"])] if str(row["label"]).isdigit() else None
        out.append(Example(text=ctx, family="hellaswag", source=dataset_spec.registry_key, task_type="multiple_choice", choices=choices, answer=answer))
        if _should_stop(len(out), cap):
            break
    return out


def _load_piqa(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
    out = []
    for row in ds:
        prompt = _trim(row["goal"])
        choices = [_trim(row["sol1"]), _trim(row["sol2"])]
        answer = choices[int(row["label"])]
        out.append(Example(text=prompt, family="piqa", source=dataset_spec.registry_key, task_type="multiple_choice", choices=choices, answer=answer))
        if _should_stop(len(out), cap):
            break
    return out


def _load_arc(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
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
        if _should_stop(len(out), cap):
            break
    return out


def _load_gsm8k(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], registry["subset"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
    out = []
    for row in ds:
        q = _trim(row["question"])
        a = _trim(row["answer"])
        out.append(Example(text=q, family="gsm8k", source=dataset_spec.registry_key, task_type="reasoning", answer=a))
        if _should_stop(len(out), cap):
            break
    return out


def _load_alpaca(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
    out = []
    for row in ds:
        prompt = _trim(row["instruction"])
        if row.get("input"):
            prompt = prompt + "\nInput: " + _trim(row["input"])
        out.append(Example(text=prompt, family="alpaca", source=dataset_spec.registry_key, task_type="instruction", answer=_trim(row.get("output", ""))))
        if _should_stop(len(out), cap):
            break
    return out


def _load_ultrachat(dataset_spec, registry) -> List[Example]:
    ds = load_dataset(registry["dataset_name"], split=dataset_spec.split or registry["default_split"])
    cap = _sample_cap(dataset_spec)
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
        if _should_stop(len(out), cap):
            break
    return out
