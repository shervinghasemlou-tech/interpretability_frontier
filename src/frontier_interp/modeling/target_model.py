
"""Target model loading and frozen extraction helpers."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from frontier_interp.registries.models import resolve_model_spec


class FrozenTargetModel:
    """Wrapper around a frozen open-weight causal LM.

    The wrapper hides tokenizer quirks, dtype decisions, and attention extraction.
    """

    def __init__(self, model_key: str, runtime, model_spec_override=None):
        registry = resolve_model_spec(model_key)
        self.registry = registry
        self.model_key = model_key

        hf_name = registry["hf_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=runtime.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if runtime.allow_fp16 and torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=dtype,
            trust_remote_code=runtime.trust_remote_code,
            attn_implementation=getattr(model_spec_override, "attn_implementation", "eager") if model_spec_override else "eager",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if runtime.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = runtime.device
        self.model.to(self.device)

        self.num_layers = self.model.config.num_hidden_layers
        if hasattr(self.model.config, "num_attention_heads"):
            self.num_heads = self.model.config.num_attention_heads
        elif hasattr(self.model.config, "num_key_value_heads"):
            self.num_heads = self.model.config.num_key_value_heads
        else:
            raise ValueError("Could not infer number of heads from model config.")

    def tokenize_batch(self, texts: List[str], max_prompt_len: int) -> Dict[str, torch.Tensor]:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in toks.items()}

    @torch.no_grad()
    def extract_logits_and_attentions(self, token_batch: Dict[str, torch.Tensor]):
        outputs = self.model(
            **token_batch,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()
        attentions = [a.float() for a in outputs.attentions]
        return logits, attentions
