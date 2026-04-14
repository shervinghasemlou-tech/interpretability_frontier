"""Target model loading and frozen extraction helpers.

This wrapper hides tokenizer quirks and provides helper methods used by both the
behavioral and mechanistic experiments. Keeping these utilities centralized is
important because they define what "matched comparison" means.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from frontier_interp.registries.models import resolve_model_spec


class FrozenTargetModel:
    """Thin wrapper around a frozen open-weight causal LM."""

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
    def extract_logits_and_attentions(self, token_batch: Dict[str, torch.Tensor], *, output_attentions: bool = True):
        outputs = self.model(
            **token_batch,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()
        attentions = [a.float() for a in outputs.attentions] if output_attentions and outputs.attentions is not None else []
        return logits, attentions

    def _score_continuation_from_logits(self, logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
        """Return average log-prob of the continuation portion of a prompt+choice string.

        ``input_ids`` encodes the full prompt followed by a choice string. ``prompt_len`` is
        the token count of the prompt-only encoding. We score choice tokens conditioned on
        the prompt and prior choice tokens.
        """
        if prompt_len >= input_ids.shape[0]:
            return float("-inf")
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        total = 0.0
        count = 0
        for pos in range(prompt_len, input_ids.shape[0]):
            prev_pos = pos - 1
            total += float(log_probs[prev_pos, input_ids[pos]].item())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def score_choices_with_target(self, prompt: str, choices: List[str], max_prompt_len: int) -> List[float]:
        """Score answer choices using the frozen target model itself."""
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        scores = []
        for choice in choices:
            full = prompt + " " + choice
            toks = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=max_prompt_len)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            outputs = self.model(**toks, use_cache=False, return_dict=True)
            scores.append(self._score_continuation_from_logits(outputs.logits[0].float(), toks["input_ids"][0], prompt_len))
        return scores

    @torch.no_grad()
    def score_choices_with_interpreter(self, interpreter, prompt: str, choices: List[str], max_prompt_len: int) -> List[float]:
        """Score answer choices using an interpreter's behavioral head.

        This lets us evaluate behavior in a restricted-choice setting without changing the
        interpreter architecture. The interpreter still predicts logits token-by-token.
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        scores = []
        for choice in choices:
            full = prompt + " " + choice
            toks = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=max_prompt_len)
            input_ids = toks["input_ids"].to(self.device)
            logits = interpreter.forward_behavior(input_ids)[0].float()
            scores.append(self._score_continuation_from_logits(logits, input_ids[0], prompt_len))
        return scores
