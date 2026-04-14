"""Target model loading and frozen extraction helpers.

This wrapper hides tokenizer quirks and provides helper methods used by both the
behavioral and mechanistic experiments. Keeping these utilities centralized is
important because they define what "matched comparison" means.

The frozen target model is often the largest memory consumer in the repository,
so this module also supports memory-saving load modes for small GPUs:
- bitsandbytes 8-bit quantization,
- bitsandbytes 4-bit quantization,
- device_map="auto" loading.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency path
    BitsAndBytesConfig = None

from frontier_interp.registries.models import resolve_model_spec


def _resolve_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    name = str(name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[name]


class FrozenTargetModel:
    """Thin wrapper around a frozen open-weight causal LM.

    Notes
    -----
    Quantized loading is only applied to the *target* model. Interpreters remain
    ordinary PyTorch modules because they are trained in the experiment loop.
    """

    def __init__(self, model_key: str, runtime, model_spec_override=None):
        registry = resolve_model_spec(model_key)
        self.registry = registry
        self.model_key = model_key
        self.runtime = runtime

        hf_name = registry["hf_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            trust_remote_code=runtime.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decide the target model weight dtype when not quantizing.
        default_dtype = torch.float16 if runtime.allow_fp16 and torch.cuda.is_available() else torch.float32
        requested_dtype = _resolve_torch_dtype(getattr(model_spec_override, "torch_dtype", None))
        model_dtype = requested_dtype or default_dtype

        load_kwargs = {
            "trust_remote_code": runtime.trust_remote_code,
            "attn_implementation": getattr(model_spec_override, "attn_implementation", "eager") if model_spec_override else "eager",
            "low_cpu_mem_usage": getattr(runtime, "low_cpu_mem_usage", True),
        }

        load_in_8bit = bool(getattr(runtime, "load_in_8bit", False))
        load_in_4bit = bool(getattr(runtime, "load_in_4bit", False))
        if load_in_8bit and load_in_4bit:
            raise ValueError("Choose only one of runtime.load_in_8bit or runtime.load_in_4bit.")

        quantization_config = None
        if load_in_8bit or load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes quantization was requested, but BitsAndBytesConfig is unavailable. "
                    "Install bitsandbytes and a recent Transformers build."
                )
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                compute_dtype = _resolve_torch_dtype(getattr(runtime, "bnb_4bit_compute_dtype", "float16")) or torch.float16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=getattr(runtime, "bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=bool(getattr(runtime, "bnb_4bit_use_double_quant", True)),
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            load_kwargs["quantization_config"] = quantization_config
            # For quantized models, let Accelerate/Transformers place weights automatically.
            load_kwargs["device_map"] = getattr(runtime, "device_map", None) or "auto"
            load_kwargs["dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = model_dtype

        self.model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Figure out where tokenized inputs should live. For device_map='auto', most small-GPU
        # runs still need inputs on CUDA when available.
        if getattr(runtime, "device_map", None) == "auto" or load_in_8bit or load_in_4bit:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif runtime.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
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

    @torch.no_grad()
    def extract_logits_attentions_hidden(self, token_batch: Dict[str, torch.Tensor]):
        """Return logits, attentions, and hidden states for frozen-target self-reflection runs."""
        outputs = self.model(
            **token_batch,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits.float()
        attentions = [a.float() for a in outputs.attentions]
        hidden_states = [h.float() for h in outputs.hidden_states]
        return logits, attentions, hidden_states

    @torch.no_grad()
    def generate_texts(self, prompts: List[str], max_prompt_len: int, max_new_tokens: int = 8, temperature: float = 0.0) -> List[str]:
        """Generate short continuations from the frozen target for prompted self-report experiments."""
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
            padding=True,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        do_sample = temperature is not None and temperature > 0.0
        out = self.model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = toks["input_ids"].shape[1]
        decoded = []
        for row in out:
            decoded.append(self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip())
        return decoded

    def decode_token_id(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()

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
