"""Mechanism-prediction experiment.

The primary mechanism family is attention-based because it is cheap to extract and
straightforward to visualize. We expose several mechanism *views* over the same
predicted and target attention matrices:
- full probabilities,
- top-1 attended token,
- row-wise entropy.
"""

from __future__ import annotations

import time
from typing import Dict, List
import torch
import torch.nn.functional as F

from frontier_interp.analysis.metrics import mechanism_attention_metrics
from frontier_interp.experiments.controls import apply_control
from frontier_interp.experiments.complexity import attention_complexity_features
from frontier_interp.utils.time import format_seconds



def _attention_training_loss(pred_attn: torch.Tensor, target_attn: torch.Tensor, mechanism_type: str) -> torch.Tensor:
    if mechanism_type == "attention_probs":
        return F.mse_loss(pred_attn, target_attn)
    if mechanism_type == "attention_top1":
        target_idx = target_attn.argmax(dim=-1)
        return F.nll_loss(torch.log(pred_attn.clamp_min(1e-8)).reshape(-1, pred_attn.shape[-1]), target_idx.reshape(-1))
    if mechanism_type == "attention_entropy":
        eps = 1e-8
        target_entropy = -(target_attn.clamp_min(eps) * target_attn.clamp_min(eps).log()).sum(dim=-1)
        pred_entropy = -(pred_attn.clamp_min(eps) * pred_attn.clamp_min(eps).log()).sum(dim=-1)
        return F.mse_loss(pred_entropy, target_entropy)
    raise ValueError(f"Unsupported mechanism type: {mechanism_type}")



def train_attention_mechanism_interpreter(interpreter, target_model, train_batches, *, layer_idx: int, head_idx: int, control_name: str, mechanism_type: str, steps: int, logger, run_label: str, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(interpreter.parameters(), lr=lr, weight_decay=weight_decay)
    interpreter.train()

    baseline_loss = None
    complexity_cache = None
    start = time.time()
    batches = list(train_batches)

    for step in range(steps):
        batch = batches[step % len(batches)]
        token_batch = target_model.tokenize_batch([ex.text for ex in batch["examples"]], batch["max_prompt_len"])

        with torch.no_grad():
            _, attentions = target_model.extract_logits_and_attentions(token_batch)
            target_attn = attentions[layer_idx][:, head_idx]
            target_attn = apply_control(control_name, target_attn)
            if complexity_cache is None:
                complexity_cache = attention_complexity_features(target_attn)

        pred_attn = interpreter.forward_mechanism(token_batch["input_ids"])
        loss = _attention_training_loss(pred_attn, target_attn, mechanism_type)

        if baseline_loss is None:
            bsz, seq_len, _ = target_attn.shape
            tri = torch.tril(torch.ones(seq_len, seq_len, device=target_attn.device))
            tri = tri / tri.sum(dim=-1, keepdim=True).clamp_min(1.0)
            baseline = tri.unsqueeze(0).expand_as(target_attn)
            baseline_loss = float(_attention_training_loss(baseline, target_attn, mechanism_type).item())
            baseline_loss = max(baseline_loss, 1e-8)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - start
            done = step + 1
            eta = (steps - done) * (elapsed / done)
            logger.log(
                f"{run_label}[mechanism] step={step:4d}/{steps} "
                f"| loss={loss.item():.6f} "
                f"| normed={loss.item()/baseline_loss:.4f} "
                f"| elapsed={format_seconds(elapsed)} "
                f"| eta={format_seconds(eta)}"
            )

    return baseline_loss, complexity_cache or {}


@torch.no_grad()
def eval_attention_mechanism_interpreter(interpreter, target_model, val_batches, *, layer_idx: int, head_idx: int, control_name: str, mechanism_type: str, baseline_loss: float):
    interpreter.eval()
    rows = []
    example_artifacts = []

    for batch in val_batches:
        token_batch = target_model.tokenize_batch([ex.text for ex in batch["examples"]], batch["max_prompt_len"])
        _, attentions = target_model.extract_logits_and_attentions(token_batch)
        target_attn = attentions[layer_idx][:, head_idx]
        target_attn = apply_control(control_name, target_attn)
        pred_attn = interpreter.forward_mechanism(token_batch["input_ids"])

        metrics = mechanism_attention_metrics(pred_attn, target_attn, mechanism_type=mechanism_type)
        metrics["normed_loss"] = metrics["mse_loss"] / baseline_loss
        metrics["family"] = batch["family"]
        metrics["source"] = batch["source"]
        rows.append(metrics)

        # export one example heatmap candidate per batch for later paper figures
        example_artifacts.append({
            "family": batch["family"],
            "source": batch["source"],
            "text": batch["examples"][0].text,
            "target_attn": target_attn[0].detach().cpu().numpy(),
            "pred_attn": pred_attn[0].detach().cpu().numpy(),
        })

    interpreter.train()
    return rows, example_artifacts
