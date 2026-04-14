
"""Mechanism-prediction experiment."""

from __future__ import annotations

import time
from typing import Dict, List
import torch
import torch.nn.functional as F

from frontier_interp.analysis.metrics import mechanism_attention_metrics
from frontier_interp.experiments.controls import apply_control
from frontier_interp.experiments.complexity import attention_complexity_features
from frontier_interp.utils.time import format_seconds


def train_attention_mechanism_interpreter(interpreter, target_model, train_batches, *, layer_idx: int, head_idx: int, control_name: str, steps: int, logger, run_label: str, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(interpreter.parameters(), lr=lr, weight_decay=weight_decay)
    interpreter.train()

    baseline_loss = None
    complexity_cache = None
    start = time.time()
    batches = list(train_batches)

    for step in range(steps):
        batch = batches[step % len(batches)]
        token_batch = target_model.tokenize_batch(batch["texts"], batch["max_prompt_len"])

        with torch.no_grad():
            _, attentions = target_model.extract_logits_and_attentions(token_batch)
            target_attn = attentions[layer_idx][:, head_idx]
            target_attn = apply_control(control_name, target_attn)
            if complexity_cache is None:
                complexity_cache = attention_complexity_features(target_attn)

        pred_attn = interpreter.forward_mechanism(token_batch["input_ids"])
        loss = F.mse_loss(pred_attn, target_attn)

        if baseline_loss is None:
            bsz, seq_len, _ = target_attn.shape
            tri = torch.tril(torch.ones(seq_len, seq_len, device=target_attn.device))
            tri = tri / tri.sum(dim=-1, keepdim=True).clamp_min(1.0)
            baseline = tri.unsqueeze(0).expand_as(target_attn)
            baseline_loss = F.mse_loss(baseline, target_attn).item()
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
def eval_attention_mechanism_interpreter(interpreter, target_model, val_batches, *, layer_idx: int, head_idx: int, control_name: str, baseline_loss: float):
    interpreter.eval()
    rows = []

    for batch in val_batches:
        token_batch = target_model.tokenize_batch(batch["texts"], batch["max_prompt_len"])
        _, attentions = target_model.extract_logits_and_attentions(token_batch)
        target_attn = attentions[layer_idx][:, head_idx]
        target_attn = apply_control(control_name, target_attn)
        pred_attn = interpreter.forward_mechanism(token_batch["input_ids"])

        metrics = mechanism_attention_metrics(pred_attn, target_attn)
        metrics["normed_loss"] = metrics["mse_loss"] / baseline_loss
        metrics["family"] = batch["family"]
        rows.append(metrics)

    interpreter.train()
    return rows
