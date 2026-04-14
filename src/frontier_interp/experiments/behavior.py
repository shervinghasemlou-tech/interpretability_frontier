
"""Behavioral distillation experiment."""

from __future__ import annotations

import time
from typing import Dict, List
import torch
import torch.nn.functional as F

from frontier_interp.analysis.metrics import behavior_distillation_metrics
from frontier_interp.utils.time import format_seconds


def train_behavior_interpreter(interpreter, target_model, train_batches, steps: int, logger, run_label: str, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(interpreter.parameters(), lr=lr, weight_decay=weight_decay)
    interpreter.train()

    baseline_loss = None
    start = time.time()
    batches = list(train_batches)

    for step in range(steps):
        batch = batches[step % len(batches)]
        token_batch = target_model.tokenize_batch(batch["texts"], batch["max_prompt_len"])

        with torch.no_grad():
            target_logits, _ = target_model.extract_logits_and_attentions(token_batch)

        pred_logits = interpreter.forward_behavior(token_batch["input_ids"])
        loss = F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction="batchmean",
        )

        if baseline_loss is None:
            uniform_logits = torch.zeros_like(pred_logits)
            baseline_loss = F.kl_div(
                F.log_softmax(uniform_logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="batchmean",
            ).item()
            baseline_loss = max(baseline_loss, 1e-8)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - start
            done = step + 1
            eta = (steps - done) * (elapsed / done)
            logger.log(
                f"{run_label}[behavior] step={step:4d}/{steps} "
                f"| loss={loss.item():.4f} "
                f"| normed={loss.item()/baseline_loss:.4f} "
                f"| elapsed={format_seconds(elapsed)} "
                f"| eta={format_seconds(eta)}"
            )

    return baseline_loss


@torch.no_grad()
def eval_behavior_interpreter(interpreter, target_model, val_batches, baseline_loss: float):
    interpreter.eval()
    rows = []

    for batch in val_batches:
        token_batch = target_model.tokenize_batch(batch["texts"], batch["max_prompt_len"])
        target_logits, _ = target_model.extract_logits_and_attentions(token_batch)
        pred_logits = interpreter.forward_behavior(token_batch["input_ids"])
        metrics = behavior_distillation_metrics(pred_logits, target_logits)
        metrics["normed_loss"] = metrics["kl_loss"] / baseline_loss
        metrics["family"] = batch["family"]
        rows.append(metrics)

    interpreter.train()
    return rows
