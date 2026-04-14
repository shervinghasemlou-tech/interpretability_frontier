
"""Metric helpers for behavioral and mechanistic tasks."""

from __future__ import annotations

from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def behavior_distillation_metrics(pred_logits: torch.Tensor, target_logits: torch.Tensor) -> Dict[str, float]:
    """Compute KL, top-1 agreement, and top-5 overlap for next-token distributions."""
    loss = F.kl_div(
        F.log_softmax(pred_logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction="batchmean",
    ).item()

    target_top1 = target_logits.argmax(dim=-1)
    pred_top1 = pred_logits.argmax(dim=-1)
    top1_match = (target_top1 == pred_top1).float().mean().item()

    target_top5 = torch.topk(target_logits, k=min(5, target_logits.shape[-1]), dim=-1).indices
    pred_top5 = torch.topk(pred_logits, k=min(5, pred_logits.shape[-1]), dim=-1).indices
    overlaps = []
    for b in range(target_top5.shape[0]):
        for t in range(target_top5.shape[1]):
            a = set(target_top5[b, t].tolist())
            c = set(pred_top5[b, t].tolist())
            overlaps.append(len(a & c) / max(1, len(a)))

    return {
        "kl_loss": float(loss),
        "top1_match": float(top1_match),
        "top5_overlap": float(np.mean(overlaps)),
    }


@torch.no_grad()
def mechanism_attention_metrics(pred_attn: torch.Tensor, target_attn: torch.Tensor) -> Dict[str, float]:
    """Metrics for attention-distribution prediction."""
    loss = F.mse_loss(pred_attn, target_attn).item()
    target_argmax = target_attn.argmax(dim=-1)
    pred_argmax = pred_attn.argmax(dim=-1)
    top1_match = (target_argmax == pred_argmax).float().mean().item()
    return {
        "mse_loss": float(loss),
        "top1_match": float(top1_match),
    }
