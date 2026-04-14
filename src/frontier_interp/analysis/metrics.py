"""Metric helpers for behavioral and mechanistic tasks."""

from __future__ import annotations

from typing import Dict, List, Optional
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


def restricted_choice_metrics(target_scores: List[float], pred_scores: List[float], gold_index: Optional[int]) -> Dict[str, float]:
    """Metrics for multiple-choice style behavioral evaluation.

    The target model defines the reference ranking and the interpreter is scored against it.
    If a gold label is available, we also compute exact-choice accuracy.
    """
    target_scores = np.asarray(target_scores, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    target_argmax = int(np.argmax(target_scores))
    pred_argmax = int(np.argmax(pred_scores))

    metrics = {
        "choice_match_to_target": float(target_argmax == pred_argmax),
        "choice_rank_corr": float(np.corrcoef(target_scores, pred_scores)[0, 1]) if len(target_scores) > 1 else 1.0,
    }
    if gold_index is not None and gold_index >= 0:
        metrics["choice_acc_vs_gold"] = float(pred_argmax == gold_index)
        metrics["target_acc_vs_gold"] = float(target_argmax == gold_index)
    else:
        metrics["choice_acc_vs_gold"] = float("nan")
        metrics["target_acc_vs_gold"] = float("nan")
    return metrics


@torch.no_grad()
def mechanism_attention_metrics(pred_attn: torch.Tensor, target_attn: torch.Tensor, mechanism_type: str = "attention_probs") -> Dict[str, float]:
    """Metrics for attention-distribution prediction.

    Supported mechanism types:
    - ``attention_probs``: full probability matrix MSE + top-1 target token match.
    - ``attention_top1``: only top-1 attended token agreement, still measured from the
      predicted and target probability matrices.
    - ``attention_entropy``: row-wise attention entropy agreement.
    """
    out: Dict[str, float] = {}
    if mechanism_type == "attention_probs":
        out["mse_loss"] = float(F.mse_loss(pred_attn, target_attn).item())
        target_argmax = target_attn.argmax(dim=-1)
        pred_argmax = pred_attn.argmax(dim=-1)
        out["top1_match"] = float((target_argmax == pred_argmax).float().mean().item())
        return out

    if mechanism_type == "attention_top1":
        target_argmax = target_attn.argmax(dim=-1)
        pred_argmax = pred_attn.argmax(dim=-1)
        out["mse_loss"] = float((pred_argmax != target_argmax).float().mean().item())
        out["top1_match"] = float((target_argmax == pred_argmax).float().mean().item())
        return out

    if mechanism_type == "attention_entropy":
        eps = 1e-8
        target_entropy = -(target_attn.clamp_min(eps) * target_attn.clamp_min(eps).log()).sum(dim=-1)
        pred_entropy = -(pred_attn.clamp_min(eps) * pred_attn.clamp_min(eps).log()).sum(dim=-1)
        out["mse_loss"] = float(F.mse_loss(pred_entropy, target_entropy).item())
        out["top1_match"] = float((pred_attn.argmax(dim=-1) == target_attn.argmax(dim=-1)).float().mean().item())
        return out

    raise ValueError(f"Unsupported mechanism type: {mechanism_type}")
