
"""Counterfactual and control transforms.

These controls are first-class because reviewers will ask for them. They can be
applied to targets before training or evaluation.
"""

from __future__ import annotations

from typing import Optional
import torch


def apply_control(name: str, target_tensor: torch.Tensor, *, rng: Optional[torch.Generator] = None) -> torch.Tensor:
    if name == "none":
        return target_tensor
    if name == "label_shuffle":
        idx = torch.randperm(target_tensor.shape[0], generator=rng, device=target_tensor.device)
        return target_tensor[idx]
    if name == "head_shuffle":
        # The experiment runner handles this by selecting a different head. This function is a no-op placeholder.
        return target_tensor
    if name == "random_target":
        out = torch.rand_like(target_tensor)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if name == "diagonal_baseline":
        bsz, seq_len, _ = target_tensor.shape
        diag = torch.eye(seq_len, device=target_tensor.device).unsqueeze(0).expand(bsz, -1, -1)
        return diag
    if name == "causal_uniform":
        bsz, seq_len, _ = target_tensor.shape
        tri = torch.tril(torch.ones(seq_len, seq_len, device=target_tensor.device))
        tri = tri / tri.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return tri.unsqueeze(0).expand(bsz, -1, -1)
    raise ValueError(f"Unknown control: {name}")
