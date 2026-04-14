
"""Mechanism complexity covariates.

These covariates let us move beyond raw gap measurements and analyze *why* some
heads are easy while others are difficult.
"""

from __future__ import annotations

from typing import Dict
import torch


@torch.no_grad()
def attention_complexity_features(attn: torch.Tensor) -> Dict[str, float]:
    """Compute simple descriptive covariates for an attention tensor.

    Parameters
    ----------
    attn:
        Tensor of shape ``[B, T, T]`` containing row-stochastic attention matrices.
    """
    eps = 1e-8
    bsz, seq_len, _ = attn.shape

    entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1).mean().item()

    diag = torch.eye(seq_len, device=attn.device).unsqueeze(0)
    diagonal_mass = (attn * diag).sum(dim=(-1, -2)).mean().item() / max(seq_len, 1)

    near_diag = torch.zeros_like(diag)
    for offset in (-1, 0, 1):
        near_diag = near_diag + torch.diag_embed(torch.ones(seq_len - abs(offset), device=attn.device), offset=offset)
    near_diag = near_diag.clamp_max(1.0)
    local_mass = (attn * near_diag).sum(dim=(-1, -2)).mean().item() / max(seq_len, 1)
    nonlocal_mass = max(0.0, 1.0 - local_mass)

    row_var = attn.var(dim=0).mean().item()

    return {
        "attn_entropy": float(entropy),
        "diagonal_mass": float(diagonal_mass),
        "nonlocal_mass": float(nonlocal_mass),
        "prompt_variance": float(row_var),
    }
