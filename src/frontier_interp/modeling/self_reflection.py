"""Self-reflection probes attached directly to frozen target hidden states.

These modules operationalize a stronger version of the paper's thesis: rather
than asking whether an *external* interpreter can compress the target model, we
ask whether lightweight heads attached to the model's own representations can
recover its behavior and mechanisms.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSelfReflectionProbe(nn.Module):
    """Common interface for self-reflection probes."""

    def forward_behavior_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_mechanism_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearSelfReflectionProbe(BaseSelfReflectionProbe):
    """Linear readout over target hidden states.

    This is the strongest test of direct self-accessibility under a tiny probe.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, probe_dim: int = 64):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, probe_dim)
        self.behavior_head = nn.Linear(probe_dim, vocab_size)
        self.query_proj = nn.Linear(probe_dim, probe_dim)
        self.key_proj = nn.Linear(probe_dim, probe_dim)

    def _base(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_proj(hidden)

    def forward_behavior_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.behavior_head(self._base(hidden))

    def forward_mechanism_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self._base(hidden)
        q = self.query_proj(h)
        k = self.key_proj(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = hidden.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        return F.softmax(scores, dim=-1)


class MLPSelfReflectionProbe(BaseSelfReflectionProbe):
    """A slightly stronger nonlinear probe on top of frozen target states."""

    def __init__(self, hidden_dim: int, vocab_size: int, probe_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, probe_dim),
            nn.GELU(),
        )
        self.behavior_head = nn.Linear(probe_dim, vocab_size)
        self.query_proj = nn.Linear(probe_dim, probe_dim)
        self.key_proj = nn.Linear(probe_dim, probe_dim)

    def _base(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.backbone(hidden)

    def forward_behavior_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.behavior_head(self._base(hidden))

    def forward_mechanism_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self._base(hidden)
        q = self.query_proj(h)
        k = self.key_proj(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = hidden.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        return F.softmax(scores, dim=-1)


SELF_REFLECTION_PROBE_REGISTRY = {
    'linear': LinearSelfReflectionProbe,
    'mlp': MLPSelfReflectionProbe,
}


def build_self_reflection_probe(arch: str, hidden_dim: int, vocab_size: int, probe_dim: int):
    if arch not in SELF_REFLECTION_PROBE_REGISTRY:
        raise KeyError(f'Unknown self-reflection probe architecture: {arch}')
    return SELF_REFLECTION_PROBE_REGISTRY[arch](hidden_dim=hidden_dim, vocab_size=vocab_size, probe_dim=probe_dim)
