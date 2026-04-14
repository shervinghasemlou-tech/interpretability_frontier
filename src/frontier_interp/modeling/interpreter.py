
"""Interpreter models used for both behavioral and mechanistic prediction.

The main scientific constraint is that behavior and mechanism tasks should use the
same *family* of interpreter architecture. This module therefore exposes a small
registry of interpreter classes with a common API.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: str) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class BaseInterpreter(nn.Module):
    """Shared interface for all interpreters."""

    def forward_behavior(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_mechanism(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerInterpreter(BaseInterpreter):
    """Default causal-transformer interpreter.

    This is the main architecture used in the paper's matched-comparison runs.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.logit_head = nn.Linear(d_model, vocab_size)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        h = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        mask = causal_mask(seq_len, input_ids.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.ln(h)

    def forward_behavior(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.backbone(input_ids))

    def forward_mechanism(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids)
        q = self.query_proj(h)
        k = self.key_proj(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = input_ids.shape[1]
        scores = scores.masked_fill(causal_mask(seq_len, input_ids.device).unsqueeze(0), float("-inf"))
        return F.softmax(scores, dim=-1)


class MLPInterpreter(BaseInterpreter):
    """A weaker architecture control.

    This interpreter intentionally lacks explicit sequence modeling beyond token
    embeddings and position embeddings. It serves as a counterargument test.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.logit_head = nn.Linear(d_model, vocab_size)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        return self.mlp(self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :])

    def forward_behavior(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.backbone(input_ids))

    def forward_mechanism(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids)
        q = self.query_proj(h)
        k = self.key_proj(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = input_ids.shape[1]
        scores = scores.masked_fill(causal_mask(seq_len, input_ids.device).unsqueeze(0), float("-inf"))
        return F.softmax(scores, dim=-1)


class PositionOnlyInterpreter(BaseInterpreter):
    """A deliberately weak control that uses only learned position embeddings."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.logit_head = nn.Linear(d_model, vocab_size)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_emb[:, : input_ids.shape[1], :].expand(input_ids.shape[0], -1, -1)

    def forward_behavior(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.backbone(input_ids))

    def forward_mechanism(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids)
        q = self.query_proj(h)
        k = self.key_proj(h)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        seq_len = input_ids.shape[1]
        scores = scores.masked_fill(causal_mask(seq_len, input_ids.device).unsqueeze(0), float("-inf"))
        return F.softmax(scores, dim=-1)


INTERPRETER_REGISTRY = {
    "transformer": TransformerInterpreter,
    "mlp": MLPInterpreter,
    "position_only": PositionOnlyInterpreter,
}


def build_interpreter(arch: str, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.0) -> BaseInterpreter:
    if arch not in INTERPRETER_REGISTRY:
        raise KeyError(f"Unknown interpreter architecture: {arch}")
    return INTERPRETER_REGISTRY[arch](vocab_size, d_model, n_heads, n_layers, max_len, dropout)
