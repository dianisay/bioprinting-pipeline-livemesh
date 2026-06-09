"""DETR-style Parallel Cartesian Decoder — Ablation baseline (from scratch).

Predicts N boundary points simultaneously using learned queries and
cross-attention. Requires Hungarian matching for loss assignment.
All attention mechanisms implemented manually (no nn.TransformerDecoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional

from livemesh.perception.encoder import MultiHeadAttention, FeedForwardNetwork


class CrossAttention(nn.Module):
    """Cross-attention: queries attend to encoder memory (key/value from encoder).

    Identical to self-attention except Q comes from decoder, K and V from encoder.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:  (B, N_q, d_model) — decoder queries
            memory: (B, N_m, d_model) — encoder output

        Returns:
            (B, N_q, d_model)
        """
        B, N_q, _ = query.shape
        N_m = memory.shape[1]

        Q = self.W_q(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(memory).view(B, N_m, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(memory).view(B, N_m, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        return self.W_o(out)


class DecoderLayer(nn.Module):
    """Single decoder layer: Self-Attention → Cross-Attention → FFN (all Pre-LN)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) decoder input
            memory: (B, M, d_model) encoder output
            self_attn_mask: optional causal mask for autoregressive decoding
        """
        # Self-attention
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, self_attn_mask))

        # Cross-attention to encoder memory
        normed = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(normed, memory))

        # Feed-forward
        normed = self.norm3(x)
        x = x + self.dropout3(self.ffn(normed))

        return x


class DETRDecoder(nn.Module):
    """Parallel set prediction decoder (DETR-style), from scratch.

    N learned query embeddings attend to encoder features via stacked
    decoder layers with self-attention + cross-attention. All N points
    are predicted simultaneously (no autoregressive dependency).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_queries: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Stacked decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Point prediction head
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        Args:
            encoder_output: (B, seq_len, d_model) from encoder

        Returns:
            dict with 'points': (B, N, 2)
        """
        B = encoder_output.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        x = queries
        for layer in self.layers:
            x = layer(x, encoder_output)
        x = self.norm(x)

        points = self.point_head(x)  # (B, N, 2)
        return {"points": points}


class HungarianLoss(nn.Module):
    """Hungarian matching loss for unordered set prediction.

    Finds optimal bipartite assignment between predicted and GT points
    using the Hungarian algorithm, then computes L2 loss on matched pairs.
    """

    def forward(self, pred: dict, target: dict) -> dict:
        pred_points = pred["points"]  # (B, N, 2)
        target_points = target["points"]  # (B, N, 2)

        B, N, _ = pred_points.shape
        total_loss = torch.tensor(0.0, device=pred_points.device)

        for b in range(B):
            cost = torch.cdist(pred_points[b], target_points[b], p=2)
            cost_np = cost.detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_np)

            matched_pred = pred_points[b, row_idx]
            matched_target = target_points[b, col_idx]
            total_loss += F.mse_loss(matched_pred, matched_target)

        return {"total": total_loss / B}
