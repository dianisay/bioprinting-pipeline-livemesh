"""Autoregressive Cartesian Decoder — Ablation baseline (from scratch).

Predicts boundary points sequentially, each conditioned on all previously
predicted points via causal self-attention + cross-attention to encoder.
All attention mechanisms implemented manually (no nn.TransformerDecoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from livemesh.perception.encoder import MultiHeadAttention, FeedForwardNetwork
from livemesh.perception.detr_decoder import CrossAttention, DecoderLayer


class AutoregressiveDecoder(nn.Module):
    """Sequential boundary prediction via causal Transformer decoder (from scratch).

    Each point is predicted conditioned on encoder features and all previously
    generated points. Uses causal masking to prevent attending to future positions.
    Teacher forcing during training; autoregressive sampling at inference.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_points: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model

        # Input embedding: map 2D coordinates to d_model
        self.point_embed = nn.Linear(2, d_model)

        # Learned start-of-sequence token
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learned positional embeddings for sequence positions
        self.pos_embed = nn.Embedding(num_points, d_model)

        # Stacked decoder layers (same architecture as DETR but with causal mask)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output head: predict 2D coordinate
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (lower-triangular) attention mask.

        Returns mask where 1 = attend, 0 = block.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(
        self, encoder_output: torch.Tensor, target_points: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            encoder_output: (B, seq_len, d_model) from encoder
            target_points: (B, N, 2) GT points for teacher forcing (training only)

        Returns:
            dict with 'points': (B, N, 2)
        """
        if target_points is not None:
            return self._forward_teacher_forcing(encoder_output, target_points)
        return self._forward_autoregressive(encoder_output)

    def _forward_teacher_forcing(
        self, encoder_output: torch.Tensor, target_points: torch.Tensor
    ) -> dict:
        """Training mode: all positions predicted in parallel with causal mask."""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Embed target points and prepend start token (shift right)
        embedded = self.point_embed(target_points)  # (B, N, d_model)
        start = self.start_token.expand(B, -1, -1)  # (B, 1, d_model)
        tgt = torch.cat([start, embedded[:, :-1]], dim=1)  # (B, N, d_model)

        # Add positional encoding
        positions = torch.arange(self.num_points, device=device)
        tgt = tgt + self.pos_embed(positions).unsqueeze(0)

        # Causal mask: prevent attending to future tokens
        causal_mask = self._causal_mask(self.num_points, device)

        # Pass through decoder layers
        x = tgt
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask=causal_mask)
        x = self.norm(x)

        points = self.output_head(x)  # (B, N, 2)
        return {"points": points}

    @torch.no_grad()
    def _forward_autoregressive(self, encoder_output: torch.Tensor) -> dict:
        """Inference mode: generate points one by one."""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with just the start token
        generated_embeds = self.start_token.expand(B, -1, -1)  # (B, 1, d_model)
        generated_points = []

        for i in range(self.num_points):
            seq_len = generated_embeds.shape[1]

            # Add positional encoding to current sequence
            positions = torch.arange(seq_len, device=device)
            tgt = generated_embeds + self.pos_embed(positions).unsqueeze(0)

            # Causal mask for current length
            causal_mask = self._causal_mask(seq_len, device)

            # Forward through all layers
            x = tgt
            for layer in self.layers:
                x = layer(x, encoder_output, self_attn_mask=causal_mask)
            x = self.norm(x)

            # Predict next point from last position
            next_point = self.output_head(x[:, -1:])  # (B, 1, 2)
            generated_points.append(next_point)

            # Embed and append for next iteration
            next_embed = self.point_embed(next_point)  # (B, 1, d_model)
            generated_embeds = torch.cat([generated_embeds, next_embed], dim=1)

        points = torch.cat(generated_points, dim=1)  # (B, N, 2)
        return {"points": points}


class AutoregressiveLoss(nn.Module):
    """MSE loss for autoregressive predictions (points are already ordered)."""

    def forward(self, pred: dict, target: dict) -> dict:
        loss = F.mse_loss(pred["points"], target["points"])
        return {"total": loss}
