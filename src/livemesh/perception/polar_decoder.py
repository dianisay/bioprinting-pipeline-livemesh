"""Polar Decoder — Main contribution of the thesis.

Predicts wound boundary as centroid (x_c, y_c) + N radii at fixed angular
intervals. Guarantees ordered waypoints and closed-loop contour by construction.
"""

import torch
import torch.nn as nn
import math


class PolarDecoder(nn.Module):
    """Decodes encoder features into a polar boundary representation.

    Output: centroid (x_c, y_c) + N radii, convertible to ordered Cartesian points.
    """

    def __init__(self, d_model: int = 256, num_radii: int = 64):
        super().__init__()
        self.num_radii = num_radii

        # Fixed angles (not learned) — evenly spaced, excluding 2*pi
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / num_radii), num_radii)
        self.register_buffer("angles", angles)

        # Centroid prediction head
        self.centroid_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),  # Normalize to [0, 1] (image coordinates)
        )

        # Radii prediction head
        self.radii_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_radii),
            nn.Softplus(),  # Radii must be positive
        )

    def forward(self, encoder_output: torch.Tensor):
        """
        Args:
            encoder_output: (B, seq_len, d_model) from CNNTransformerEncoder

        Returns:
            dict with keys:
                - centroid: (B, 2) predicted centroid in [0, 1]
                - radii: (B, N) predicted radii
                - points: (B, N, 2) Cartesian boundary points
        """
        # Global average pooling over sequence
        pooled = encoder_output.mean(dim=1)  # (B, d_model)

        # Predict centroid and radii
        centroid = self.centroid_head(pooled)  # (B, 2)
        radii = self.radii_head(pooled)  # (B, N)

        # Convert polar to Cartesian
        points = self._polar_to_cartesian(centroid, radii)

        return {
            "centroid": centroid,
            "radii": radii,
            "points": points,
        }

    def _polar_to_cartesian(
        self, centroid: torch.Tensor, radii: torch.Tensor
    ) -> torch.Tensor:
        """Convert polar representation to ordered Cartesian points.

        Args:
            centroid: (B, 2) — (x_c, y_c)
            radii: (B, N) — radii at fixed angles

        Returns:
            (B, N, 2) — ordered boundary points
        """
        cos_a = torch.cos(self.angles)  # (N,)
        sin_a = torch.sin(self.angles)  # (N,)

        x = centroid[:, 0:1] + radii * cos_a.unsqueeze(0)  # (B, N)
        y = centroid[:, 1:2] + radii * sin_a.unsqueeze(0)  # (B, N)

        return torch.stack([x, y], dim=-1)  # (B, N, 2)


class PolarBoundaryLoss(nn.Module):
    """Combined loss for polar decoder training.

    L = λ_c * L_centroid + λ_r * L_radii + λ_p * L_points
    """

    def __init__(
        self,
        lambda_centroid: float = 1.0,
        lambda_radii: float = 1.0,
        lambda_points: float = 0.5,
    ):
        super().__init__()
        self.lambda_centroid = lambda_centroid
        self.lambda_radii = lambda_radii
        self.lambda_points = lambda_points
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred: dict, target: dict) -> dict:
        """
        Args:
            pred: dict with centroid, radii, points
            target: dict with centroid, radii, points (ground truth)

        Returns:
            dict with total loss and components
        """
        loss_centroid = self.mse(pred["centroid"], target["centroid"])
        loss_radii = self.smooth_l1(pred["radii"], target["radii"])
        loss_points = self.mse(pred["points"], target["points"])

        total = (
            self.lambda_centroid * loss_centroid
            + self.lambda_radii * loss_radii
            + self.lambda_points * loss_points
        )

        return {
            "total": total,
            "centroid": loss_centroid,
            "radii": loss_radii,
            "points": loss_points,
        }
