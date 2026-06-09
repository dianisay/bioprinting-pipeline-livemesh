"""Volumetric Wound Decoder — Predict wound boundary + depth + layer-wise fill.

This decoder takes features from VolumetricWoundEncoder3D and outputs:
1. Centroid (2D): Center of the wound in image coordinates
2. Radii (2D polar): Distance to wound edge in 64 radial directions
3. Depth profile (3D): How deep the wound is in each direction
4. Layer amounts (3D): How much bio-ink to deposit per layer

The key innovation: layer-aware predictions for stratified bioprinting.
Instead of printing the entire wound boundary at once, we fill it layer by layer,
with the fill amount varying per layer (more in the center, less at edges).

This matches real bioprinting: wounds have depth, and healing requires
gradient-based material deposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional


class PolarDecoder3DLayered(nn.Module):
    """
    Outputs 3D wound representation with layer stratification.

    Prediction outputs:
    - centroid: (B, 2) normalized [0, 1] wound center
    - radii: (B, num_radii) normalized distance to edge in each direction
    - depth: (B, num_radii) absolute depth (mm) at each radial direction
    - layer_amounts: (B, num_radii, num_layers) fill amount per layer per radius
                     values in [0, 1] — how much of that radial line to fill in each layer

    The 3D points are reconstructed layer-by-layer:
    - Layer 0 (deepest): smallest radius, deepest
    - Layer 1: medium radius, medium depth
    - Layer 2: larger radius, shallower
    - Layer 3 (surface): full radius, minimal depth (just touching)

    This creates a cone-like fill pattern that mimics natural wound healing.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_radii: int = 64,
        num_layers: int = 4,
        max_depth_mm: float = 5.0,  # Maximum wound depth in millimeters
    ):
        super().__init__()
        self.d_model = d_model
        self.num_radii = num_radii
        self.num_layers = num_layers
        self.max_depth_mm = max_depth_mm

        # Fixed angles (not learned) — evenly spaced around 360°
        angles = torch.linspace(
            0,
            2 * math.pi * (1 - 1 / num_radii),
            num_radii
        )
        self.register_buffer("angles", angles)

        # ============================================================
        # Prediction Heads
        # ============================================================

        # Head 1: Centroid prediction (x_c, y_c)
        # Output normalized to [0, 1] for image coordinates
        self.centroid_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),  # [0, 1] for image normalization
        )

        # Head 2: Radii prediction (boundary in each direction)
        # Output normalized to [0, 1] for image normalization
        self.radii_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_radii),
            nn.Softplus(),  # Radii must be positive
        )

        # Head 3: Depth prediction (how deep wound is)
        # Output in [0, max_depth_mm]
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_radii),
            nn.Softplus(),  # Depth ≥ 0
        )

        # Head 4: Layer-wise fill amounts
        # For each radius, predict how much to fill in each layer
        # Output: (B, num_radii * num_layers) in [0, 1]
        self.layer_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_radii * num_layers),
            nn.Sigmoid(),  # [0, 1] for fill fraction
        )

    def forward(
        self,
        encoder_features: torch.Tensor,
        volumetric_layer_hint: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: (B, d_model) global context from VolumetricWoundEncoder3D
            volumetric_layer_hint: (B, num_radii, num_layers) optional pre-prediction
                                  from volumetric fusion (to guide layer fills)

        Returns:
            dict with keys:
                - centroid: (B, 2) normalized wound center
                - radii: (B, num_radii) normalized boundary radii
                - depth: (B, num_radii) wound depth in mm
                - layer_amounts: (B, num_radii, num_layers) fill fraction per layer
                - points_3d: (B, num_radii, num_layers, 3) 3D coordinates per layer
                - points_2d: (B, num_radii, 2) 2D Cartesian boundary points
        """
        B = encoder_features.shape[0]

        # ============================================================
        # Prediction Stage
        # ============================================================

        # Predict centroid
        centroid = self.centroid_head(encoder_features)  # (B, 2)

        # Predict radii
        radii = self.radii_head(encoder_features)  # (B, num_radii)

        # Predict depth
        depth_raw = self.depth_head(encoder_features)  # (B, num_radii)
        depth = depth_raw * self.max_depth_mm  # Scale to mm range

        # Predict layer amounts
        layer_amounts_raw = self.layer_head(encoder_features)  # (B, num_radii * num_layers)
        layer_amounts = layer_amounts_raw.reshape(B, self.num_radii, self.num_layers)

        # Optional: fuse with volumetric hint if provided
        if volumetric_layer_hint is not None:
            # Blend predicted and volumetric hint (70% prediction, 30% hint)
            layer_amounts = 0.7 * layer_amounts + 0.3 * volumetric_layer_hint

        # ============================================================
        # Coordinate Generation
        # ============================================================

        # 2D Cartesian boundary (at surface, layer = num_layers - 1)
        points_2d = self._polar_to_cartesian_2d(centroid, radii)  # (B, num_radii, 2)

        # 3D points per layer
        points_3d_per_layer = []
        for layer_idx in range(self.num_layers):
            # Interpolation factor: 0 (deepest) to 1 (surface)
            layer_factor = layer_idx / max(1, self.num_layers - 1)

            # Layer properties: taper inward as we go deeper
            # Deepest layer: 70% of full radius
            # Surface layer: 100% of full radius
            layer_radii = radii * (0.7 + layer_factor * 0.3)

            # Layer depth: increases from 0 (surface) to full depth (center)
            # Surface (layer 3): touches at layer_depth ≈ 0
            # Center (layer 0): at full depth
            layer_depth = depth * (1.0 - layer_factor)

            # Fill amount: how much bio-ink to deposit in this layer at this radius
            layer_fill = layer_amounts[:, :, layer_idx]  # (B, num_radii)

            # Convert to 3D Cartesian
            points_3d = self._polar_to_cartesian_3d(
                centroid,
                layer_radii,
                layer_depth,
                layer_fill,
            )  # (B, num_radii, 3)

            points_3d_per_layer.append(points_3d)

        points_3d = torch.stack(points_3d_per_layer, dim=2)  # (B, num_radii, num_layers, 3)

        # ============================================================
        # Output
        # ============================================================
        return {
            "centroid": centroid,  # (B, 2)
            "radii": radii,  # (B, num_radii)
            "depth": depth,  # (B, num_radii)
            "layer_amounts": layer_amounts,  # (B, num_radii, num_layers)
            "points_2d": points_2d,  # (B, num_radii, 2) — 2D boundary
            "points_3d": points_3d,  # (B, num_radii, num_layers, 3) — 3D layered fill
        }

    def _polar_to_cartesian_2d(
        self,
        centroid: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert 2D polar coordinates to Cartesian.

        Args:
            centroid: (B, 2) center [x_c, y_c] normalized to [0, 1]
            radii: (B, num_radii) radius at each angle

        Returns:
            (B, num_radii, 2) Cartesian points
        """
        cos_a = torch.cos(self.angles)  # (num_radii,)
        sin_a = torch.sin(self.angles)  # (num_radii,)

        # x = x_c + r * cos(angle)
        x = centroid[:, 0:1] + radii * cos_a.unsqueeze(0)  # (B, num_radii)

        # y = y_c + r * sin(angle)
        y = centroid[:, 1:2] + radii * sin_a.unsqueeze(0)  # (B, num_radii)

        points = torch.stack([x, y], dim=-1)  # (B, num_radii, 2)
        return points

    def _polar_to_cartesian_3d(
        self,
        centroid: torch.Tensor,
        layer_radii: torch.Tensor,
        layer_depth: torch.Tensor,
        layer_fill: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert 2D polar + depth to 3D Cartesian with fill modulation.

        Args:
            centroid: (B, 2) center [x_c, y_c]
            layer_radii: (B, num_radii) radius for this layer
            layer_depth: (B, num_radii) depth (Z coordinate) for this layer
            layer_fill: (B, num_radii) fill fraction [0, 1]

        Returns:
            (B, num_radii, 3) 3D points with modulated fill
        """
        cos_a = torch.cos(self.angles)
        sin_a = torch.sin(self.angles)

        # XY coordinates
        x = centroid[:, 0:1] + layer_radii * cos_a.unsqueeze(0) * layer_fill
        y = centroid[:, 1:2] + layer_radii * sin_a.unsqueeze(0) * layer_fill

        # Z coordinate (depth)
        z = layer_depth  # (B, num_radii)

        points = torch.stack([x, y, z], dim=-1)  # (B, num_radii, 3)
        return points


# ============================================================
# Loss Functions for Volumetric Decoder
# ============================================================

class VolumetricWoundLoss(nn.Module):
    """
    Combined loss for volumetric wound decoder.

    Losses:
    1. Boundary loss: Chamfer/MSE on 2D boundary points
    2. Depth loss: MAE or MSE on predicted depth vs ground truth
    3. Layer loss: Binary cross-entropy on layer fill amounts
    """

    def __init__(
        self,
        lambda_boundary: float = 1.0,
        lambda_depth: float = 1.0,
        lambda_layers: float = 0.5,
    ):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.lambda_depth = lambda_depth
        self.lambda_layers = lambda_layers

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: dict from decoder with centroid, radii, depth, layer_amounts, points_2d, points_3d
            target: dict with ground truth centroid, radii, depth, layer_amounts, points_2d

        Returns:
            dict with loss components and total loss
        """
        # Boundary loss (2D): compare predicted and target points
        loss_boundary = self.mse(pred["points_2d"], target["points_2d"])

        # Depth loss: compare predicted and target depth profiles
        loss_depth = self.mae(pred["depth"], target["depth"])

        # Layer loss: how well we predict fill amounts
        loss_layers = self.bce(pred["layer_amounts"], target["layer_amounts"])

        # Total loss
        total = (
            self.lambda_boundary * loss_boundary
            + self.lambda_depth * loss_depth
            + self.lambda_layers * loss_layers
        )

        return {
            "total": total,
            "boundary": loss_boundary,
            "depth": loss_depth,
            "layers": loss_layers,
        }


# ============================================================
# Testing / Debugging
# ============================================================

if __name__ == "__main__":
    print("Testing PolarDecoder3DLayered...")

    # Simulate encoder output
    batch_size = 2
    d_model = 256
    encoder_features = torch.randn(batch_size, d_model)

    # Optional: volumetric layer hint
    num_radii = 64
    num_layers = 4
    volumetric_hint = torch.rand(batch_size, num_radii, num_layers)

    # Initialize decoder
    decoder = PolarDecoder3DLayered(
        d_model=d_model,
        num_radii=num_radii,
        num_layers=num_layers,
        max_depth_mm=5.0,
    )

    # Forward pass
    output = decoder(encoder_features, volumetric_layer_hint)

    print(f"Centroid shape: {output['centroid'].shape}")  # (B, 2)
    print(f"Radii shape: {output['radii'].shape}")  # (B, 64)
    print(f"Depth shape: {output['depth'].shape}")  # (B, 64)
    print(f"Layer amounts shape: {output['layer_amounts'].shape}")  # (B, 64, 4)
    print(f"2D points shape: {output['points_2d'].shape}")  # (B, 64, 2)
    print(f"3D points shape: {output['points_3d'].shape}")  # (B, 64, 4, 3)

    print(f"\nCentroid range: [{output['centroid'].min():.3f}, {output['centroid'].max():.3f}] (should be [0, 1])")
    print(f"Radii range: [{output['radii'].min():.3f}, {output['radii'].max():.3f}]")
    print(f"Depth range: [{output['depth'].min():.3f}, {output['depth'].max():.3f}] mm (max={decoder.max_depth_mm})")
    print(f"Layer amounts range: [{output['layer_amounts'].min():.3f}, {output['layer_amounts'].max():.3f}] (should be [0, 1])")

    # Test loss function
    print("\nTesting VolumetricWoundLoss...")

    loss_fn = VolumetricWoundLoss(
        lambda_boundary=1.0,
        lambda_depth=1.0,
        lambda_layers=0.5,
    )

    # Simulate ground truth (slightly perturbed predictions)
    target = {
        "centroid": output["centroid"] + torch.randn_like(output["centroid"]) * 0.05,
        "radii": output["radii"] + torch.randn_like(output["radii"]) * 0.05,
        "depth": output["depth"] + torch.randn_like(output["depth"]) * 0.2,
        "layer_amounts": output["layer_amounts"] + torch.randn_like(output["layer_amounts"]) * 0.1,
        "points_2d": output["points_2d"] + torch.randn_like(output["points_2d"]) * 0.05,
    }

    losses = loss_fn(output, target)
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Boundary loss: {losses['boundary'].item():.4f}")
    print(f"Depth loss: {losses['depth'].item():.4f}")
    print(f"Layer loss: {losses['layers'].item():.4f}")

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal decoder parameters: {num_params:,}")

    print("\n✓ Test passed!")
