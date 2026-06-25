"""Depth fusion: combine predicted depth (from volumetric encoder) with measured depth (from sensor).

Implements confidence-weighted fusion with three operating modes:
1. Both available: Kalman-style weighted blend based on confidence
2. Measurement invalid (specular): fall back to prediction
3. Large disagreement: trust measurement, flag for replanning

This module is the decision point between the vision stack's learned depth
estimate and the physical sensor's direct measurement.
"""

import numpy as np
from typing import Dict, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class DepthFusionConfig:
    """Configuration for depth fusion behavior."""

    def __init__(
        self,
        prediction_trust: float = 0.4,
        measurement_trust: float = 0.8,
        disagreement_threshold_mm: float = 2.0,
        replan_threshold_mm: float = 5.0,
        min_valid_fraction: float = 0.3,
    ):
        """
        Args:
            prediction_trust: base confidence weight for learned prediction [0,1]
            measurement_trust: base confidence weight for sensor measurement [0,1]
            disagreement_threshold_mm: if |pred - meas| > this, log a warning
            replan_threshold_mm: if |pred - meas| > this, trigger replanning
            min_valid_fraction: minimum fraction of valid measurement pixels
                               to consider the measurement reliable
        """
        self.prediction_trust = prediction_trust
        self.measurement_trust = measurement_trust
        self.disagreement_threshold_mm = disagreement_threshold_mm
        self.replan_threshold_mm = replan_threshold_mm
        self.min_valid_fraction = min_valid_fraction


def fuse_depth(
    predicted_depth_mm: np.ndarray,
    measured_depth_mm: np.ndarray,
    measurement_confidence: np.ndarray,
    config: Optional[DepthFusionConfig] = None,
) -> Dict[str, np.ndarray]:
    """Fuse predicted and measured depth using confidence-weighted blending.

    The fusion formula at each pixel is:
        w_m = measurement_confidence * config.measurement_trust
        w_p = (1 - measurement_confidence) * config.prediction_trust + (1 - w_m)
        fused = (w_p * predicted + w_m * measured) / (w_p + w_m)

    When measurement_confidence = 0 (invalid pixel): fused = predicted
    When measurement_confidence = 1 (perfect): fused ~= measured

    Args:
        predicted_depth_mm: (N,) or (H, W) predicted depth from decoder
        measured_depth_mm: same shape, measured depth from sensor (0 = invalid)
        measurement_confidence: same shape, confidence in [0, 1]
        config: fusion parameters

    Returns:
        dict with:
            - fused_depth_mm: blended depth estimate
            - uncertainty_mm: estimated uncertainty at each point
            - needs_replan: bool, True if disagreement exceeds threshold
            - disagreement_mm: per-point absolute difference where both valid
            - valid_fraction: fraction of measurement pixels that are valid
    """
    if config is None:
        config = DepthFusionConfig()

    predicted = predicted_depth_mm.astype(np.float64)
    measured = measured_depth_mm.astype(np.float64)
    confidence = measurement_confidence.astype(np.float64)

    # Measurement weight: high where confidence is high
    w_m = confidence * config.measurement_trust

    # Prediction weight: fills in where measurement is weak
    w_p = np.full_like(w_m, config.prediction_trust)

    # Normalize weights
    w_total = w_p + w_m
    w_total = np.maximum(w_total, 1e-8)

    # Weighted blend
    fused = (w_p * predicted + w_m * measured) / w_total

    # Where measurement is completely invalid (confidence=0), use prediction
    invalid_measurement = confidence < 0.01
    fused[invalid_measurement] = predicted[invalid_measurement]

    # Compute disagreement (only where measurement is valid)
    valid_measurement = confidence > 0.3
    disagreement = np.zeros_like(predicted)
    if valid_measurement.any():
        disagreement[valid_measurement] = np.abs(
            predicted[valid_measurement] - measured[valid_measurement]
        )

    # Uncertainty estimate: higher where disagreement is large or confidence low
    uncertainty = np.sqrt(
        (w_p / w_total) ** 2 * 1.0 ** 2  # prediction uncertainty ~1mm
        + (w_m / w_total) ** 2 * (0.3 / np.maximum(confidence, 0.1)) ** 2
        + disagreement ** 2 * 0.5
    )

    # Check if replanning is needed
    valid_fraction = float(valid_measurement.sum()) / max(1, valid_measurement.size)
    mean_disagreement = float(disagreement[valid_measurement].mean()) if valid_measurement.any() else 0.0
    needs_replan = mean_disagreement > config.replan_threshold_mm

    if mean_disagreement > config.disagreement_threshold_mm:
        logger.warning(
            "Depth disagreement: mean=%.1fmm (threshold=%.1fmm), replan=%s",
            mean_disagreement, config.replan_threshold_mm, needs_replan,
        )
    else:
        logger.info(
            "Depth fusion: valid=%.0f%%, mean_disagreement=%.2fmm, uncertainty=%.2fmm",
            valid_fraction * 100, mean_disagreement, float(uncertainty[valid_measurement].mean()) if valid_measurement.any() else 0,
        )

    return {
        "fused_depth_mm": fused.astype(np.float32),
        "uncertainty_mm": uncertainty.astype(np.float32),
        "needs_replan": needs_replan,
        "disagreement_mm": disagreement.astype(np.float32),
        "valid_fraction": valid_fraction,
        "mean_disagreement_mm": mean_disagreement,
    }


def fuse_depth_polar(
    predicted_radial_depth_mm: np.ndarray,
    measured_depth_map_mm: np.ndarray,
    measurement_confidence: np.ndarray,
    centroid_px: np.ndarray,
    num_radii: int = 64,
    image_size: int = 128,
    config: Optional[DepthFusionConfig] = None,
) -> Dict[str, np.ndarray]:
    """Fuse depth in polar representation (matches decoder output format).

    The decoder outputs depth at N angular directions from the wound center.
    The sensor gives a 2D depth map. This function samples the depth map
    along the same radial directions and fuses.

    Args:
        predicted_radial_depth_mm: (num_radii,) depth at each angle from decoder
        measured_depth_map_mm: (H, W) full depth map from sensor
        measurement_confidence: (H, W) confidence map
        centroid_px: (2,) wound center in pixel coordinates [x, y]
        num_radii: number of angular samples
        image_size: depth map size (assumes square)
        config: fusion parameters

    Returns:
        Same as fuse_depth() but operating on the polar representation
    """
    if config is None:
        config = DepthFusionConfig()

    H, W = measured_depth_map_mm.shape
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_radii), num_radii)

    # Sample measured depth along radial directions from centroid
    measured_radial = np.zeros(num_radii, dtype=np.float32)
    confidence_radial = np.zeros(num_radii, dtype=np.float32)

    max_radius_px = min(H, W) // 2

    for i, angle in enumerate(angles):
        # Walk along the ray from centroid outward, find wound edge depth
        samples = []
        confs = []
        for r in range(5, max_radius_px):
            px = int(centroid_px[0] + r * np.cos(angle))
            py = int(centroid_px[1] + r * np.sin(angle))
            if 0 <= px < W and 0 <= py < H:
                d = measured_depth_map_mm[py, px]
                c = measurement_confidence[py, px]
                if d > 0 and c > 0.1:
                    samples.append(d)
                    confs.append(c)

        if samples:
            # Use the deepest measurement along this ray as wound depth
            max_idx = np.argmax(samples)
            measured_radial[i] = samples[max_idx]
            confidence_radial[i] = confs[max_idx]

    return fuse_depth(
        predicted_radial_depth_mm,
        measured_radial,
        confidence_radial,
        config=config,
    )
