"""Closed-loop bioprinting controller: scan-plan-deposit-verify-correct cycle.

Implements the real-time feedback loop for layer-by-layer wound filling:
1. SCAN: acquire depth measurement of current wound state
2. PLAN: compute (or update) the trajectory for the next layer
3. DEPOSIT: execute one layer of bioink deposition
4. VERIFY: re-scan to measure actual fill vs. expected fill
5. CORRECT: adjust next layer's fill amounts based on error

This produces monotonically decreasing wound depth until the wound
is filled to the target level, regardless of initial prediction errors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import logging
from .depth_sensor import DepthSensorModel, DepthSensorBase
from .depth_fusion import fuse_depth, DepthFusionConfig

logger = logging.getLogger(__name__)


@dataclass
class PrintingState:
    """Tracks the state of the wound filling process."""
    current_layer: int = 0
    total_layers: int = 4
    initial_depth_mm: Optional[np.ndarray] = None
    current_depth_mm: Optional[np.ndarray] = None
    target_depth_mm: float = 0.0  # fill to surface level
    layer_history: List[Dict] = field(default_factory=list)
    total_fill_error_mm: float = 0.0
    is_complete: bool = False


@dataclass
class LayerResult:
    """Result of one deposit-verify cycle."""
    layer_idx: int
    planned_fill_mm: np.ndarray
    actual_fill_mm: np.ndarray
    fill_error_mm: np.ndarray
    mean_error_mm: float
    max_error_mm: float
    correction_applied: np.ndarray


class PrintingLoopController:
    """Closed-loop bioprinting with depth feedback.

    The controller operates in a loop:
    - Before printing: full scan to establish wound geometry
    - Per layer: deposit -> verify -> correct
    - Completion: when remaining depth < layer_height

    In simulation mode, the "deposition" is simulated by reducing
    the wound depth by the planned fill amount (with noise).
    In real hardware mode, the robot executes the trajectory and
    the sensor measures the actual result.
    """

    def __init__(
        self,
        sensor: DepthSensorBase,
        layer_height_mm: float = 0.4,
        num_layers: int = 4,
        correction_gain: float = 0.7,
        max_correction_factor: float = 1.5,
        deposition_noise_sigma: float = 0.1,
        fusion_config: Optional[DepthFusionConfig] = None,
    ):
        """
        Args:
            sensor: depth sensor (simulated or real)
            layer_height_mm: nominal height per deposition layer
            num_layers: maximum number of layers to deposit
            correction_gain: how aggressively to correct (0-1, higher = more aggressive)
            max_correction_factor: max multiplier on layer_amounts (cap over-correction)
            deposition_noise_sigma: simulation noise on deposited material (mm)
            fusion_config: depth fusion parameters
        """
        self.sensor = sensor
        self.layer_height_mm = layer_height_mm
        self.num_layers = num_layers
        self.correction_gain = correction_gain
        self.max_correction_factor = max_correction_factor
        self.deposition_noise_sigma = deposition_noise_sigma
        self.fusion_config = fusion_config or DepthFusionConfig()

        self.state = PrintingState(total_layers=num_layers)
        self._rng = np.random.default_rng(42)

        logger.info(
            "PrintingLoopController: layers=%d, height=%.1fmm, "
            "correction_gain=%.2f",
            num_layers, layer_height_mm, correction_gain,
        )

    def scan_and_plan(
        self,
        true_depth_mm: np.ndarray,
        predicted_depth_mm: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Initial scan to establish wound geometry and plan filling.

        Args:
            true_depth_mm: (N,) ground truth wound depth (for simulation)
            predicted_depth_mm: (N,) depth from volumetric decoder (optional)

        Returns:
            dict with:
                - initial_depth_mm: measured wound depth
                - layer_plan: (N, num_layers) planned fill per point per layer
                - total_fill_needed_mm: (N,) total depth to fill
        """
        # Simulate sensor measurement of the wound
        measurement = self.sensor.simulate_measurement(
            true_depth_mm.reshape(1, -1) if true_depth_mm.ndim == 1
            else true_depth_mm
        )
        measured_depth = measurement["depth_mm"].flatten()[:len(true_depth_mm)]
        confidence = measurement["confidence"].flatten()[:len(true_depth_mm)]

        # Fuse with prediction if available
        if predicted_depth_mm is not None:
            fusion_result = fuse_depth(
                predicted_depth_mm, measured_depth, confidence,
                config=self.fusion_config,
            )
            initial_depth = fusion_result["fused_depth_mm"]
        else:
            initial_depth = np.where(confidence > 0.3, measured_depth, 0.0)

        self.state.initial_depth_mm = initial_depth.copy()
        self.state.current_depth_mm = initial_depth.copy()

        # Plan layer-by-layer fill: distribute depth evenly across layers
        total_fill = np.maximum(initial_depth - self.state.target_depth_mm, 0)
        layer_plan = np.zeros((len(total_fill), self.num_layers), dtype=np.float32)

        for i in range(self.num_layers):
            # Each layer fills an equal portion of remaining depth
            layer_plan[:, i] = total_fill / self.num_layers

        logger.info(
            "Scan complete: mean_depth=%.1fmm, max_depth=%.1fmm, "
            "fill_needed=%.1fmm over %d layers",
            initial_depth.mean(), initial_depth.max(),
            total_fill.mean(), self.num_layers,
        )

        return {
            "initial_depth_mm": initial_depth,
            "layer_plan": layer_plan,
            "total_fill_needed_mm": total_fill,
        }

    def deposit_layer(
        self,
        layer_plan_mm: np.ndarray,
    ) -> np.ndarray:
        """Simulate depositing one layer of material.

        In simulation: reduces current_depth by planned amount + noise.
        In real hardware: this would execute the trajectory and return.

        Args:
            layer_plan_mm: (N,) planned fill amount for this layer

        Returns:
            (N,) actual deposited amount (with noise)
        """
        # Simulate imperfect deposition (noise represents material flow variation)
        noise = self._rng.normal(0, self.deposition_noise_sigma, size=layer_plan_mm.shape)
        actual_deposit = np.maximum(layer_plan_mm + noise, 0)

        # Update wound state
        self.state.current_depth_mm = np.maximum(
            self.state.current_depth_mm - actual_deposit, 0
        )

        return actual_deposit.astype(np.float32)

    def verify_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Re-scan after deposition to verify fill quality.

        Args:
            layer_idx: which layer was just deposited

        Returns:
            dict with current_depth_mm, fill_error_mm, etc.
        """
        # Measure current state
        measurement = self.sensor.simulate_measurement(
            self.state.current_depth_mm.reshape(1, -1)
        )
        measured_depth = measurement["depth_mm"].flatten()[:len(self.state.current_depth_mm)]
        confidence = measurement["confidence"].flatten()[:len(self.state.current_depth_mm)]

        # Where measurement is valid, update our depth estimate
        valid = confidence > 0.3
        verified_depth = self.state.current_depth_mm.copy()
        verified_depth[valid] = measured_depth[valid]

        # Expected depth after this layer
        expected_remaining = self.state.initial_depth_mm * (
            (self.num_layers - layer_idx - 1) / self.num_layers
        )
        fill_error = verified_depth - expected_remaining

        self.state.current_depth_mm = verified_depth

        result = {
            "current_depth_mm": verified_depth,
            "expected_depth_mm": expected_remaining,
            "fill_error_mm": fill_error,
            "mean_error_mm": float(np.abs(fill_error).mean()),
            "max_error_mm": float(np.abs(fill_error).max()),
            "valid_fraction": float(valid.mean()),
        }

        logger.info(
            "Layer %d verified: remaining=%.1fmm, error=%.2fmm (max=%.2fmm)",
            layer_idx, verified_depth.mean(), result["mean_error_mm"],
            result["max_error_mm"],
        )

        return result

    def correct_next_layer(
        self,
        base_plan_mm: np.ndarray,
        fill_error_mm: np.ndarray,
    ) -> np.ndarray:
        """Adjust next layer's fill plan based on measured error.

        If the previous layer under-filled (error > 0), increase next layer.
        If over-filled (error < 0), reduce next layer.

        Args:
            base_plan_mm: (N,) original plan for next layer
            fill_error_mm: (N,) signed error from verification

        Returns:
            (N,) corrected plan for next layer
        """
        correction = fill_error_mm * self.correction_gain
        corrected = base_plan_mm + correction

        # Clamp to prevent negative deposition or excessive over-correction
        corrected = np.clip(
            corrected,
            0.0,
            base_plan_mm * self.max_correction_factor,
        )

        mean_correction = float(np.abs(correction).mean())
        logger.info(
            "Correction applied: mean=%.3fmm, gain=%.2f",
            mean_correction, self.correction_gain,
        )

        return corrected.astype(np.float32)

    def run_full_cycle(
        self,
        true_depth_mm: np.ndarray,
        predicted_depth_mm: Optional[np.ndarray] = None,
    ) -> Dict:
        """Execute the complete closed-loop printing cycle.

        This is the main entry point for simulation. Runs:
        scan -> (deposit -> verify -> correct) x num_layers -> summary

        Args:
            true_depth_mm: (N,) ground truth wound depth
            predicted_depth_mm: (N,) optional prediction from decoder

        Returns:
            dict with full history and summary metrics
        """
        logger.info("=" * 50)
        logger.info("CLOSED-LOOP PRINTING CYCLE START")
        logger.info("=" * 50)

        # Initial scan and plan
        plan = self.scan_and_plan(true_depth_mm, predicted_depth_mm)
        layer_plan = plan["layer_plan"]

        layer_results = []

        for layer_idx in range(self.num_layers):
            logger.info("--- Layer %d/%d ---", layer_idx + 1, self.num_layers)

            # Get current plan for this layer
            current_plan = layer_plan[:, layer_idx].copy()

            # Apply correction from previous layer
            if layer_idx > 0 and layer_results:
                prev_error = layer_results[-1].fill_error_mm
                current_plan = self.correct_next_layer(current_plan, prev_error)

            # Deposit
            actual_deposit = self.deposit_layer(current_plan)

            # Verify
            verification = self.verify_layer(layer_idx)

            # Record result
            result = LayerResult(
                layer_idx=layer_idx,
                planned_fill_mm=current_plan,
                actual_fill_mm=actual_deposit,
                fill_error_mm=verification["fill_error_mm"],
                mean_error_mm=verification["mean_error_mm"],
                max_error_mm=verification["max_error_mm"],
                correction_applied=(
                    current_plan - layer_plan[:, layer_idx]
                    if layer_idx > 0 else np.zeros_like(current_plan)
                ),
            )
            layer_results.append(result)
            self.state.layer_history.append({
                "layer": layer_idx,
                "mean_error": result.mean_error_mm,
                "max_error": result.max_error_mm,
            })

            # Check if wound is filled
            if self.state.current_depth_mm.max() < self.layer_height_mm:
                logger.info("Wound filled! Remaining depth < layer height.")
                self.state.is_complete = True
                break

        self.state.current_layer = len(layer_results)

        # Summary
        initial_depth_mean = float(self.state.initial_depth_mm.mean())
        final_depth_mean = float(self.state.current_depth_mm.mean())
        fill_percentage = (
            (initial_depth_mean - final_depth_mean) / max(initial_depth_mean, 0.01) * 100
        )

        summary = {
            "layers_deposited": len(layer_results),
            "initial_depth_mean_mm": initial_depth_mean,
            "final_depth_mean_mm": final_depth_mean,
            "fill_percentage": fill_percentage,
            "layer_errors_mm": [r.mean_error_mm for r in layer_results],
            "is_complete": self.state.is_complete,
            "layer_results": layer_results,
        }

        logger.info("=" * 50)
        logger.info(
            "CYCLE COMPLETE: %d layers, fill=%.1f%%, "
            "residual=%.2fmm",
            len(layer_results), fill_percentage, final_depth_mean,
        )
        logger.info("=" * 50)

        return summary
