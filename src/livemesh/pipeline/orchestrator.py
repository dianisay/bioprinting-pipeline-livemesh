"""
LiveMesh Pipeline Orchestrator

End-to-end pipeline connecting all modules:

    Camera frame
        |
        v
    [1. PERCEIVE]  RGB image --> polar boundary (CNN-Transformer + PolarDecoder)
        |                    --> wound mask (U-Net segmentation)
        v
    [2. RECONSTRUCT]  Depth point cloud --> smooth mesh (Poisson / DeepCurrents)
        |                               + boundary-aware mesh update
        v
    [3. PLAN]  Mesh --> geodesic toolpaths (heat method, curvature-adaptive)
        |           --> conformal UV-to-XYZ mapping
        |           --> coverage validation (OT-based)
        v
    [4. EXECUTE]  Toolpath --> G-code / ROS2 trajectory / 8-DOF IK
        |
        v
    [5. FEEDBACK]  CNN visual feedback --> mesh update --> re-plan
        |
        v
    (loop back to PERCEIVE)

Each stage is independently testable. The orchestrator wires them together
and manages the real-time loop timing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Unified configuration for the full pipeline."""

    # Perception
    perception_model: str = "polar"  # polar | unet
    encoder_checkpoint: str | None = None
    decoder_checkpoint: str | None = None
    input_size: tuple[int, int] = (256, 256)
    device: str = "cuda"

    # Reconstruction
    reconstruction_method: str = "poisson"  # poisson | deep_currents
    poisson_depth: int = 8
    target_latency_ms: float = 100.0

    # Toolpath
    toolpath_method: str = "geodesic"  # geodesic | planar | honeycomb
    spacing_mm: float = 1.5
    adaptive_curvature: bool = True
    replan_budget_ms: float = 500.0

    # Robot
    robot_type: str = "gcode"  # gcode | ros2 | mycobot
    safe_z_mm: float = 5.0
    feed_rate_mm_min: float = 1500.0

    # Pipeline timing
    depth_fps: int = 30
    reconstruction_hz: int = 10
    cnn_hz: int = 15
    log_dir: str = "logs/"


@dataclass
class PipelineState:
    """Mutable state tracked across pipeline iterations."""

    current_mesh: Any = None
    current_toolpath: Any = None
    current_waypoints: NDArray | None = None
    current_normals: NDArray | None = None
    iteration: int = 0
    total_elapsed_ms: float = 0.0
    stage_timings: dict[str, list[float]] = field(default_factory=dict)


class LiveMeshPipeline:
    """Main orchestrator connecting perception, reconstruction, toolpath, and execution.

    Usage:
        config = PipelineConfig(
            perception_model="polar",
            reconstruction_method="poisson",
            toolpath_method="geodesic",
            robot_type="gcode",
        )
        pipeline = LiveMeshPipeline(config)
        pipeline.load_models()

        # Single frame processing
        result = pipeline.process_frame(rgb_image, depth_points)

        # Or full loop
        pipeline.run_loop(camera_stream)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState()
        self._models_loaded = False

    def load_models(self) -> None:
        """Load perception models (encoder + decoder)."""
        if self.config.perception_model == "polar":
            from livemesh.perception.encoder import CNNTransformerEncoder
            from livemesh.perception.polar_decoder import PolarDecoder

            self.encoder = CNNTransformerEncoder(pretrained=True)
            self.decoder = PolarDecoder()

            if self.config.encoder_checkpoint:
                ckpt = torch.load(self.config.encoder_checkpoint, map_location=self.config.device)
                self.encoder.load_state_dict(ckpt.get("encoder", ckpt))
            if self.config.decoder_checkpoint:
                ckpt = torch.load(self.config.decoder_checkpoint, map_location=self.config.device)
                self.decoder.load_state_dict(ckpt.get("decoder", ckpt))

            self.encoder = self.encoder.to(self.config.device).eval()
            self.decoder = self.decoder.to(self.config.device).eval()

        elif self.config.perception_model == "unet":
            from livemesh.segmentation.unet import UNet

            self.segmentation_model = UNet()
            if self.config.encoder_checkpoint:
                ckpt = torch.load(self.config.encoder_checkpoint, map_location=self.config.device)
                self.segmentation_model.load_state_dict(ckpt)
            self.segmentation_model = self.segmentation_model.to(self.config.device).eval()

        self._models_loaded = True
        logger.info("Models loaded: %s", self.config.perception_model)

    def process_frame(
        self,
        rgb_image: NDArray[np.uint8] | None = None,
        depth_points: NDArray[np.float64] | None = None,
        depth_normals: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Process a single camera frame through the full pipeline.

        Parameters
        ----------
        rgb_image : (H, W, 3) uint8 RGB image for wound perception
        depth_points : (N, 3) point cloud from depth camera for surface reconstruction
        depth_normals : (N, 3) optional normals for the point cloud

        Returns dict with keys: perception, reconstruction, toolpath, robot_commands
        """
        t0 = time.perf_counter()
        result = {}

        # Stage 1: PERCEIVE
        if rgb_image is not None:
            result["perception"] = self._perceive(rgb_image)

        # Stage 2: RECONSTRUCT
        if depth_points is not None:
            result["reconstruction"] = self._reconstruct(depth_points, depth_normals)

        # Stage 3: PLAN
        if self.state.current_mesh is not None:
            result["toolpath"] = self._plan_toolpath()

        # Stage 4: GENERATE COMMANDS
        if self.state.current_waypoints is not None:
            result["robot_commands"] = self._generate_commands()

        self.state.iteration += 1
        elapsed = (time.perf_counter() - t0) * 1000
        self.state.total_elapsed_ms += elapsed
        result["elapsed_ms"] = elapsed
        result["iteration"] = self.state.iteration

        return result

    def _perceive(self, rgb_image: NDArray[np.uint8]) -> dict[str, Any]:
        """Stage 1: Extract wound boundary from RGB image."""
        t0 = time.perf_counter()

        if self.config.perception_model == "polar":
            import cv2

            resized = cv2.resize(rgb_image, self.config.input_size)
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            tensor = tensor.unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                features = self.encoder(tensor)
                prediction = self.decoder(features)

            result = {
                "centroid": prediction["centroid"].cpu().numpy(),
                "radii": prediction["radii"].cpu().numpy(),
                "points": prediction["points"].cpu().numpy(),
            }

        elif self.config.perception_model == "unet":
            from livemesh.segmentation.wound_pipeline import segment_wound

            seg_result = segment_wound(
                rgb_image, self.segmentation_model,
                device=self.config.device,
                input_size=self.config.input_size,
            )
            result = {
                "mask": seg_result.mask,
                "boundary": seg_result.boundary,
                "boundary_mm": seg_result.boundary_mm,
                "area_mm2": seg_result.area_mm2,
            }

        elapsed = (time.perf_counter() - t0) * 1000
        self._log_timing("perceive", elapsed)
        result["elapsed_ms"] = elapsed
        return result

    def _reconstruct(
        self,
        points: NDArray[np.float64],
        normals: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Stage 2: Build mesh from depth point cloud."""
        t0 = time.perf_counter()

        from livemesh.reconstruction.poisson import poisson_reconstruct

        rec = poisson_reconstruct(
            points, normals=normals, depth=self.config.poisson_depth
        )
        self.state.current_mesh = rec.mesh

        elapsed = (time.perf_counter() - t0) * 1000
        self._log_timing("reconstruct", elapsed)

        return {
            "mesh": rec.mesh,
            "num_vertices": rec.num_output_vertices,
            "elapsed_ms": elapsed,
        }

    def _plan_toolpath(self) -> dict[str, Any]:
        """Stage 3: Generate deposition paths on the reconstructed mesh."""
        t0 = time.perf_counter()

        if self.config.toolpath_method == "geodesic":
            from livemesh.toolpath.geodesic import geodesic_toolpaths

            tp = geodesic_toolpaths(
                self.state.current_mesh,
                spacing_mm=self.config.spacing_mm,
                adaptive_curvature=self.config.adaptive_curvature,
            )
            self.state.current_waypoints = tp.waypoints
            self.state.current_normals = tp.normals
            self.state.current_toolpath = tp

            result = {
                "num_paths": tp.num_paths,
                "total_length_mm": tp.total_length_mm,
                "waypoints": tp.waypoints,
            }

        elif self.config.toolpath_method == "planar":
            from livemesh.toolpath.planar_slicer import planar_slice

            ps = planar_slice(self.state.current_mesh, line_spacing_mm=self.config.spacing_mm)
            self.state.current_waypoints = ps.waypoints
            self.state.current_toolpath = ps

            result = {
                "num_layers": ps.num_layers,
                "total_length_mm": ps.total_length_mm,
                "waypoints": ps.waypoints,
            }

        elif self.config.toolpath_method == "honeycomb":
            from livemesh.toolpath.honeycomb import compute_grid_params, create_hex_grid

            result = {"method": "honeycomb", "note": "Use trajectory_planner for full honeycomb pipeline"}

        elapsed = (time.perf_counter() - t0) * 1000
        self._log_timing("plan", elapsed)
        result["elapsed_ms"] = elapsed
        return result

    def _generate_commands(self) -> dict[str, Any]:
        """Stage 4: Convert toolpath to robot-executable commands."""
        t0 = time.perf_counter()

        if self.config.robot_type == "gcode":
            from livemesh.toolpath.path_to_robot import toolpath_to_gcode

            is_dep = None
            if hasattr(self.state.current_toolpath, "is_deposition"):
                is_dep = self.state.current_toolpath.is_deposition
                if len(is_dep) == 0:
                    is_dep = None

            gcode = toolpath_to_gcode(
                self.state.current_waypoints,
                is_deposition=is_dep,
                safe_z_mm=self.config.safe_z_mm,
                feed_rate=self.config.feed_rate_mm_min,
            )
            result = {"format": "gcode", "num_lines": len(gcode), "commands": gcode}

        elif self.config.robot_type == "mycobot":
            from livemesh.robot.inverse_kinematics import InverseKinematicsSolver

            solver = InverseKinematicsSolver()
            result = {
                "format": "joint_angles",
                "note": "Use solver.solve_trajectory(waypoints, orientations) for full IK",
                "num_waypoints": len(self.state.current_waypoints),
            }

        elif self.config.robot_type == "ros2":
            from livemesh.toolpath.path_to_robot import toolpath_to_ros2_trajectory

            tool_frames = np.tile(np.eye(3), (len(self.state.current_waypoints), 1, 1))
            if self.state.current_normals is not None and len(self.state.current_normals) > 0:
                pass  # tool_frames would be computed from normals

            traj = toolpath_to_ros2_trajectory(
                self.state.current_waypoints,
                self.state.current_normals if self.state.current_normals is not None else np.zeros_like(self.state.current_waypoints),
                tool_frames,
            )
            result = {"format": "ros2", "num_poses": len(traj), "trajectory": traj}

        elapsed = (time.perf_counter() - t0) * 1000
        self._log_timing("execute", elapsed)
        result["elapsed_ms"] = elapsed
        return result

    def benchmark(
        self,
        ground_truth_mesh: Any,
        num_trials: int = 5,
    ) -> dict[str, Any]:
        """Run benchmark comparing toolpath methods on the current mesh.

        Returns comparative metrics: geodesic vs planar vs honeycomb.
        """
        from livemesh.reconstruction.benchmarks import benchmark_reconstruction
        from livemesh.toolpath.coverage import coverage_comparison_table
        from livemesh.toolpath.geodesic import geodesic_toolpaths
        from livemesh.toolpath.planar_slicer import planar_slice

        mesh = self.state.current_mesh or ground_truth_mesh

        rec_bench = None
        if self.state.current_mesh is not None and ground_truth_mesh is not None:
            rec_bench = benchmark_reconstruction(self.state.current_mesh, ground_truth_mesh)

        geodesic_tp = geodesic_toolpaths(mesh, spacing_mm=self.config.spacing_mm)
        planar_tp = planar_slice(mesh, line_spacing_mm=self.config.spacing_mm)

        toolpaths = {}
        if len(geodesic_tp.waypoints) > 0:
            toolpaths["geodesic"] = geodesic_tp.waypoints
        if len(planar_tp.waypoints) > 0:
            toolpaths["planar"] = planar_tp.waypoints

        coverage = coverage_comparison_table(mesh, toolpaths) if toolpaths else {}

        return {
            "reconstruction": rec_bench,
            "coverage": coverage,
            "geodesic_stats": {
                "num_paths": geodesic_tp.num_paths,
                "total_length_mm": geodesic_tp.total_length_mm,
                "elapsed_ms": geodesic_tp.elapsed_ms,
            },
            "planar_stats": {
                "num_layers": planar_tp.num_layers,
                "total_length_mm": planar_tp.total_length_mm,
            },
        }

    def _log_timing(self, stage: str, elapsed_ms: float) -> None:
        if stage not in self.state.stage_timings:
            self.state.stage_timings[stage] = []
        self.state.stage_timings[stage].append(elapsed_ms)

    def timing_summary(self) -> dict[str, dict[str, float]]:
        """Return mean/max/min timing per pipeline stage."""
        summary = {}
        for stage, times in self.state.stage_timings.items():
            arr = np.array(times)
            summary[stage] = {
                "mean_ms": float(np.mean(arr)),
                "max_ms": float(np.max(arr)),
                "min_ms": float(np.min(arr)),
                "count": len(times),
            }
        return summary
