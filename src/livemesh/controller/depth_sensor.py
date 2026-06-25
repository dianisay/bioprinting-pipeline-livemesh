"""Depth sensor model: simulated Intel RealSense D405 and real hardware interface.

The D405 is a short-range stereo depth camera optimized for robotic manipulation:
- Range: 7-500mm (optimal 70-350mm)
- Depth accuracy: <0.5mm at 100mm working distance
- Resolution: 1280x720 (depth), 1280x720 (RGB)
- Frame rate: up to 90fps
- FOV: 87 x 58 degrees
- Baseline: 18mm (stereo)

This module provides:
1. DepthSensorModel: simulation with realistic noise for in-silico validation
2. RealSenseD405: real hardware interface stub (MIT phase)

Both share the same output interface for seamless swap.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


# D405 specifications (from Intel datasheet)
D405_SPECS = {
    "resolution": (1280, 720),
    "depth_resolution": (1280, 720),
    "fov_h_deg": 87.0,
    "fov_v_deg": 58.0,
    "min_range_mm": 7.0,
    "max_range_mm": 500.0,
    "optimal_min_mm": 70.0,
    "optimal_max_mm": 350.0,
    "baseline_mm": 18.0,
    "depth_noise_sigma_mm": 0.3,  # at 100mm working distance
    "fps": 90,
    "form_factor_mm": (42, 42, 23),
}


class DepthSensorBase(ABC):
    """Abstract interface for depth sensors (simulated or real)."""

    @abstractmethod
    def capture(self) -> Dict[str, np.ndarray]:
        """Capture a single frame.

        Returns:
            dict with:
                - depth_mm: (H, W) float32 depth in millimeters
                - confidence: (H, W) float32 in [0, 1], 0 = invalid/no data
                - rgb: (H, W, 3) uint8 color image (aligned to depth)
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Dict[str, float]:
        """Return camera intrinsic parameters.

        Returns:
            dict with fx, fy, cx, cy, width, height
        """
        pass


class DepthSensorModel(DepthSensorBase):
    """Simulated RealSense D405 with realistic noise model.

    Noise characteristics modeled from D405 datasheet:
    - Depth noise increases quadratically with distance
    - Invalid pixels at specular/wet surfaces (simulated)
    - Quantization to 0.1mm steps
    - Range clipping outside [min_range, max_range]
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (128, 128),
        working_distance_mm: float = 150.0,
        noise_sigma_base_mm: float = 0.3,
        specular_dropout_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Args:
            resolution: output depth map resolution (downsampled from native)
            working_distance_mm: expected camera-to-surface distance
            noise_sigma_base_mm: base noise at 100mm (scales with distance^2)
            specular_dropout_rate: fraction of pixels invalidated by specularity
            seed: random seed for reproducibility
        """
        self.resolution = resolution
        self.working_distance_mm = working_distance_mm
        self.noise_sigma_base_mm = noise_sigma_base_mm
        self.specular_dropout_rate = specular_dropout_rate
        self.rng = np.random.default_rng(seed)

        self.min_range_mm = D405_SPECS["min_range_mm"]
        self.max_range_mm = D405_SPECS["max_range_mm"]
        self.quantization_mm = 0.1

        # Compute focal length from FOV and resolution
        fov_h_rad = np.radians(D405_SPECS["fov_h_deg"])
        self.fx = resolution[0] / (2 * np.tan(fov_h_rad / 2))
        self.fy = self.fx  # square pixels
        self.cx = resolution[0] / 2.0
        self.cy = resolution[1] / 2.0

        logger.info(
            "DepthSensorModel: res=%dx%d, working_dist=%.0fmm, "
            "noise_sigma=%.2fmm, dropout=%.1f%%",
            resolution[0], resolution[1], working_distance_mm,
            noise_sigma_base_mm, specular_dropout_rate * 100,
        )

    def get_intrinsics(self) -> Dict[str, float]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.resolution[0],
            "height": self.resolution[1],
        }

    def simulate_measurement(
        self,
        true_depth_mm: np.ndarray,
        surface_normals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate a depth measurement given ground truth.

        Args:
            true_depth_mm: (H, W) ground truth depth in mm
            surface_normals: (H, W, 3) optional surface normals for
                            grazing-angle dropout simulation

        Returns:
            dict with depth_mm, confidence, rgb (synthetic)
        """
        H, W = true_depth_mm.shape
        depth = true_depth_mm.copy().astype(np.float32)

        # 1. Distance-dependent noise (quadratic scaling from D405 model)
        # sigma(d) = sigma_base * (d / 100)^2
        distance_factor = (depth / 100.0) ** 2
        noise_sigma = self.noise_sigma_base_mm * distance_factor
        noise = self.rng.normal(0, np.maximum(noise_sigma, 0.01)).astype(np.float32)
        depth += noise

        # 2. Quantization (stereo depth cameras have discrete depth steps)
        depth = np.round(depth / self.quantization_mm) * self.quantization_mm

        # 3. Range clipping
        out_of_range = (depth < self.min_range_mm) | (depth > self.max_range_mm)
        depth[out_of_range] = 0.0

        # 4. Specular/wet surface dropout (random pixels become invalid)
        specular_mask = self.rng.random((H, W)) < self.specular_dropout_rate
        depth[specular_mask] = 0.0

        # 5. Grazing angle dropout (stereo fails at oblique viewing angles)
        if surface_normals is not None:
            view_dir = np.array([0, 0, 1], dtype=np.float32)
            cos_angle = np.abs(np.dot(surface_normals, view_dir))
            grazing = cos_angle < 0.3  # >72 degrees from normal
            depth[grazing] = 0.0

        # Confidence map: 1.0 for valid, 0.0 for invalid
        confidence = (depth > 0).astype(np.float32)

        # Reduce confidence near edges and at range limits
        valid = depth > 0
        if valid.any():
            dist_from_optimal = np.abs(depth - self.working_distance_mm)
            confidence[valid] *= np.exp(
                -(dist_from_optimal[valid] ** 2) / (200.0 ** 2)
            )
            confidence[valid] = np.clip(confidence[valid], 0.3, 1.0)

        # Synthetic RGB (placeholder - in real system comes from camera)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        return {
            "depth_mm": depth,
            "confidence": confidence,
            "rgb": rgb,
        }

    def capture(self) -> Dict[str, np.ndarray]:
        """Capture from a flat surface at working distance (for testing)."""
        H, W = self.resolution[1], self.resolution[0]
        true_depth = np.full((H, W), self.working_distance_mm, dtype=np.float32)
        return self.simulate_measurement(true_depth)

    def depth_to_pointcloud(
        self,
        depth_mm: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        min_confidence: float = 0.5,
    ) -> np.ndarray:
        """Convert depth map to 3D point cloud in camera frame.

        Args:
            depth_mm: (H, W) depth values in mm
            confidence: (H, W) optional confidence mask
            min_confidence: threshold for including points

        Returns:
            (N, 3) point cloud in mm [x, y, z]
        """
        H, W = depth_mm.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        valid = depth_mm > 0
        if confidence is not None:
            valid &= confidence >= min_confidence

        z = depth_mm[valid]
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy

        return np.column_stack([x, y, z])


class RealSenseD405(DepthSensorBase):
    """Real Intel RealSense D405 hardware interface.

    Requires: pip install pyrealsense2

    This is a stub for the MIT physical implementation phase.
    The interface matches DepthSensorModel exactly, so swapping
    from simulation to real hardware requires only changing the
    instantiation.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        align_to_color: bool = True,
    ):
        self.resolution = resolution
        self.fps = fps
        self.align_to_color = align_to_color
        self._pipeline = None
        self._intrinsics = None

        logger.info(
            "RealSenseD405: res=%dx%d, fps=%d (hardware interface)",
            resolution[0], resolution[1], fps,
        )

    def connect(self):
        """Initialize the RealSense pipeline. Call once before capture()."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 not installed. Install with: "
                "pip install pyrealsense2\n"
                "For simulation, use DepthSensorModel instead."
            )

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            self.resolution[0], self.resolution[1],
            rs.format.z16, self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.resolution[0], self.resolution[1],
            rs.format.bgr8, self.fps,
        )

        profile = self._pipeline.start(config)
        depth_intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._intrinsics = {
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "cx": depth_intrinsics.ppx,
            "cy": depth_intrinsics.ppy,
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
        }

        logger.info("RealSense D405 connected: %s", self._intrinsics)

    def disconnect(self):
        """Stop the pipeline."""
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            logger.info("RealSense D405 disconnected")

    def capture(self) -> Dict[str, np.ndarray]:
        """Capture aligned RGB-D frame from hardware."""
        import pyrealsense2 as rs

        if self._pipeline is None:
            raise RuntimeError("Call connect() before capture()")

        frames = self._pipeline.wait_for_frames()

        if self.align_to_color:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_mm = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * depth_frame.get_units()
            * 1000.0
        )

        rgb = np.asanyarray(color_frame.get_data())

        confidence = (depth_mm > 0).astype(np.float32)

        return {
            "depth_mm": depth_mm,
            "confidence": confidence,
            "rgb": rgb,
        }

    def get_intrinsics(self) -> Dict[str, float]:
        if self._intrinsics is None:
            raise RuntimeError("Call connect() first to get intrinsics")
        return self._intrinsics
