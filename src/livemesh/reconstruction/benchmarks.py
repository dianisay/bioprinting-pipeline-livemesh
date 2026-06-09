"""
Reconstruction benchmarking: compare methods on accuracy, latency, and boundary fidelity.

Metrics:
- Hausdorff distance (max error)
- Mean surface distance
- Normal angular deviation
- Latency (ms per reconstruction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)
import trimesh
from numpy.typing import NDArray


@dataclass
class BenchmarkResult:
    hausdorff_mm: float
    mean_distance_mm: float
    normal_deviation_deg: float
    elapsed_ms: float
    num_vertices: int


def benchmark_reconstruction(
    reconstructed: trimesh.Trimesh,
    ground_truth: trimesh.Trimesh,
    elapsed_ms: float = 0.0,
    num_samples: int = 10000,
) -> BenchmarkResult:
    """Compare a reconstructed mesh against ground truth.

    Uses dense point sampling for robust distance computation.
    """
    logger.info(
        f"Benchmarking reconstruction: {len(reconstructed.vertices)} vertices vs "
        f"{len(ground_truth.vertices)} GT vertices, {num_samples} surface samples"
    )
    gt_points, gt_face_idx = trimesh.sample.sample_surface(ground_truth, num_samples)
    gt_normals = ground_truth.face_normals[gt_face_idx]

    closest_points, distances, face_idx = trimesh.proximity.closest_point(
        reconstructed, gt_points
    )
    rec_normals = reconstructed.face_normals[face_idx]

    hausdorff = float(np.max(distances))
    mean_dist = float(np.mean(distances))

    cos_angles = np.sum(gt_normals * rec_normals, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(np.abs(cos_angles)))
    mean_normal_dev = float(np.mean(angles_deg))

    logger.info(
        f"Benchmark metrics: hausdorff={hausdorff:.3f} mm, mean_distance={mean_dist:.3f} mm, "
        f"normal_deviation={mean_normal_dev:.2f} deg, elapsed={elapsed_ms:.1f} ms"
    )

    return BenchmarkResult(
        hausdorff_mm=hausdorff,
        mean_distance_mm=mean_dist,
        normal_deviation_deg=mean_normal_dev,
        elapsed_ms=elapsed_ms,
        num_vertices=len(reconstructed.vertices),
    )


def hausdorff_symmetric(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    num_samples: int = 10000,
) -> float:
    """Symmetric Hausdorff distance: max(d(A,B), d(B,A))."""
    logger.debug(f"Computing symmetric Hausdorff with {num_samples} samples per mesh")
    pts_a = trimesh.sample.sample_surface(mesh_a, num_samples)[0]
    pts_b = trimesh.sample.sample_surface(mesh_b, num_samples)[0]

    _, dists_ab, _ = trimesh.proximity.closest_point(mesh_b, pts_a)
    _, dists_ba, _ = trimesh.proximity.closest_point(mesh_a, pts_b)

    result = float(max(np.max(dists_ab), np.max(dists_ba)))
    logger.info(f"Symmetric Hausdorff distance: {result:.3f} mm")
    return result
