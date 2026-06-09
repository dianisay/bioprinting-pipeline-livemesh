"""
Poisson surface reconstruction baseline.

This is the "known good" method: fast, well-understood, easy to tune.
If implicit methods (DeepCurrents) turn out too slow for real-time,
this is the fallback with boundary-aware post-processing.

Wraps Open3D's Poisson reconstruction with pre/post-processing
tailored to depth-camera point clouds of tissue surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
import open3d as o3d
import trimesh
from numpy.typing import NDArray


@dataclass
class ReconstructionResult:
    mesh: trimesh.Trimesh
    density: NDArray[np.float64]
    elapsed_ms: float
    num_input_points: int
    num_output_vertices: int


def poisson_reconstruct(
    points: NDArray[np.float64],
    normals: NDArray[np.float64] | None = None,
    depth: int = 8,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_threshold_quantile: float = 0.01,
    estimate_normals_k: int = 30,
) -> ReconstructionResult:
    """Reconstruct a surface from a noisy point cloud using Poisson reconstruction.

    Parameters
    ----------
    points : (N, 3) point cloud in mm
    normals : (N, 3) per-point normals. If None, estimated from local neighborhoods.
    depth : octree depth (higher = more detail, slower)
    scale : ratio between reconstruction cube and bounding box
    linear_fit : use linear interpolation at lowest octree levels
    density_threshold_quantile : remove low-density vertices (trims boundary artifacts)
    estimate_normals_k : neighbors for normal estimation if normals not provided
    """
    import time

    t0 = time.perf_counter()
    num_input = len(points)
    logger.info(
        f"Poisson reconstruction starting: {num_input} input points, "
        f"depth={depth}, scale={scale}, linear_fit={linear_fit}, "
        f"density_quantile={density_threshold_quantile}"
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
        logger.debug(f"Using {len(normals)} provided normals")
    else:
        logger.debug(f"Estimating normals with k={estimate_normals_k} neighbors")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=estimate_normals_k)
        )
        pcd.orient_normals_consistent_tangent_plane(k=estimate_normals_k)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )

    densities = np.asarray(densities)
    if density_threshold_quantile > 0:
        threshold = np.quantile(densities, density_threshold_quantile)
        vertices_to_remove = densities < threshold
        n_removed = int(np.sum(vertices_to_remove))
        logger.debug(
            f"Density trim: threshold={threshold:.4f}, removing {n_removed} low-density vertices"
        )
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        densities = densities[~vertices_to_remove]

    mesh = _o3d_to_trimesh(mesh_o3d)

    elapsed = (time.perf_counter() - t0) * 1000
    num_output = len(mesh.vertices)
    logger.info(
        f"Poisson reconstruction complete: {num_output} output vertices, "
        f"{len(mesh.faces)} faces, elapsed={elapsed:.1f} ms"
    )

    return ReconstructionResult(
        mesh=mesh,
        density=densities,
        elapsed_ms=elapsed,
        num_input_points=len(points),
        num_output_vertices=len(mesh.vertices),
    )


def _o3d_to_trimesh(mesh_o3d: Any) -> trimesh.Trimesh:
    """Convert Open3D mesh to trimesh."""
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
