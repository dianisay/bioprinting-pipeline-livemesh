"""
Coverage uniformity metrics for comparing toolpath strategies.

Measures how uniformly material is distributed over the wound surface.
Uses optimal-transport-based metrics following Solomon et al. (SIGGRAPH 2015).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)
import trimesh
from numpy.typing import NDArray


@dataclass
class CoverageResult:
    coverage_fraction: float       # 0-1, fraction of surface within reach of a path
    uniformity_score: float        # 0-1, 1 = perfectly uniform
    mean_distance_mm: float        # average distance from surface to nearest path
    max_gap_mm: float              # worst-case gap
    wasserstein_distance: float    # OT-based distribution distance (lower = better)


def coverage_uniformity(
    mesh: trimesh.Trimesh,
    waypoints: NDArray[np.float64],
    nozzle_width_mm: float = 1.0,
    num_samples: int = 5000,
) -> CoverageResult:
    """Evaluate how uniformly a toolpath covers a mesh surface.

    Samples points on the mesh surface, computes distance to nearest
    waypoint, and evaluates coverage statistics.

    Parameters
    ----------
    mesh : target surface
    waypoints : (M, 3) deposition path points
    nozzle_width_mm : material deposition width (reach radius)
    num_samples : surface sampling density
    """
    logger.info(
        f"Coverage analysis starting: {len(waypoints)} waypoints, "
        f"nozzle_width={nozzle_width_mm} mm, {num_samples} surface samples"
    )
    surface_pts = trimesh.sample.sample_surface(mesh, num_samples)[0]

    from scipy.spatial import KDTree

    tree = KDTree(waypoints)
    distances, _ = tree.query(surface_pts)

    covered = distances <= nozzle_width_mm
    coverage_fraction = float(np.mean(covered))

    mean_dist = float(np.mean(distances))
    max_gap = float(np.max(distances))

    std_dist = float(np.std(distances))
    uniformity = 1.0 - min(std_dist / (mean_dist + 1e-10), 1.0)

    w_dist = _wasserstein_1d_approx(distances)

    logger.info(
        f"Coverage metrics: fraction={coverage_fraction:.1%}, "
        f"uniformity={uniformity:.3f}, mean_distance={mean_dist:.2f} mm, "
        f"max_gap={max_gap:.2f} mm, wasserstein={w_dist:.4f}"
    )
    if coverage_fraction < 0.9:
        logger.warning(
            f"Low surface coverage: {coverage_fraction:.1%} "
            f"(max_gap={max_gap:.2f} mm)"
        )

    return CoverageResult(
        coverage_fraction=coverage_fraction,
        uniformity_score=uniformity,
        mean_distance_mm=mean_dist,
        max_gap_mm=max_gap,
        wasserstein_distance=w_dist,
    )


def _wasserstein_1d_approx(distances: NDArray[np.float64]) -> float:
    """1D Wasserstein distance between the observed distance distribution
    and the ideal (uniform at zero distance).

    This is a simplified version; for full 2D Wasserstein on the surface,
    use the POT library with the mesh geodesic cost matrix.
    """
    sorted_d = np.sort(distances)
    n = len(sorted_d)
    ideal = np.zeros(n)
    return float(np.mean(np.abs(sorted_d - ideal)))


def coverage_comparison_table(
    mesh: trimesh.Trimesh,
    toolpaths: dict[str, NDArray[np.float64]],
    nozzle_width_mm: float = 1.0,
) -> dict[str, CoverageResult]:
    """Run coverage analysis on multiple toolpath strategies and return a comparison.

    Usage:
        results = coverage_comparison_table(mesh, {
            "planar": planar_waypoints,
            "geodesic_uniform": geodesic_waypoints,
            "geodesic_adaptive": adaptive_waypoints,
        })
    """
    logger.info(f"Comparing coverage across {len(toolpaths)} toolpath strategies")
    results = {
        name: coverage_uniformity(mesh, waypoints, nozzle_width_mm)
        for name, waypoints in toolpaths.items()
    }
    for name, res in results.items():
        logger.info(
            f"  [{name}] coverage={res.coverage_fraction:.1%}, "
            f"uniformity={res.uniformity_score:.3f}, max_gap={res.max_gap_mm:.2f} mm"
        )
    return results
