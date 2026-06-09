"""
Synthetic surface generators for testing reconstruction and toolpath algorithms
without requiring real depth camera data.

Surfaces are parameterized with known ground truth so we can compute exact
reconstruction error (Hausdorff distance, normal deviation).
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def sphere_cap(
    radius: float = 50.0,
    cap_angle_deg: float = 60.0,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Spherical cap simulating a convex wound site (e.g., scalp, shoulder).

    Parameters
    ----------
    radius : mm
    cap_angle_deg : angular extent from pole
    resolution : grid points per axis
    """
    logger.info(
        f"Generating sphere_cap surface: radius={radius:.1f} mm, "
        f"cap_angle={cap_angle_deg:.1f} deg, resolution={resolution}"
    )
    cap_angle = np.radians(cap_angle_deg)
    theta = np.linspace(0, cap_angle, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    T, P = np.meshgrid(theta, phi)

    x = radius * np.sin(T) * np.cos(P)
    y = radius * np.sin(T) * np.sin(P)
    z = radius * np.cos(T)

    mesh = _grid_to_mesh(x, y, z)
    logger.info(f"sphere_cap complete: {len(mesh.vertices)} vertices")
    return mesh


def saddle_surface(
    size: float = 40.0,
    curvature: float = 0.01,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Hyperbolic paraboloid (saddle) simulating concave-convex anatomy.

    z = curvature * (x^2 - y^2)
    """
    logger.info(
        f"Generating saddle_surface: size={size:.1f} mm, "
        f"curvature={curvature}, resolution={resolution}"
    )
    u = np.linspace(-size / 2, size / 2, resolution)
    v = np.linspace(-size / 2, size / 2, resolution)
    U, V = np.meshgrid(u, v)
    Z = curvature * (U**2 - V**2)
    mesh = _grid_to_mesh(U, V, Z)
    logger.info(f"saddle_surface complete: {len(mesh.vertices)} vertices")
    return mesh


def wound_crater(
    outer_radius: float = 40.0,
    inner_radius: float = 25.0,
    depth: float = 8.0,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Concave crater simulating a deep wound with raised edges.

    Gaussian profile: z = -depth * exp(-r^2 / (2*sigma^2))
    surrounded by a flat annulus.
    """
    logger.info(
        f"Generating wound_crater: outer_r={outer_radius:.1f} mm, "
        f"inner_r={inner_radius:.1f} mm, depth={depth:.1f} mm"
    )
    sigma = inner_radius / 2.5
    u = np.linspace(-outer_radius, outer_radius, resolution)
    v = np.linspace(-outer_radius, outer_radius, resolution)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)
    Z = -depth * np.exp(-(R**2) / (2 * sigma**2))
    mask = R <= outer_radius
    Z[~mask] = np.nan
    mesh = _grid_to_mesh(U, V, Z, drop_nan=True)
    logger.info(f"wound_crater complete: {len(mesh.vertices)} vertices")
    return mesh


def cylinder_patch(
    radius: float = 50.0,
    arc_angle_deg: float = 90.0,
    axial_length: float = 80.0,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Cylindrical patch matching your Conformal-Trajectory scaffold geometry.

    Parameters
    ----------
    radius : cylinder radius in mm (your STL used ~48.6 mm)
    arc_angle_deg : angular extent of the patch
    axial_length : extent along cylinder axis (X)
    """
    logger.info(
        f"Generating cylinder_patch: radius={radius:.1f} mm, "
        f"arc={arc_angle_deg:.1f} deg, axial_length={axial_length:.1f} mm"
    )
    arc = np.radians(arc_angle_deg)
    theta = np.linspace(-arc / 2, arc / 2, resolution)
    x = np.linspace(-axial_length / 2, axial_length / 2, resolution)
    T, X = np.meshgrid(theta, x)

    Y = radius * np.sin(T)
    Z = radius * np.cos(T)

    mesh = _grid_to_mesh(X, Y, Z)
    logger.info(f"cylinder_patch complete: {len(mesh.vertices)} vertices")
    return mesh


def flat_plane(
    size: float = 60.0,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Flat plane baseline for comparison."""
    logger.info(f"Generating flat_plane: size={size:.1f} mm, resolution={resolution}")
    u = np.linspace(-size / 2, size / 2, resolution)
    v = np.linspace(-size / 2, size / 2, resolution)
    U, V = np.meshgrid(u, v)
    Z = np.zeros_like(U)
    mesh = _grid_to_mesh(U, V, Z)
    logger.info(f"flat_plane complete: {len(mesh.vertices)} vertices")
    return mesh


def add_noise(
    mesh: trimesh.Trimesh,
    sigma: float = 0.5,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Add Gaussian noise to mesh vertices, returning a noisy point cloud.

    This simulates the output of a depth camera: the ground-truth mesh is known,
    and the noisy point cloud is what the reconstruction module receives.

    Returns (N, 3) point cloud in mm.
    """
    logger.info(
        f"Adding noise to mesh: {len(mesh.vertices)} vertices, sigma={sigma:.3f} mm"
    )
    if rng is None:
        rng = np.random.default_rng(42)
    points = np.array(mesh.vertices, dtype=np.float64)
    noise = rng.normal(0, sigma, size=points.shape)
    noisy = points + noise
    logger.debug(f"Noisy point cloud: {len(noisy)} points, sigma={sigma:.3f} mm")
    return noisy


def add_occlusion(
    points: NDArray[np.float64],
    fraction: float = 0.2,
    center: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Remove a fraction of points near a center to simulate nozzle occlusion.

    If center is None, uses the centroid.
    """
    logger.info(
        f"Adding occlusion: {len(points)} points, fraction={fraction:.1%}"
    )
    if rng is None:
        rng = np.random.default_rng(42)
    if center is None:
        center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    threshold = np.quantile(dists, fraction)
    remaining = points[dists > threshold]
    logger.info(
        f"Occlusion complete: {len(remaining)}/{len(points)} points retained "
        f"({100 * len(remaining) / len(points):.1f}%)"
    )
    return remaining


def _grid_to_mesh(
    X: NDArray, Y: NDArray, Z: NDArray, drop_nan: bool = False
) -> trimesh.Trimesh:
    """Convert a parameter grid to a triangulated mesh."""
    rows, cols = X.shape
    vertices = []
    index_map = np.full((rows, cols), -1, dtype=int)

    for i in range(rows):
        for j in range(cols):
            if drop_nan and np.isnan(Z[i, j]):
                continue
            index_map[i, j] = len(vertices)
            vertices.append([X[i, j], Y[i, j], Z[i, j]])

    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = [index_map[i, j], index_map[i, j + 1],
                   index_map[i + 1, j], index_map[i + 1, j + 1]]
            if -1 in idx:
                continue
            faces.append([idx[0], idx[2], idx[1]])
            faces.append([idx[1], idx[2], idx[3]])

    return trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        process=True,
    )
