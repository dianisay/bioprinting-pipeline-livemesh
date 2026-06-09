"""
Synthetic surface generators for testing reconstruction and toolpath algorithms
without requiring real depth camera data.

Surfaces are parameterized with known ground truth so we can compute exact
reconstruction error (Hausdorff distance, normal deviation).
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray


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
    cap_angle = np.radians(cap_angle_deg)
    theta = np.linspace(0, cap_angle, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    T, P = np.meshgrid(theta, phi)

    x = radius * np.sin(T) * np.cos(P)
    y = radius * np.sin(T) * np.sin(P)
    z = radius * np.cos(T)

    return _grid_to_mesh(x, y, z)


def saddle_surface(
    size: float = 40.0,
    curvature: float = 0.01,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Hyperbolic paraboloid (saddle) simulating concave-convex anatomy.

    z = curvature * (x^2 - y^2)
    """
    u = np.linspace(-size / 2, size / 2, resolution)
    v = np.linspace(-size / 2, size / 2, resolution)
    U, V = np.meshgrid(u, v)
    Z = curvature * (U**2 - V**2)
    return _grid_to_mesh(U, V, Z)


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
    sigma = inner_radius / 2.5
    u = np.linspace(-outer_radius, outer_radius, resolution)
    v = np.linspace(-outer_radius, outer_radius, resolution)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)
    Z = -depth * np.exp(-(R**2) / (2 * sigma**2))
    mask = R <= outer_radius
    Z[~mask] = np.nan
    return _grid_to_mesh(U, V, Z, drop_nan=True)


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
    arc = np.radians(arc_angle_deg)
    theta = np.linspace(-arc / 2, arc / 2, resolution)
    x = np.linspace(-axial_length / 2, axial_length / 2, resolution)
    T, X = np.meshgrid(theta, x)

    Y = radius * np.sin(T)
    Z = radius * np.cos(T)

    return _grid_to_mesh(X, Y, Z)


def flat_plane(
    size: float = 60.0,
    resolution: int = 64,
) -> trimesh.Trimesh:
    """Flat plane baseline for comparison."""
    u = np.linspace(-size / 2, size / 2, resolution)
    v = np.linspace(-size / 2, size / 2, resolution)
    U, V = np.meshgrid(u, v)
    Z = np.zeros_like(U)
    return _grid_to_mesh(U, V, Z)


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
    if rng is None:
        rng = np.random.default_rng(42)
    points = np.array(mesh.vertices, dtype=np.float64)
    noise = rng.normal(0, sigma, size=points.shape)
    return points + noise


def add_occlusion(
    points: NDArray[np.float64],
    fraction: float = 0.2,
    center: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Remove a fraction of points near a center to simulate nozzle occlusion.

    If center is None, uses the centroid.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if center is None:
        center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    threshold = np.quantile(dists, fraction)
    return points[dists > threshold]


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
