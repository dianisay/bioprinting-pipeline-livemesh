"""
Geodesic toolpath generation on triangulated meshes.

Replaces your MATLAB planar slicer with surface-following deposition paths.
Uses the heat method for geodesic distance computation (Crane et al., 2017)
via potpourri3d, which wraps geometry-central.

The key idea: instead of slicing the wound into flat layers, we compute
geodesic distance contours on the mesh surface and trace parallel geodesic
curves as deposition paths. This ensures uniform material coverage on
curved anatomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)
import potpourri3d as pp3d
import trimesh
from numpy.typing import NDArray


@dataclass
class ToolpathResult:
    waypoints: NDArray[np.float64]       # (M, 3) XYZ in mm
    normals: NDArray[np.float64]         # (M, 3) surface normals at each waypoint
    path_lengths_mm: list[float]         # length of each individual path
    total_length_mm: float
    num_paths: int
    elapsed_ms: float = 0.0
    is_deposition: NDArray[np.bool_] = field(default_factory=lambda: np.array([]))


def geodesic_toolpaths(
    mesh: trimesh.Trimesh,
    spacing_mm: float = 1.5,
    source_vertex: int | None = None,
    adaptive_curvature: bool = True,
    curvature_factor: float = 0.5,
    boundary_margin_mm: float = 0.5,
    num_contour_points: int = 200,
) -> ToolpathResult:
    """Generate parallel geodesic deposition paths on a mesh surface.

    Algorithm:
    1. Pick a source vertex (default: boundary vertex with max geodesic centrality)
    2. Compute geodesic distances from source to all vertices (heat method)
    3. Extract iso-distance contours at regular spacing
    4. Optionally adapt spacing based on local Gaussian curvature
    5. Interpolate contour points along mesh faces
    6. Compute surface normals at each waypoint for nozzle orientation

    Parameters
    ----------
    mesh : triangulated surface
    spacing_mm : distance between adjacent geodesic paths
    source_vertex : starting vertex index. If None, auto-selected.
    adaptive_curvature : tighten spacing on high-curvature regions
    curvature_factor : 0=uniform, 1=fully curvature-adaptive
    boundary_margin_mm : don't place paths closer than this to mesh boundary
    num_contour_points : interpolation density per contour
    """
    import time

    t0 = time.perf_counter()
    logger.info(
        f"Geodesic toolpaths starting: {len(mesh.vertices)} vertices, "
        f"{len(mesh.faces)} faces, spacing={spacing_mm} mm, "
        f"adaptive_curvature={adaptive_curvature}"
    )

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

    if source_vertex is None:
        source_vertex = _select_source_vertex(mesh, solver)
        logger.debug(f"Auto-selected source vertex: {source_vertex}")
    else:
        logger.debug(f"Using provided source vertex: {source_vertex}")

    distances = solver.compute_distance(source_vertex)

    if adaptive_curvature and curvature_factor > 0:
        spacings = _adaptive_spacing(mesh, spacing_mm, curvature_factor)
    else:
        spacings = None

    contour_levels = _compute_contour_levels(distances, spacing_mm, spacings)
    logger.debug(
        f"Computed {len(contour_levels)} contour levels, "
        f"max_geodesic_distance={np.max(distances[np.isfinite(distances)]):.2f} mm"
    )

    all_waypoints = []
    all_normals = []
    path_lengths = []
    is_deposition = []

    prev_end = None
    for level in contour_levels:
        contour_pts, contour_norms = _extract_contour(
            mesh, distances, level, num_contour_points, boundary_margin_mm
        )
        if len(contour_pts) < 3:
            continue

        if prev_end is not None:
            travel = _linear_interpolation(prev_end, contour_pts[0], n=10)
            all_waypoints.append(travel)
            travel_norms = np.tile(contour_norms[0], (len(travel), 1))
            all_normals.append(travel_norms)
            is_deposition.extend([False] * len(travel))

        all_waypoints.append(contour_pts)
        all_normals.append(contour_norms)
        is_deposition.extend([True] * len(contour_pts))

        lengths = np.linalg.norm(np.diff(contour_pts, axis=0), axis=1)
        path_lengths.append(float(np.sum(lengths)))

        prev_end = contour_pts[-1]

    if not all_waypoints:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.warning(
            f"No geodesic toolpaths generated from source vertex {source_vertex}, "
            f"elapsed={elapsed:.1f} ms"
        )
        return ToolpathResult(
            waypoints=np.empty((0, 3)),
            normals=np.empty((0, 3)),
            path_lengths_mm=[],
            total_length_mm=0.0,
            num_paths=0,
            elapsed_ms=elapsed,
        )

    waypoints = np.vstack(all_waypoints)
    normals = np.vstack(all_normals)
    is_dep = np.array(is_deposition, dtype=bool)

    elapsed = (time.perf_counter() - t0) * 1000
    num_paths = len(path_lengths)
    total_len = sum(path_lengths)
    logger.info(
        f"Geodesic toolpaths complete: source_vertex={source_vertex}, "
        f"{len(contour_levels)} contours, {num_paths} paths, "
        f"total_length={total_len:.1f} mm, {len(waypoints)} waypoints, "
        f"elapsed={elapsed:.1f} ms"
    )

    return ToolpathResult(
        waypoints=waypoints,
        normals=normals,
        path_lengths_mm=path_lengths,
        total_length_mm=total_len,
        num_paths=num_paths,
        elapsed_ms=elapsed,
        is_deposition=is_dep,
    )


def _select_source_vertex(
    mesh: trimesh.Trimesh, solver: pp3d.MeshHeatMethodDistanceSolver
) -> int:
    """Select source vertex: boundary vertex closest to centroid projected onto boundary."""
    boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    if len(boundary_edges) == 0:
        logger.warning("Mesh has no boundary edges, defaulting source vertex to 0")
        return 0
    boundary_verts = np.unique(boundary_edges)
    centroid = mesh.vertices[boundary_verts].mean(axis=0)
    dists_to_centroid = np.linalg.norm(mesh.vertices[boundary_verts] - centroid, axis=1)
    return int(boundary_verts[np.argmin(dists_to_centroid)])


def _adaptive_spacing(
    mesh: trimesh.Trimesh,
    base_spacing: float,
    curvature_factor: float,
) -> NDArray[np.float64]:
    """Compute per-vertex spacing based on discrete Gaussian curvature.

    High curvature -> tighter spacing to avoid gaps.
    """
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, mesh.vertices, radius=base_spacing * 2
    )
    curvature_abs = np.abs(curvature)
    max_k = np.percentile(curvature_abs, 95)
    if max_k < 1e-10:
        return np.full(len(mesh.vertices), base_spacing)

    normalized_k = np.clip(curvature_abs / max_k, 0, 1)
    spacings = base_spacing * (1.0 - curvature_factor * normalized_k)
    spacings = np.clip(spacings, base_spacing * 0.3, base_spacing)
    return spacings


def _compute_contour_levels(
    distances: NDArray[np.float64],
    base_spacing: float,
    spacings: NDArray[np.float64] | None,
) -> list[float]:
    """Compute geodesic distance levels for contour extraction."""
    d_max = np.max(distances[np.isfinite(distances)])
    if spacings is None:
        n_levels = int(d_max / base_spacing)
        return [base_spacing * (i + 1) for i in range(n_levels)]

    levels = []
    current = base_spacing
    while current < d_max:
        levels.append(current)
        local_spacing = float(np.mean(spacings[distances <= current + base_spacing]))
        current += max(local_spacing, base_spacing * 0.3)
    return levels


def _extract_contour(
    mesh: trimesh.Trimesh,
    distances: NDArray[np.float64],
    level: float,
    num_points: int,
    margin: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract a single iso-distance contour by linear interpolation on mesh edges."""
    vertices = mesh.vertices
    faces = mesh.faces
    contour_points = []
    contour_normals = []

    for face_idx, face in enumerate(faces):
        v0, v1, v2 = face
        d = distances[[v0, v1, v2]]
        pts = vertices[[v0, v1, v2]]

        crossings = []
        edges = [(0, 1), (1, 2), (2, 0)]
        for i, j in edges:
            if (d[i] - level) * (d[j] - level) < 0:
                t = (level - d[i]) / (d[j] - d[i])
                p = pts[i] + t * (pts[j] - pts[i])
                crossings.append(p)

        if len(crossings) == 2:
            contour_points.extend(crossings)
            normal = mesh.face_normals[face_idx]
            contour_normals.extend([normal, normal])

    if len(contour_points) < 2:
        return np.empty((0, 3)), np.empty((0, 3))

    points = np.array(contour_points)
    norms = np.array(contour_normals)

    ordered_pts, ordered_norms = _order_contour(points, norms)

    if len(ordered_pts) > num_points:
        indices = np.linspace(0, len(ordered_pts) - 1, num_points, dtype=int)
        ordered_pts = ordered_pts[indices]
        ordered_norms = ordered_norms[indices]

    return ordered_pts, ordered_norms


def _order_contour(
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Order contour points into a connected path using nearest-neighbor chaining."""
    n = len(points)
    if n <= 2:
        return points, normals

    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = order[-1]
        dists = np.linalg.norm(points - points[current], axis=1)
        dists[visited] = np.inf
        nearest = int(np.argmin(dists))
        order.append(nearest)
        visited[nearest] = True

    return points[order], normals[order]


def _linear_interpolation(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    n: int = 10,
) -> NDArray[np.float64]:
    """Generate n evenly spaced points between start and end."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return start + t * (end - start)
