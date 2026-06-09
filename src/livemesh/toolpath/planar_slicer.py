"""
Planar slicing baseline -- the method your current system uses.

We keep this so we always have a direct comparison: geodesic vs planar.
Ported from your MATLAB G-code generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray


@dataclass
class PlanarSliceResult:
    waypoints: NDArray[np.float64]   # (M, 3) XYZ in mm
    layer_heights: list[float]
    total_length_mm: float
    num_layers: int


def planar_slice(
    mesh: trimesh.Trimesh,
    layer_height_mm: float = 0.4,
    line_spacing_mm: float = 1.5,
    direction: str = "zigzag",
) -> PlanarSliceResult:
    """Slice a mesh into flat layers with zigzag infill.

    This is the baseline that fails on curved surfaces: it treats
    every layer as a flat plane, ignoring surface curvature.

    Parameters
    ----------
    mesh : surface mesh
    layer_height_mm : Z distance between layers
    line_spacing_mm : spacing between infill lines within a layer
    direction : "zigzag" or "contour"
    """
    bounds = mesh.bounds
    z_min, z_max = bounds[0][2], bounds[1][2]

    heights = np.arange(z_min + layer_height_mm, z_max, layer_height_mm)

    all_waypoints = []
    for i, z in enumerate(heights):
        try:
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section is None:
                continue
        except Exception:
            continue

        path_2d, transform = section.to_planar()

        if direction == "zigzag":
            layer_pts = _zigzag_infill(path_2d, line_spacing_mm, z, transform, flip=(i % 2 == 1))
        else:
            layer_pts = _contour_infill(path_2d, z, transform)

        if len(layer_pts) > 0:
            all_waypoints.append(layer_pts)

    if not all_waypoints:
        return PlanarSliceResult(
            waypoints=np.empty((0, 3)),
            layer_heights=[],
            total_length_mm=0.0,
            num_layers=0,
        )

    waypoints = np.vstack(all_waypoints)
    total_len = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))

    return PlanarSliceResult(
        waypoints=waypoints,
        layer_heights=list(heights),
        total_length_mm=total_len,
        num_layers=len(heights),
    )


def _zigzag_infill(
    path_2d: trimesh.path.Path2D,
    spacing: float,
    z: float,
    transform: NDArray,
    flip: bool = False,
) -> NDArray[np.float64]:
    """Generate zigzag raster lines within a 2D cross-section."""
    bounds = path_2d.bounds
    if bounds is None:
        return np.empty((0, 3))

    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]

    lines_y = np.arange(y_min + spacing / 2, y_max, spacing)
    points_2d = []

    for i, y in enumerate(lines_y):
        x_start, x_end = x_min, x_max
        if (i % 2 == 1) != flip:
            x_start, x_end = x_end, x_start

        n_pts = max(int(abs(x_end - x_start) / (spacing / 4)), 2)
        xs = np.linspace(x_start, x_end, n_pts)
        for x in xs:
            if path_2d.contains_points([[x, y]])[0]:
                points_2d.append([x, y])

    if not points_2d:
        return np.empty((0, 3))

    pts_2d = np.array(points_2d)
    pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), z)])

    inv_transform = np.linalg.inv(transform)
    pts_3d_homo = np.column_stack([pts_3d, np.ones(len(pts_3d))])
    pts_3d_world = (inv_transform @ pts_3d_homo.T).T[:, :3]

    return pts_3d_world


def _contour_infill(
    path_2d: trimesh.path.Path2D,
    z: float,
    transform: NDArray,
) -> NDArray[np.float64]:
    """Follow the cross-section contour (like your MATLAB bwboundaries approach)."""
    try:
        polygons = path_2d.polygons_full
        if not polygons:
            return np.empty((0, 3))
    except Exception:
        return np.empty((0, 3))

    pts_2d = np.array(polygons[0].exterior.coords)
    pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), z)])

    inv_transform = np.linalg.inv(transform)
    pts_3d_homo = np.column_stack([pts_3d, np.ones(len(pts_3d))])
    pts_3d_world = (inv_transform @ pts_3d_homo.T).T[:, :3]

    return pts_3d_world
