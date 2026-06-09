"""Honeycomb grid generation in UV parameter space.

Generates a hexagonal lattice sized to fill the detected void.
Translates Section 3 of MuffinFresa_ConformalMapping.m.
"""

import numpy as np
from typing import Tuple


def create_hex_grid(nx: int, ny: int, hex_side: float) -> np.ndarray:
    """Create a hexagonal grid in UV space.

    Odd columns are shifted by ySpacing/2 for the honeycomb pattern.

    Args:
        nx: number of columns
        ny: number of rows
        hex_side: side length of each hexagon

    Returns:
        (ny, nx, 2) array of cell center positions in UV space
    """
    x_spacing = 1.5 * hex_side
    y_spacing = hex_side * np.sqrt(3)

    x_coords = np.arange(nx) * x_spacing
    y_coords = np.arange(ny) * y_spacing

    X, Y = np.meshgrid(x_coords, y_coords)
    # Shift odd columns
    Y[:, ::2] = Y[:, ::2] + y_spacing / 2.0

    grid = np.stack([X, Y], axis=-1)
    return grid


def hexagon_perimeter(center: np.ndarray, hex_side: float, n_per_edge: int = 20) -> np.ndarray:
    """Generate perimeter points for a flat-top hexagon.

    Args:
        center: (2,) center position [u, v]
        hex_side: radius to vertices
        n_per_edge: points per edge (excluding last vertex to avoid duplication)

    Returns:
        (6*n_per_edge, 2) ordered perimeter points
    """
    angles = np.arange(0, 360, 60) * np.pi / 180.0
    vertices = np.column_stack([
        center[0] + hex_side * np.cos(angles),
        center[1] + hex_side * np.sin(angles),
    ])
    # Close the polygon
    vertices_closed = np.vstack([vertices, vertices[0:1]])

    pts = []
    for i in range(6):
        x = np.linspace(vertices_closed[i, 0], vertices_closed[i + 1, 0], n_per_edge, endpoint=False)
        y = np.linspace(vertices_closed[i, 1], vertices_closed[i + 1, 1], n_per_edge, endpoint=False)
        pts.append(np.column_stack([x, y]))

    return np.vstack(pts)


def hex_fill_points(center: np.ndarray, hex_side: float, shrink: float = 0.85, n_grid: int = 12) -> np.ndarray:
    """Generate dense interior fill points for a hexagonal cell.

    Used for hydrogel injection volume visualization.

    Args:
        center: (2,) center position
        hex_side: hexagon radius
        shrink: fraction of hex_side for fill radius
        n_grid: grid resolution

    Returns:
        (M, 2) interior points
    """
    r = hex_side * shrink
    xx = np.linspace(center[0] - r, center[0] + r, n_grid)
    yy = np.linspace(center[1] - r * np.sqrt(3) / 2, center[1] + r * np.sqrt(3) / 2, n_grid)
    Xg, Yg = np.meshgrid(xx, yy)
    candidates = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Point-in-hexagon test
    angles = np.arange(0, 360, 60) * np.pi / 180.0
    hex_verts = np.column_stack([
        center[0] + hex_side * np.cos(angles),
        center[1] + hex_side * np.sin(angles),
    ])

    inside = _point_in_polygon(candidates, hex_verts)
    return candidates[inside]


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test.

    Args:
        points: (N, 2) query points
        polygon: (M, 2) vertices (closed automatically)

    Returns:
        (N,) boolean mask
    """
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)

    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        cond1 = (yi > points[:, 1]) != (yj > points[:, 1])
        slope = (xj - xi) / (yj - yi + 1e-15)
        x_intersect = xi + slope * (points[:, 1] - yi)
        cond2 = points[:, 0] < x_intersect

        inside ^= (cond1 & cond2)

    return inside


def compute_grid_params(void_width: float, void_length: float) -> Tuple[int, int, float]:
    """Compute honeycomb grid dimensions from void size.

    Returns:
        (nx, ny, hex_side)
    """
    hex_side = min(void_width, void_length) / 6.0
    nx = max(2, int(void_width / (hex_side * 1.5)))
    ny = max(2, int(void_length / (hex_side * np.sqrt(3))))
    return nx, ny, hex_side


def line_points(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Generate n linearly interpolated 3D points between start and end.

    Args:
        start: (3,) start position
        end: (3,) end position
        n: number of points

    Returns:
        (3, n) trajectory points (columns = points, matching MATLAB convention)
    """
    t = np.linspace(0, 1, n)
    traj = start[:, None] + (end - start)[:, None] * t[None, :]
    return traj
