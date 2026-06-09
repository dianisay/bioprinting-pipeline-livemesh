"""
Conformal UV-to-XYZ mapping for cylindrical and general surfaces.

Ported from your MATLAB MuffinFresa_ConformalMapping.m.
Maps flat UV toolpaths onto curved surfaces while preserving local orientation
via surface normals, ensuring the nozzle stays orthogonal to the tissue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ConformalMapResult:
    xyz_mm: NDArray[np.float64]       # (N, 3) mapped positions
    normals: NDArray[np.float64]      # (N, 3) outward surface normals
    tool_frames: NDArray[np.float64]  # (N, 3, 3) rotation matrices [x_tool, y_tool, z_tool]


def cylinder_conformal_map(
    traj_uv: NDArray[np.float64],
    cyl_radius: float,
    cyl_center_y: float = 0.0,
    cyl_center_z: float = 0.0,
) -> ConformalMapResult:
    """Map UV+h trajectories onto a cylindrical surface.

    Direct port of your MATLAB conformal mapping.
    Cylinder axis = X, curvature in YZ plane.

    Parameters
    ----------
    traj_uv : (N, 3) array with columns [u_arc_mm, v_axial_mm, h_normal_mm]
              u = arc-length coordinate (theta * R)
              v = axial coordinate (along X)
              h = height above surface (>0 travel, <=0 depositing into surface)
    cyl_radius : cylinder radius in mm
    cyl_center_y, cyl_center_z : cylinder axis center in YZ
    """
    u = traj_uv[:, 0]
    v = traj_uv[:, 1]
    h = traj_uv[:, 2]

    theta = u / cyl_radius

    x = v
    y = cyl_center_y + cyl_radius * np.sin(theta)
    z = cyl_center_z + cyl_radius * np.cos(theta)

    # Outward radial normal in YZ plane
    ny = np.sin(theta)
    nz = np.cos(theta)
    nx = np.zeros_like(ny)
    normals = np.column_stack([nx, ny, nz])

    # TCP = surface point + h * normal
    xyz = np.column_stack([x, y, z]) + h[:, np.newaxis] * normals

    # Tool frame: Z_tool = -normal (into surface), X_tool from cross with world X
    tool_frames = np.zeros((len(traj_uv), 3, 3))
    world_x = np.array([1.0, 0.0, 0.0])

    for i in range(len(traj_uv)):
        z_tool = -normals[i]
        x_tool = np.cross(world_x, z_tool)
        x_norm = np.linalg.norm(x_tool)
        if x_norm < 1e-6:
            x_tool = np.cross(np.array([0.0, 1.0, 0.0]), z_tool)
            x_norm = np.linalg.norm(x_tool)
        x_tool /= x_norm
        y_tool = np.cross(z_tool, x_tool)
        tool_frames[i] = np.column_stack([x_tool, y_tool, z_tool])

    return ConformalMapResult(xyz_mm=xyz, normals=normals, tool_frames=tool_frames)


def general_conformal_map(
    traj_uv: NDArray[np.float64],
    mesh_vertices: NDArray[np.float64],
    mesh_faces: NDArray[np.int32],
    mesh_normals: NDArray[np.float64],
    uv_coords: NDArray[np.float64],
) -> ConformalMapResult:
    """Map UV toolpaths onto a general triangulated mesh using barycentric interpolation.

    For Solomon's project, this extends the cylinder-specific map to arbitrary wound geometry.

    Parameters
    ----------
    traj_uv : (N, 3) with columns [u, v, h]
    mesh_vertices : (V, 3) vertex positions
    mesh_faces : (F, 3) triangle indices
    mesh_normals : (F, 3) per-face normals
    uv_coords : (V, 2) UV parameterization of the mesh
    """
    from scipy.spatial import Delaunay

    uv_tri = Delaunay(uv_coords)

    uv_query = traj_uv[:, :2]
    h = traj_uv[:, 2]

    simplex_indices = uv_tri.find_simplex(uv_query)

    xyz_list = []
    normal_list = []
    frames_list = []

    for i in range(len(traj_uv)):
        si = simplex_indices[i]
        if si == -1:
            _, si = _nearest_simplex(uv_tri, uv_query[i])

        tri_verts_uv = uv_coords[uv_tri.simplices[si]]
        bary = _barycentric(uv_query[i], tri_verts_uv)

        mesh_face = mesh_faces[si % len(mesh_faces)]
        tri_verts_3d = mesh_vertices[mesh_face]
        point_3d = bary @ tri_verts_3d
        normal = mesh_normals[si % len(mesh_normals)]
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        tcp = point_3d + h[i] * normal
        xyz_list.append(tcp)
        normal_list.append(normal)

        z_tool = -normal
        x_tool = np.cross(np.array([1.0, 0.0, 0.0]), z_tool)
        x_norm = np.linalg.norm(x_tool)
        if x_norm < 1e-6:
            x_tool = np.cross(np.array([0.0, 1.0, 0.0]), z_tool)
            x_norm = np.linalg.norm(x_tool)
        x_tool /= x_norm
        y_tool = np.cross(z_tool, x_tool)
        frames_list.append(np.column_stack([x_tool, y_tool, z_tool]))

    return ConformalMapResult(
        xyz_mm=np.array(xyz_list),
        normals=np.array(normal_list),
        tool_frames=np.array(frames_list),
    )


def _barycentric(point: NDArray, triangle: NDArray) -> NDArray:
    """Compute barycentric coordinates of a 2D point in a triangle."""
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1 / 3, 1 / 3, 1 / 3])

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def _nearest_simplex(tri, point):
    """Find the nearest simplex to an out-of-hull point."""
    centroids = np.mean(tri.points[tri.simplices], axis=1)
    dists = np.linalg.norm(centroids - point, axis=1)
    idx = np.argmin(dists)
    return dists[idx], idx


def uv_to_xyz(
    traj_uv: NDArray[np.float64],
    cyl_radius: float,
    cyl_cy: float,
    cyl_cz: float,
    u_offset: float,
    v_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map (3, N) UV trajectory to XYZ on a cylinder with UV offsets."""
    u = traj_uv[0] + u_offset
    v = traj_uv[1] + v_offset
    h = traj_uv[2]
    result = cylinder_conformal_map(
        np.column_stack([u, v, h]), cyl_radius, cyl_cy, cyl_cz
    )
    return result.xyz_mm.T, result.normals.T


def compute_nozzle_orientations(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute tool rotation matrices from (3, N) surface normals."""
    n = normals.T
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-10)
    R_targets = np.zeros((len(n), 3, 3))
    world_x = np.array([1.0, 0.0, 0.0])

    for i in range(len(n)):
        z_tool = -n[i]
        x_tool = np.cross(world_x, z_tool)
        x_norm = np.linalg.norm(x_tool)
        if x_norm < 1e-6:
            x_tool = np.cross(np.array([0.0, 1.0, 0.0]), z_tool)
            x_norm = np.linalg.norm(x_tool)
        x_tool /= x_norm
        y_tool = np.cross(z_tool, x_tool)
        R_targets[i] = np.column_stack([x_tool, y_tool, z_tool])

    return R_targets


def apply_workspace_transform(
    traj_xyz: NDArray[np.float64],
    normals: NDArray[np.float64],
    z_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scale mm→m, apply Rx(90°) base rotation, and offset Z for robot workspace."""
    pts = traj_xyz.T * 0.001
    nrm = normals.T.copy()
    Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pts = pts @ Rx90.T
    nrm = nrm @ Rx90.T
    pts[:, 2] += z_offset
    return pts.T, nrm.T
