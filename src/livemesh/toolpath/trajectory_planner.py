"""Full trajectory planner: generates outline, fill, and deposit trajectories.

Combines honeycomb grid, TSP ordering, and UV trajectory generation.
Translates Section 4 of MuffinFresa_ConformalMapping.m.
"""

import logging

import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from .honeycomb import create_hex_grid, hexagon_perimeter, compute_grid_params, line_points
from .tsp_solver import optimize_visitation_order
from .conformal_map import uv_to_xyz, compute_nozzle_orientations, apply_workspace_transform


def generate_uv_trajectories(
    grid: np.ndarray,
    cell_indices: np.ndarray,
    hex_side: float,
    rise: float = 20.0,
    layer_height: float = 0.4,
    wall_height: float = 4.0,
    n_interp: int = 20,
) -> Dict[str, np.ndarray]:
    """Generate outline, fill, and deposit trajectories in UV space.

    Args:
        grid: (ny, nx, 2) hexagonal grid
        cell_indices: (K, 2) ordered cell indices (0-based)
        hex_side: hexagon side length (mm)
        rise: travel altitude above surface (mm)
        layer_height: deposition layer height (mm)
        wall_height: total wall height = shell thickness (mm)
        n_interp: interpolation points for linear moves

    Returns:
        dict with 'outline', 'fill', 'deposit' as (3, N) UV trajectories,
        and 'combined' as the concatenated full trajectory
    """
    num_layers = int(np.ceil(wall_height / layer_height))
    logger.info(
        f"UV trajectory generation: {len(cell_indices)} cells, "
        f"hex_side={hex_side:.2f} mm, {num_layers} layers, rise={rise:.1f} mm"
    )

    def _trace_cells(indices):
        """Generate trajectory for a set of cells (outline or fill)."""
        traj_list = []
        pos = np.array([0.0, 0.0, rise])

        for ix, iy in indices:
            center = grid[iy, ix]
            pts_uv = hexagon_perimeter(center, hex_side)

            # Travel to first vertex at rise altitude
            target = np.array([pts_uv[0, 0], pts_uv[0, 1], rise])
            traj_list.append(line_points(pos, target, n_interp))
            pos = target.copy()

            # Lower to surface
            target[2] = 0.0
            traj_list.append(line_points(pos, target, n_interp))
            pos = target.copy()

            # Trace perimeter for each layer
            for layer in range(num_layers):
                h_layer = -(layer * layer_height)
                hex_3d = np.column_stack([pts_uv, np.full(len(pts_uv), h_layer)]).T
                traj_list.append(hex_3d)
                pos = hex_3d[:, -1].copy()

            # Rise back
            target = np.array([pts_uv[0, 0], pts_uv[0, 1], rise])
            traj_list.append(line_points(pos, target, n_interp))
            pos = target.copy()

        return np.hstack(traj_list) if traj_list else np.empty((3, 0))

    outline_traj = _trace_cells(cell_indices)
    fill_traj = _trace_cells(cell_indices)

    # Deposit trajectory (vertical fill at cell centers)
    all_indices = np.vstack([cell_indices, cell_indices])
    deposit_list = []
    pos = np.array([0.0, 0.0, rise])

    for ix, iy in all_indices:
        center = grid[iy, ix]

        target = np.array([center[0], center[1], rise])
        deposit_list.append(line_points(pos, target, n_interp))
        pos = target.copy()

        target[2] = 0.0
        deposit_list.append(line_points(pos, target, n_interp))
        pos = target.copy()

        target[2] = -wall_height
        deposit_list.append(line_points(pos, target, n_interp))
        pos = target.copy()

        target[2] = rise
        deposit_list.append(line_points(pos, target, n_interp))
        pos = target.copy()

    deposit_traj = np.hstack(deposit_list) if deposit_list else np.empty((3, 0))

    combined = np.hstack([outline_traj, fill_traj, deposit_traj])
    logger.info(
        f"UV trajectories complete: outline={outline_traj.shape[1]} pts, "
        f"fill={fill_traj.shape[1]} pts, deposit={deposit_traj.shape[1]} pts, "
        f"combined={combined.shape[1]} pts"
    )

    return {
        "outline": outline_traj,
        "fill": fill_traj,
        "deposit": deposit_traj,
        "combined": combined,
    }


def plan_full_trajectory(
    void_bounds: Dict,
    cyl_radius: float,
    cyl_cy: float,
    cyl_cz: float,
    rise: float = 20.0,
    layer_height: float = 0.4,
    z_offset: float = -0.35,
    optimize_tsp: bool = True,
) -> Dict:
    """Complete trajectory planning pipeline.

    1. Compute honeycomb grid parameters from void bounds
    2. Generate hexagonal grid
    3. Optimize cell visitation order (TSP)
    4. Generate UV trajectories (outline + fill + deposit)
    5. Map UV → XYZ on cylinder surface
    6. Compute nozzle orientations
    7. Apply workspace transform (mm→m, base rotation)

    Args:
        void_bounds: dict from stl_analysis.compute_void_bounds
        cyl_radius, cyl_cy, cyl_cz: cylinder parameters
        rise: travel altitude (mm)
        layer_height: deposition layer height (mm)
        z_offset: workspace Z offset (m)
        optimize_tsp: whether to run TSP optimization

    Returns:
        dict with all trajectory data and metadata
    """
    void_width = void_bounds["void_width"]
    void_length = void_bounds["void_length"]
    wall_height = void_bounds["shell_thickness"]

    logger.info(
        f"Full trajectory planning starting: void={void_width:.1f}x{void_length:.1f} mm, "
        f"cylinder radius={cyl_radius:.2f} mm, optimize_tsp={optimize_tsp}"
    )

    # Grid parameters
    nx, ny, hex_side = compute_grid_params(void_width, void_length)
    num_layers = int(np.ceil(wall_height / layer_height))
    logger.info(
        f"Planning phases: honeycomb {nx}x{ny} cells, hex_side={hex_side:.1f} mm, "
        f"wall_height={wall_height:.1f} mm ({num_layers} layers)"
    )

    # Generate grid
    grid = create_hex_grid(nx, ny, hex_side)

    # Cell indices (all cells)
    cell_indices = np.array([[ix, iy] for iy in range(ny) for ix in range(nx)])

    # TSP optimization
    if optimize_tsp and len(cell_indices) > 2:
        logger.info(f"TSP optimization phase: {len(cell_indices)} cells")
        cell_indices = optimize_visitation_order(grid, cell_indices, rise)

    # UV trajectories
    logger.info("UV trajectory generation phase")
    traj_uv_dict = generate_uv_trajectories(
        grid, cell_indices, hex_side, rise, layer_height, wall_height
    )
    traj_uv = traj_uv_dict["combined"]

    # Compute UV offsets to center honeycomb on void
    grid_u_extent = grid[:, :, 0].max() - grid[:, :, 0].min()
    grid_v_extent = grid[:, :, 1].max() - grid[:, :, 1].min()
    u_offset = np.mean(void_bounds["u_range"]) - grid_u_extent / 2.0
    v_offset = np.mean(void_bounds["v_range"]) - grid_v_extent / 2.0

    # UV → XYZ
    logger.info("Conformal UV→XYZ mapping phase")
    traj_xyz, normals = uv_to_xyz(traj_uv, cyl_radius, cyl_cy, cyl_cz, u_offset, v_offset)

    # Nozzle orientations
    logger.info("Nozzle orientation computation phase")
    R_targets = compute_nozzle_orientations(normals)

    # Workspace transform (mm→m, base rotation, Z offset)
    logger.info(f"Workspace transform phase: z_offset={z_offset:.3f} m")
    traj_m, normals_t = apply_workspace_transform(traj_xyz, normals, z_offset)

    n_pts = traj_uv.shape[1]
    logger.info(f"Full trajectory planning complete: {n_pts} trajectory points")

    return {
        "traj_uv": traj_uv,
        "traj_xyz_mm": traj_xyz,
        "traj_m": traj_m,
        "normals": normals_t,
        "R_targets": R_targets,
        "grid": grid,
        "cell_indices": cell_indices,
        "hex_side": hex_side,
        "nx": nx,
        "ny": ny,
        "void_bounds": void_bounds,
        "n_points": n_pts,
    }
