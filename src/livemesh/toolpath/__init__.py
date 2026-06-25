"""Toolpath planning: honeycomb, geodesic, conformal, and coverage modules."""

# Safe imports (numpy/scipy only — work on Kaggle)
from livemesh.toolpath.honeycomb import (
    compute_grid_params,
    create_hex_grid,
    hex_fill_points,
    hexagon_perimeter,
    line_points,
)
from livemesh.toolpath.tsp_solver import optimize_visitation_order, solve_tsp_mtz
from livemesh.toolpath.trajectory_planner import generate_uv_trajectories, plan_full_trajectory

# Lazy imports for modules requiring trimesh/potpourri3d (not on Kaggle)
def __getattr__(name):
    _conformal_names = {
        "ConformalMapResult", "apply_workspace_transform",
        "compute_nozzle_orientations", "cylinder_conformal_map",
        "general_conformal_map", "uv_to_xyz",
    }
    if name in _conformal_names:
        from livemesh.toolpath import conformal_map
        return getattr(conformal_map, name)
    if name == "coverage_uniformity":
        from livemesh.toolpath.coverage import coverage_uniformity
        return coverage_uniformity
    if name == "geodesic_toolpaths":
        from livemesh.toolpath.geodesic import geodesic_toolpaths
        return geodesic_toolpaths
    if name == "planar_slice":
        from livemesh.toolpath.planar_slicer import planar_slice
        return planar_slice
    raise AttributeError(f"module 'livemesh.toolpath' has no attribute {name!r}")

__all__ = [
    "geodesic_toolpaths",
    "planar_slice",
    "coverage_uniformity",
    "create_hex_grid",
    "hexagon_perimeter",
    "hex_fill_points",
    "compute_grid_params",
    "line_points",
    "solve_tsp_mtz",
    "optimize_visitation_order",
    "generate_uv_trajectories",
    "plan_full_trajectory",
    "cylinder_conformal_map",
    "general_conformal_map",
    "ConformalMapResult",
    "uv_to_xyz",
    "compute_nozzle_orientations",
    "apply_workspace_transform",
]
