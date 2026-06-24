"""Bridge module: converts volumetric decoder output to trajectory planner input.

This module closes the gap between the vision stack (PolarDecoder3DLayered) and
the robotics stack (plan_full_trajectory), enabling the full autonomous
image-to-trajectory pipeline.

Decoder outputs are in normalized image coordinates [0,1]. This module converts
them to physical millimeters using a known wound scale (from camera calibration
or wound measurement), then produces a void_bounds dict and per-cell depth map
compatible with plan_full_trajectory().
"""

import numpy as np
from typing import Dict, Optional, Tuple

from utils.logging_config import get_logger

logger = get_logger("modules.wound_to_trajectory")


def decoder_output_to_physical(
    centroid: np.ndarray,
    radii: np.ndarray,
    depth: np.ndarray,
    layer_amounts: np.ndarray,
    wound_scale_mm: float = 60.0,
    wound_center_mm: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Convert normalized decoder output to physical millimeters.

    The decoder produces values in [0, 1] normalized image space. This function
    maps them to physical wound coordinates using a known scale factor
    (mm per unit in normalized space), typically from camera calibration.

    Args:
        centroid: (2,) wound center in normalized coords [0, 1]
        radii: (num_radii,) boundary radii in normalized coords
        depth: (num_radii,) depth at each angular sample in normalized coords
        layer_amounts: (num_radii, num_layers) fill fraction per layer per angle
        wound_scale_mm: physical extent of the image field of view in mm
        wound_center_mm: (2,) physical center offset; defaults to FOV center

    Returns:
        dict with physical-scale arrays:
            centroid_mm, radii_mm, depth_mm, layer_amounts (unchanged, unitless)
    """
    if wound_center_mm is None:
        wound_center_mm = np.array([wound_scale_mm / 2.0, wound_scale_mm / 2.0])

    centroid_mm = centroid * wound_scale_mm
    radii_mm = radii * wound_scale_mm
    depth_mm = depth * wound_scale_mm

    logger.info(
        "Decoder -> physical: centroid=(%.1f, %.1f)mm, mean_radius=%.1fmm, "
        "mean_depth=%.1fmm, layers=%d",
        centroid_mm[0], centroid_mm[1],
        radii_mm.mean(), depth_mm.mean(),
        layer_amounts.shape[1] if layer_amounts.ndim == 2 else 1,
    )

    return {
        "centroid_mm": centroid_mm,
        "radii_mm": radii_mm,
        "depth_mm": depth_mm,
        "layer_amounts": layer_amounts,
    }


def physical_to_void_bounds(
    centroid_mm: np.ndarray,
    radii_mm: np.ndarray,
    depth_mm: np.ndarray,
    layer_amounts: np.ndarray,
    cyl_radius: float = 50.0,
    cyl_center_x_mm: float = 0.0,
) -> Dict:
    """Convert physical wound measurements to void_bounds for trajectory planner.

    Computes the void geometry on a cylindrical surface from the polar wound
    representation. The wound boundary in polar coords defines the void extent;
    depth defines the shell thickness to fill.

    Args:
        centroid_mm: (2,) wound center in mm
        radii_mm: (num_radii,) radii in mm (polar boundary)
        depth_mm: (num_radii,) depth at each angle in mm
        layer_amounts: (num_radii, num_layers) per-cell fill fractions
        cyl_radius: cylinder radius in mm (from STL or known scaffold geometry)
        cyl_center_x_mm: axial position of wound center on cylinder

    Returns:
        void_bounds dict compatible with plan_full_trajectory(), plus
        per-angle depth data for variable-thickness deposition
    """
    num_radii = len(radii_mm)
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_radii), num_radii)

    boundary_x = centroid_mm[0] + radii_mm * np.cos(angles)
    boundary_y = centroid_mm[1] + radii_mm * np.sin(angles)

    x_extent = boundary_x.max() - boundary_x.min()
    y_extent = boundary_y.max() - boundary_y.min()

    void_width = float(y_extent)
    void_length = float(x_extent)

    shell_thickness = float(depth_mm.mean())

    u_center = centroid_mm[1] / cyl_radius * cyl_radius
    v_center = centroid_mm[0] + cyl_center_x_mm

    u_range = np.array([u_center - void_width / 2, u_center + void_width / 2])
    v_range = np.array([v_center - void_length / 2, v_center + void_length / 2])

    void_bounds = {
        "void_width": void_width,
        "void_length": void_length,
        "shell_thickness": shell_thickness,
        "u_range": u_range,
        "v_range": v_range,
        "theta_min": u_range[0] / cyl_radius,
        "theta_max": u_range[1] / cyl_radius,
        "x_min": v_range[0],
        "x_max": v_range[1],
    }

    logger.info(
        "Void bounds: %.1f x %.1f mm, shell=%.1fmm, u=[%.1f, %.1f], v=[%.1f, %.1f]",
        void_width, void_length, shell_thickness,
        u_range[0], u_range[1], v_range[0], v_range[1],
    )

    return {
        "void_bounds": void_bounds,
        "depth_profile_mm": depth_mm,
        "layer_amounts": layer_amounts,
        "boundary_angles": angles,
        "boundary_x_mm": boundary_x,
        "boundary_y_mm": boundary_y,
    }


def bridge_decoder_to_planner(
    decoder_output: Dict[str, np.ndarray],
    wound_scale_mm: float = 60.0,
    cyl_radius: float = 50.0,
    cyl_cy: float = 0.0,
    cyl_cz: float = 50.0,
) -> Dict:
    """End-to-end bridge: decoder output dict -> trajectory planner arguments.

    This is the main entry point that connects the vision stack to the robotics
    stack. It takes the raw decoder output tensor dictionary (as produced by
    PolarDecoder3DLayered) and returns everything needed to call
    plan_full_trajectory().

    Args:
        decoder_output: dict with keys 'centroid', 'radii', 'depth',
            'layer_amounts' as numpy arrays (single sample, not batched)
        wound_scale_mm: field of view size in mm
        cyl_radius: scaffold cylinder radius in mm
        cyl_cy: cylinder center Y coordinate
        cyl_cz: cylinder center Z coordinate

    Returns:
        dict with:
            - void_bounds: ready for plan_full_trajectory()
            - cyl_radius, cyl_cy, cyl_cz: cylinder parameters
            - depth_profile_mm: per-angle depth for variable deposition
            - layer_amounts: per-cell fill fractions
    """
    centroid = decoder_output["centroid"]
    radii = decoder_output["radii"]
    depth = decoder_output["depth"]
    layer_amounts = decoder_output["layer_amounts"]

    if hasattr(centroid, "detach"):
        centroid = centroid.detach().cpu().numpy()
        radii = radii.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()
        layer_amounts = layer_amounts.detach().cpu().numpy()

    if centroid.ndim > 1:
        centroid = centroid.squeeze(0)
        radii = radii.squeeze(0)
        depth = depth.squeeze(0)
        layer_amounts = layer_amounts.squeeze(0)

    physical = decoder_output_to_physical(
        centroid, radii, depth, layer_amounts,
        wound_scale_mm=wound_scale_mm,
    )

    geometry = physical_to_void_bounds(
        physical["centroid_mm"],
        physical["radii_mm"],
        physical["depth_mm"],
        physical["layer_amounts"],
        cyl_radius=cyl_radius,
    )

    logger.info("Bridge complete: vision -> robotics ready")

    return {
        "void_bounds": geometry["void_bounds"],
        "cyl_radius": cyl_radius,
        "cyl_cy": cyl_cy,
        "cyl_cz": cyl_cz,
        "depth_profile_mm": geometry["depth_profile_mm"],
        "layer_amounts": geometry["layer_amounts"],
        "boundary_x_mm": geometry["boundary_x_mm"],
        "boundary_y_mm": geometry["boundary_y_mm"],
    }


def apply_depth_correction(
    bridge_result: Dict,
    corrected_depth_mm: np.ndarray,
    corrected_layer_amounts: Optional[np.ndarray] = None,
) -> Dict:
    """Update bridge result with corrected depth from sensor fusion.

    Called during closed-loop operation when the depth sensor provides
    a refined measurement that disagrees with the initial prediction.
    Recomputes void_bounds with the new depth values.

    Args:
        bridge_result: original output from bridge_decoder_to_planner()
        corrected_depth_mm: (num_radii,) fused depth from depth_fusion module
        corrected_layer_amounts: (num_radii, num_layers) optional updated fill plan

    Returns:
        Updated bridge_result dict with corrected void_bounds
    """
    updated = bridge_result.copy()

    # Update shell thickness (mean of corrected depth)
    new_shell_thickness = float(corrected_depth_mm.mean())
    old_shell_thickness = bridge_result["void_bounds"]["shell_thickness"]

    updated["depth_profile_mm"] = corrected_depth_mm
    updated["void_bounds"] = bridge_result["void_bounds"].copy()
    updated["void_bounds"]["shell_thickness"] = new_shell_thickness

    if corrected_layer_amounts is not None:
        updated["layer_amounts"] = corrected_layer_amounts

    logger.info(
        "Depth correction applied: shell %.1fmm -> %.1fmm (delta=%.2fmm)",
        old_shell_thickness, new_shell_thickness,
        new_shell_thickness - old_shell_thickness,
    )

    return updated
