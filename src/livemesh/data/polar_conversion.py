"""Convert binary segmentation masks to polar ground-truth representation.

Takes a binary wound mask and produces:
- centroid (x_c, y_c) normalized to [0, 1]
- N radii at fixed angular intervals
- N Cartesian boundary points
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def mask_to_polar(
    mask: np.ndarray,
    num_radii: int = 64,
    image_size: Optional[int] = None,
) -> dict:
    """Convert binary mask to polar boundary representation.

    Algorithm:
    1. Find wound contour from mask
    2. Compute centroid of the contour
    3. For each of N angles, cast a ray from centroid and find intersection with contour
    4. Record the radius (distance from centroid to boundary) at each angle

    Args:
        mask: (H, W) binary mask (255 = wound, 0 = background)
        num_radii: number of angular samples (N)
        image_size: if provided, normalize coordinates to [0, 1] using this size

    Returns:
        dict with:
            - centroid: (2,) normalized centroid coordinates
            - radii: (N,) radii at each angle
            - points: (N, 2) Cartesian boundary points
            - angles: (N,) the fixed angles used
            - valid: bool indicating if conversion succeeded
    """
    if image_size is None:
        image_size = max(mask.shape)

    # Binarize
    binary = (mask > 127).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return _empty_polar(num_radii)

    # Take largest contour
    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 50:
        return _empty_polar(num_radii)

    # Compute centroid via moments
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return _empty_polar(num_radii)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Fixed angles (same as model uses)
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_radii), num_radii)

    # Extract contour points as (N_contour, 2) array
    contour_pts = contour.reshape(-1, 2).astype(np.float64)

    # For each angle, find the boundary point by ray casting
    radii = np.zeros(num_radii)
    points = np.zeros((num_radii, 2))

    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])

        # Project all contour points onto this ray direction
        # relative to centroid
        relative = contour_pts - np.array([cx, cy])
        projections = relative @ direction

        # Only consider points roughly in this direction
        # (within +/- angular tolerance)
        cross = relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
        angular_dist = np.abs(cross) / (np.linalg.norm(relative, axis=1) + 1e-8)

        # Points within ~5.6 degrees of the ray (sin(5.6°) ≈ 0.1)
        valid_mask = (projections > 0) & (angular_dist < 0.15)

        if valid_mask.any():
            # Take the farthest valid point along this ray
            valid_distances = np.linalg.norm(relative[valid_mask], axis=1)
            max_idx = np.argmax(valid_distances)
            radius = valid_distances[max_idx]
        else:
            # Fallback: nearest contour point to the expected direction
            expected_far = np.array([cx, cy]) + direction * 50
            dists = np.linalg.norm(contour_pts - expected_far, axis=1)
            nearest_idx = np.argmin(dists)
            radius = np.linalg.norm(contour_pts[nearest_idx] - np.array([cx, cy]))

        radii[i] = radius
        points[i, 0] = cx + radius * np.cos(angle)
        points[i, 1] = cy + radius * np.sin(angle)

    # Normalize to [0, 1]
    centroid_norm = np.array([cx / image_size, cy / image_size])
    radii_norm = radii / image_size
    points_norm = points / image_size

    return {
        "centroid": centroid_norm.astype(np.float32),
        "radii": radii_norm.astype(np.float32),
        "points": points_norm.astype(np.float32),
        "angles": angles.astype(np.float32),
        "valid": True,
    }


def polar_to_cartesian(
    centroid: np.ndarray, radii: np.ndarray, angles: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert polar representation back to Cartesian points.

    Args:
        centroid: (2,) centroid coordinates
        radii: (N,) radii values
        angles: (N,) angles in radians. If None, uses evenly spaced.

    Returns:
        (N, 2) Cartesian boundary points
    """
    N = len(radii)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi * (1 - 1 / N), N)

    x = centroid[0] + radii * np.cos(angles)
    y = centroid[1] + radii * np.sin(angles)
    return np.stack([x, y], axis=-1)


def polar_to_mask(
    centroid: np.ndarray, radii: np.ndarray, image_size: int = 256
) -> np.ndarray:
    """Reconstruct binary mask from polar representation (for visualization/IoU).

    Args:
        centroid: (2,) normalized centroid
        radii: (N,) normalized radii
        image_size: output mask size

    Returns:
        (H, W) binary mask
    """
    points = polar_to_cartesian(centroid, radii)
    pts_px = (points * image_size).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 255)
    return mask


def _empty_polar(num_radii: int) -> dict:
    """Return empty polar representation for invalid masks."""
    return {
        "centroid": np.zeros(2, dtype=np.float32),
        "radii": np.zeros(num_radii, dtype=np.float32),
        "points": np.zeros((num_radii, 2), dtype=np.float32),
        "angles": np.linspace(0, 2 * np.pi * (1 - 1 / num_radii), num_radii).astype(np.float32),
        "valid": False,
    }
