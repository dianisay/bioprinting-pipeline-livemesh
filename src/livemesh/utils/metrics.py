"""Evaluation metrics for boundary detection and trajectory quality."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def chamfer_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute symmetric Chamfer distance between two point sets.

    Args:
        pred: (N, 2) predicted boundary points
        target: (M, 2) ground truth boundary points

    Returns:
        Mean bidirectional nearest-neighbor distance (mm or pixels)
    """
    from scipy.spatial import cKDTree

    tree_pred = cKDTree(pred)
    tree_target = cKDTree(target)

    dist_pred_to_target, _ = tree_target.query(pred)
    dist_target_to_pred, _ = tree_pred.query(target)

    return 0.5 * (dist_pred_to_target.mean() + dist_target_to_pred.mean())


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance (worst-case boundary mismatch).

    Args:
        pred: (N, 2) predicted boundary points
        target: (M, 2) ground truth boundary points

    Returns:
        Maximum of both directed Hausdorff distances
    """
    d_forward = directed_hausdorff(pred, target)[0]
    d_backward = directed_hausdorff(target, pred)[0]
    return max(d_forward, d_backward)


def boundary_iou(pred: np.ndarray, target: np.ndarray, image_size: int = 256) -> float:
    """Compute IoU between filled polygons of predicted and target boundaries.

    Args:
        pred: (N, 2) predicted boundary points (pixel coordinates)
        target: (M, 2) ground truth boundary points
        image_size: Size of the image for rasterization

    Returns:
        IoU score in [0, 1]
    """
    import cv2

    mask_pred = np.zeros((image_size, image_size), dtype=np.uint8)
    mask_target = np.zeros((image_size, image_size), dtype=np.uint8)

    pts_pred = (pred * image_size).astype(np.int32).reshape((-1, 1, 2))
    pts_target = (target * image_size).astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask_pred, [pts_pred], 1)
    cv2.fillPoly(mask_target, [pts_target], 1)

    intersection = np.logical_and(mask_pred, mask_target).sum()
    union = np.logical_or(mask_pred, mask_target).sum()

    return intersection / max(union, 1)


def closure_error(points: np.ndarray) -> float:
    """Distance between first and last predicted boundary point.

    For the polar decoder this should always be ~0 by construction.

    Args:
        points: (N, 2) ordered boundary points

    Returns:
        Euclidean distance between first and last point
    """
    return np.linalg.norm(points[0] - points[-1])


def ordering_consistency(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of point triplets that maintain correct angular ordering.

    Measures whether the predicted boundary traverses in a consistent
    direction (CW or CCW) matching the ground truth.

    Args:
        pred: (N, 2) predicted boundary
        target: (N, 2) ground truth boundary (same N)

    Returns:
        Percentage of consistent orderings [0, 100]
    """
    def _signed_area(pts):
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    pred_sign = np.sign(_signed_area(np.vstack([pred, pred[0]])))
    target_sign = np.sign(_signed_area(np.vstack([target, target[0]])))

    if pred_sign == target_sign:
        return 100.0

    # Check if reversing fixes it
    pred_reversed_sign = np.sign(_signed_area(np.vstack([pred[::-1], pred[-1]])))
    if pred_reversed_sign == target_sign:
        return 100.0

    return 0.0
