"""
End-to-end wound segmentation pipeline.

Ported from your MATLAB ExtractingGCodeFromWoundSegment.m:
  Image -> U-Net -> binary mask -> morphological cleanup -> boundary extraction

Python equivalents of your MATLAB operations:
  bwareaopen(mask, 50)    -> skimage.morphology.remove_small_objects(min_size=50)
  imfill(mask, 'holes')   -> scipy.ndimage.binary_fill_holes
  bwboundaries(mask)      -> skimage.measure.find_contours
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects


@dataclass
class SegmentationResult:
    mask: NDArray[np.bool_]          # (H, W) binary wound mask
    boundary: NDArray[np.float64]    # (K, 2) ordered boundary points [row, col]
    boundary_mm: NDArray[np.float64] # (K, 2) boundary in mm [x, y]
    area_px: int
    area_mm2: float


def segment_wound(
    image: NDArray[np.uint8],
    model: torch.nn.Module,
    device: str = "cuda",
    min_area_px: int = 50,
    simplification_factor: int = 25,
    mm_per_pixel: float = 0.25,
    origin_offset_mm: tuple[float, float] = (50.0, 50.0),
    input_size: tuple[int, int] = (512, 512),
) -> SegmentationResult:
    """Full pipeline: RGB image -> wound mask -> cleaned boundary in mm.

    Parameters match your MATLAB pipeline:
    - min_area_px=50 matches bwareaopen(mask, 50)
    - simplification_factor=25 matches boundary(1:25:end, :)
    - mm_per_pixel=0.25 and origin_offset match your calibration placeholders
    """
    h_orig, w_orig = image.shape[:2]
    resized = cv2.resize(image, input_size)
    tensor = _image_to_tensor(resized, device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    wound_mask = pred == 1  # class 1 = wound

    if wound_mask.shape != (h_orig, w_orig):
        wound_mask = cv2.resize(
            wound_mask.astype(np.uint8), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    cleaned = remove_small_objects(wound_mask, min_size=min_area_px)
    filled = binary_fill_holes(cleaned)

    contours = find_contours(filled.astype(float), 0.5)
    if not contours:
        return SegmentationResult(
            mask=filled,
            boundary=np.empty((0, 2)),
            boundary_mm=np.empty((0, 2)),
            area_px=0,
            area_mm2=0.0,
        )

    boundary = max(contours, key=len)
    boundary = boundary[::simplification_factor]

    boundary_mm = np.column_stack([
        boundary[:, 1] * mm_per_pixel + origin_offset_mm[0],  # col -> X
        boundary[:, 0] * mm_per_pixel + origin_offset_mm[1],  # row -> Y
    ])

    area_px = int(np.sum(filled))
    area_mm2 = area_px * mm_per_pixel**2

    return SegmentationResult(
        mask=filled,
        boundary=boundary,
        boundary_mm=boundary_mm,
        area_px=area_px,
        area_mm2=area_mm2,
    )


def _image_to_tensor(image: NDArray[np.uint8], device: str) -> torch.Tensor:
    """Convert HWC uint8 image to NCHW float32 tensor, normalized to [0, 1]."""
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)
