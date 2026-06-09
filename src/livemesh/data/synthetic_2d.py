"""Synthetic wound image generator.

Generates star-convex wound shapes on skin-texture backgrounds.
Ground-truth polar representation is known exactly from the generation process.
"""

import numpy as np
import cv2
from typing import Tuple
from pathlib import Path


def generate_star_convex_wound(
    image_size: int = 256,
    num_radii: int = 64,
    min_radius_ratio: float = 0.08,
    max_radius_ratio: float = 0.35,
    irregularity: float = 0.4,
    spikiness: float = 0.2,
) -> dict:
    """Generate a single synthetic wound with known polar ground truth.

    Creates a star-convex polygon with controlled randomness, rendered
    on a skin-colored background with noise and texture.

    Args:
        image_size: output image dimensions (square)
        num_radii: number of boundary points (N)
        min_radius_ratio: minimum mean radius as fraction of image size
        max_radius_ratio: maximum mean radius as fraction of image size
        irregularity: how much angles deviate from uniform [0, 1]
        spikiness: variance of radii around mean [0, 1]

    Returns:
        dict with:
            - image: (H, W, 3) RGB uint8
            - mask: (H, W) binary uint8
            - centroid: (2,) normalized centroid
            - radii: (N,) normalized radii
            - points: (N, 2) normalized Cartesian points
            - angles: (N,) angles used
    """
    # Random centroid (keep away from edges)
    margin = 0.25
    cx = np.random.uniform(margin, 1 - margin) * image_size
    cy = np.random.uniform(margin, 1 - margin) * image_size

    # Random mean radius
    mean_radius = np.random.uniform(min_radius_ratio, max_radius_ratio) * image_size

    # Generate angles with controlled irregularity
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_radii), num_radii)
    angle_perturbation = np.random.uniform(-irregularity, irregularity, num_radii)
    angle_perturbation *= (2 * np.pi / num_radii) * 0.3
    perturbed_angles = angles + angle_perturbation

    # Generate radii with controlled spikiness
    radii = mean_radius + np.random.normal(0, spikiness * mean_radius, num_radii)
    radii = np.clip(radii, mean_radius * 0.3, mean_radius * 1.7)

    # Smooth radii slightly for more natural look
    kernel_size = max(3, num_radii // 16)
    radii_smooth = np.convolve(
        np.concatenate([radii[-kernel_size:], radii, radii[:kernel_size]]),
        np.ones(kernel_size) / kernel_size,
        mode="valid",
    )[:num_radii]

    # Compute boundary points
    points_px = np.zeros((num_radii, 2))
    points_px[:, 0] = cx + radii_smooth * np.cos(angles)
    points_px[:, 1] = cy + radii_smooth * np.sin(angles)

    # Clip to image bounds
    points_px = np.clip(points_px, 2, image_size - 3)

    # Recompute radii from clipped points (ground truth)
    final_radii = np.sqrt((points_px[:, 0] - cx) ** 2 + (points_px[:, 1] - cy) ** 2)

    # Generate mask
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    pts_int = points_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_int], 255)

    # Generate image
    image = _generate_skin_background(image_size)
    image = _apply_wound_appearance(image, mask)

    # Normalize ground truth
    centroid_norm = np.array([cx / image_size, cy / image_size], dtype=np.float32)
    radii_norm = (final_radii / image_size).astype(np.float32)
    points_norm = (points_px / image_size).astype(np.float32)

    return {
        "image": image,
        "mask": mask,
        "centroid": centroid_norm,
        "radii": radii_norm,
        "points": points_norm,
        "angles": angles.astype(np.float32),
    }


def _generate_skin_background(size: int) -> np.ndarray:
    """Generate a randomized skin-tone background with subtle texture."""
    # Random skin tone base (HSV space for natural variation)
    hue = np.random.randint(8, 25)
    sat = np.random.randint(80, 180)
    val = np.random.randint(140, 230)

    base = np.full((size, size, 3), [hue, sat, val], dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)

    # Add Perlin-like noise via gaussian blur of random noise
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (21, 21), 5)
    image = cv2.add(base, noise)

    return image


def _apply_wound_appearance(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply wound-like coloring inside the mask region."""
    wound_image = image.copy()

    # Wound color (reddish/pinkish, variable)
    hue = np.random.randint(0, 12)
    sat = np.random.randint(120, 220)
    val = np.random.randint(80, 180)

    wound_color = np.full_like(image, [hue, sat, val])
    wound_color = cv2.cvtColor(wound_color.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Add internal texture to wound
    noise = np.random.randint(0, 40, image.shape, dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (11, 11), 3)
    wound_color = cv2.add(wound_color, noise)

    # Blend wound into image
    mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0

    # Soft edge via gaussian blur on mask
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 2)
    soft_mask = np.stack([soft_mask, soft_mask, soft_mask], axis=-1) / 255.0

    wound_image = (wound_color * soft_mask + image * (1 - soft_mask)).astype(np.uint8)

    return wound_image


def generate_dataset(
    output_dir: str,
    num_samples: int = 2000,
    image_size: int = 256,
    num_radii: int = 64,
    seed: int = 42,
):
    """Generate full synthetic dataset and save to disk.

    Args:
        output_dir: directory to save images, masks, and labels
        num_samples: number of samples to generate
        image_size: image dimensions
        num_radii: boundary point count
        seed: random seed for reproducibility
    """
    np.random.seed(seed)
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    labels = []

    for i in range(num_samples):
        sample = generate_star_convex_wound(image_size=image_size, num_radii=num_radii)

        # Save image and mask
        img_path = out / "images" / f"synth_{i:05d}.png"
        mask_path = out / "masks" / f"synth_{i:05d}.png"
        cv2.imwrite(str(img_path), sample["image"])
        cv2.imwrite(str(mask_path), sample["mask"])

        labels.append({
            "filename": f"synth_{i:05d}.png",
            "centroid": sample["centroid"].tolist(),
            "radii": sample["radii"].tolist(),
        })

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{num_samples}")

    # Save labels as numpy archive
    np.savez(
        out / "labels.npz",
        filenames=[l["filename"] for l in labels],
        centroids=np.array([l["centroid"] for l in labels]),
        radii=np.array([l["radii"] for l in labels]),
    )
    print(f"Dataset saved to {out} ({num_samples} samples)")
