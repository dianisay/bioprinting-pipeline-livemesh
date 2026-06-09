"""PyTorch Dataset and DataLoader for wound boundary detection.

Handles both FUSeg (real) and synthetic data, with unified polar GT format.
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

from .polar_conversion import mask_to_polar


class WoundBoundaryDataset(Dataset):
    """Combined dataset for wound boundary detection training.

    Loads images and returns them with polar ground-truth labels.
    Supports both FUSeg (mask → polar conversion) and synthetic
    (pre-computed polar labels) data sources.
    """

    def __init__(
        self,
        fuseg_dir: Optional[str] = None,
        synthetic_dir: Optional[str] = None,
        split: str = "train",
        image_size: int = 256,
        num_radii: int = 64,
        augment: bool = False,
    ):
        """
        Args:
            fuseg_dir: path to FUSeg dataset (with images/ and masks/ subdirs)
            synthetic_dir: path to synthetic dataset (with images/, masks/, labels.npz)
            split: one of 'train', 'val', 'test'
            image_size: resize images to this size
            num_radii: number of polar samples (N)
            augment: apply data augmentation (training only)
        """
        self.image_size = image_size
        self.num_radii = num_radii
        self.augment = augment
        self.samples = []

        if fuseg_dir:
            self._load_fuseg(Path(fuseg_dir), split)
        if synthetic_dir:
            self._load_synthetic(Path(synthetic_dir), split)

    def _load_fuseg(self, root: Path, split: str):
        """Load FUSeg dataset with train/val/test split."""
        images_dir = root / "images"
        masks_dir = root / "masks"

        if not images_dir.exists():
            return

        all_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

        # Deterministic split
        np.random.seed(42)
        indices = np.random.permutation(len(all_files))
        n_train = int(0.70 * len(all_files))
        n_val = int(0.15 * len(all_files))

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        for idx in selected:
            img_path = all_files[idx]
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "source": "fuseg",
                })

    def _load_synthetic(self, root: Path, split: str):
        """Load synthetic dataset with pre-computed labels."""
        labels_path = root / "labels.npz"
        if not labels_path.exists():
            return

        data = np.load(labels_path, allow_pickle=True)
        filenames = data["filenames"]
        centroids = data["centroids"]
        radii = data["radii"]

        # Split synthetic data
        n = len(filenames)
        np.random.seed(123)
        indices = np.random.permutation(n)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        for idx in selected:
            self.samples.append({
                "image_path": str(root / "images" / filenames[idx]),
                "source": "synthetic",
                "centroid": centroids[idx],
                "radii": radii[idx],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load and preprocess image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Get polar ground truth
        if sample["source"] == "synthetic" and "centroid" in sample:
            centroid = sample["centroid"]
            radii_arr = sample["radii"]

            # Resample radii if stored count differs from requested num_radii
            if len(radii_arr) != self.num_radii:
                original_angles = np.linspace(0, 2 * np.pi, len(radii_arr), endpoint=False)
                target_angles = np.linspace(0, 2 * np.pi, self.num_radii, endpoint=False)
                radii_arr = np.interp(target_angles, original_angles, radii_arr, period=2 * np.pi)

            angles = np.linspace(0, 2 * np.pi * (1 - 1 / self.num_radii), self.num_radii)
            points = np.stack([
                centroid[0] + radii_arr * np.cos(angles),
                centroid[1] + radii_arr * np.sin(angles),
            ], axis=-1)
        else:
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            polar = mask_to_polar(mask, self.num_radii, self.image_size)
            centroid = polar["centroid"]
            radii_arr = polar["radii"]
            points = polar["points"]

        # Augmentation
        if self.augment:
            image, centroid, radii_arr, points = self._augment(
                image, centroid, radii_arr, points
            )

        # To tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        centroid_tensor = torch.from_numpy(centroid.copy()).float()
        radii_tensor = torch.from_numpy(radii_arr.copy()).float()
        points_tensor = torch.from_numpy(points.copy()).float()

        return {
            "image": image_tensor,
            "centroid": centroid_tensor,
            "radii": radii_tensor,
            "points": points_tensor,
        }

    def _augment(
        self,
        image: np.ndarray,
        centroid: np.ndarray,
        radii: np.ndarray,
        points: np.ndarray,
    ) -> Tuple:
        """Apply random augmentations (preserving polar GT consistency)."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            centroid = centroid.copy()
            points = points.copy()
            centroid[0] = 1.0 - centroid[0]
            points[:, 0] = 1.0 - points[:, 0]
            # Radii stay the same, but order reverses (mirror = flip angles)
            radii = radii[::-1].copy()
            points = points[::-1].copy()

        # Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return image, centroid, radii, points


def create_dataloaders(
    fuseg_dir: Optional[str] = None,
    synthetic_dir: Optional[str] = None,
    batch_size: int = 8,
    image_size: int = 256,
    num_radii: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = WoundBoundaryDataset(
        fuseg_dir, synthetic_dir, split="train",
        image_size=image_size, num_radii=num_radii, augment=True,
    )
    val_ds = WoundBoundaryDataset(
        fuseg_dir, synthetic_dir, split="val",
        image_size=image_size, num_radii=num_radii, augment=False,
    )
    test_ds = WoundBoundaryDataset(
        fuseg_dir, synthetic_dir, split="test",
        image_size=image_size, num_radii=num_radii, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
