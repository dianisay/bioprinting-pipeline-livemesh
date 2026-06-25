"""Multi-view synthetic dataset generator for volumetric wound reconstruction.

Generates 3D wound models and renders them from 8 orthogonal camera angles.
Similar to CT scanning: multiple 2D projections of a 3D object.

Key features:
- 3D wound topology (depth varies, not flat)
- 8 synchronized orthogonal views (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Ground truth labels: boundary, depth profile, layer fill amounts
- Synthetic but realistic: combines texture + realistic wound appearance

Dataset structure:
    dataset/
    ├── train/
    │   ├── sample_0000/
    │   │   ├── view_0000.png    (0°)
    │   │   ├── view_0045.png    (45°)
    │   │   ├── ...
    │   │   ├── view_0315.png    (315°)
    │   │   ├── centroid.npy     (2,) normalized center
    │   │   ├── radii.npy        (64,) normalized boundary radii
    │   │   ├── depth.npy        (64,) depth in mm at each radius
    │   │   └── layer_amounts.npy (64, 4) fill fractions per layer
    └── test/
        └── ...
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import math


class Wound3DModel:
    """
    Represents a 3D wound as a parametric surface.
    
    Parametrization: (u, v) ∈ [0, 1]² → 3D position
    - u: angular coordinate (0 to 2π)
    - v: radial coordinate (0 to 1, where 1 = wound edge)
    
    The surface is star-convex (radii vary with angle) and can have depth variation.
    """

    def __init__(
        self,
        image_size: int = 256,
        num_radii: int = 64,
        max_depth: float = 5.0,  # mm
    ):
        self.image_size = image_size
        self.num_radii = num_radii
        self.max_depth = max_depth

        # Random wound parameters
        self.centroid = np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(0.3, 0.7),
        ])  # Normalized [0, 1]

        # Generate radius profile (star-convex boundary)
        self.radii = self._generate_radii_profile()

        # Generate depth profile (varies with angle)
        self.depth_profile = self._generate_depth_profile()

        # Centroid in pixel coordinates
        self.centroid_px = self.centroid * image_size

    def _generate_radii_profile(self) -> np.ndarray:
        """Generate wound boundary as star-convex shape."""
        mean_radius = np.random.uniform(0.08, 0.35)
        
        # Add irregularity to boundary
        irregularity = np.random.uniform(0.3, 0.5)
        spikiness = np.random.uniform(0.15, 0.3)

        radii = mean_radius + np.random.normal(0, spikiness * mean_radius, self.num_radii)
        radii = np.clip(radii, mean_radius * 0.3, mean_radius * 1.7)

        # Smooth
        kernel = max(3, self.num_radii // 16)
        radii_smooth = np.convolve(
            np.concatenate([radii[-kernel:], radii, radii[:kernel]]),
            np.ones(kernel) / kernel,
            mode="valid",
        )[:self.num_radii]

        return radii_smooth

    def _generate_depth_profile(self) -> np.ndarray:
        """Generate depth variation across wound (not uniform)."""
        # Some angles deeper than others
        depth_base = np.random.uniform(2.0, self.max_depth)
        
        # Add sinusoidal variation
        angles = np.linspace(0, 2 * np.pi, self.num_radii, endpoint=False)
        variation = np.sin(angles + np.random.uniform(0, 2 * np.pi)) * 0.3
        
        depth = depth_base * (0.7 + variation)
        depth = np.clip(depth, 0.5, self.max_depth)
        
        return depth

    def get_3d_surface_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points on the 3D wound surface.
        
        Returns:
            X, Y, Z: (num_radii,) coordinates at the wound edge
        """
        angles = np.linspace(0, 2 * np.pi, self.num_radii, endpoint=False)
        
        # 2D boundary points
        x_2d = self.centroid[0] + self.radii * np.cos(angles)
        y_2d = self.centroid[1] + self.radii * np.sin(angles)
        
        # Convert to pixel coordinates
        X = x_2d * self.image_size
        Y = y_2d * self.image_size
        Z = self.depth_profile  # mm
        
        return X, Y, Z

    def get_layer_fill_pattern(self, num_layers: int = 4) -> np.ndarray:
        """
        Generate ground truth layer fill amounts.
        
        Deeper layers fill less radially (cone shape).
        
        Returns:
            (num_radii, num_layers) fill fraction at each radius/layer
        """
        layer_amounts = np.zeros((self.num_radii, num_layers))
        
        for layer_idx in range(num_layers):
            # Layer factor: 0 (deepest) to 1 (surface)
            layer_factor = layer_idx / max(1, num_layers - 1)
            
            # Deeper layers: fill less
            # Surface layer: fill more (approaching full radius)
            fill_fraction = 0.7 + layer_factor * 0.3
            
            # Add some per-radius variation
            variation = np.sin(np.linspace(0, 2 * np.pi, self.num_radii)) * 0.1
            fill = fill_fraction + variation
            fill = np.clip(fill, 0.5, 1.0)
            
            layer_amounts[:, layer_idx] = fill
        
        return layer_amounts


class MultiViewWoundDataset:
    """
    Generate synthetic multi-view wound dataset.
    
    Creates 3D wound models and renders them from 8 camera angles.
    Saves images and ground truth labels for training.
    """

    def __init__(
        self,
        output_dir: str = "data/synthetic_multiview",
        num_samples: int = 100,
        image_size: int = 256,
        num_views: int = 8,
        num_radii: int = 64,
        num_layers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_views = num_views
        self.num_radii = num_radii
        self.num_layers = num_layers

        # Camera angles (evenly spaced around wound)
        self.camera_angles = np.linspace(0, 360, num_views, endpoint=False)

    def generate_dataset(self, seed: int = 42) -> None:
        """Generate full dataset and save to disk."""
        np.random.seed(seed)

        # Create directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Split: 70% train, 15% val, 15% test
        num_train = int(0.70 * self.num_samples)
        num_val = int(0.15 * self.num_samples)

        all_metadata = []

        for sample_idx in range(self.num_samples):
            # Determine split
            if sample_idx < num_train:
                split_dir = train_dir
            elif sample_idx < num_train + num_val:
                split_dir = val_dir
            else:
                split_dir = test_dir

            # Create sample directory
            sample_name = f"sample_{sample_idx:04d}"
            sample_dir = split_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Generate 3D wound
            wound = Wound3DModel(
                image_size=self.image_size,
                num_radii=self.num_radii,
            )

            # Render from all camera angles
            for view_idx, angle_deg in enumerate(self.camera_angles):
                view_img = self._render_view(wound, angle_deg)

                # Save view
                view_path = sample_dir / f"view_{int(angle_deg):04d}.png"
                cv2.imwrite(str(view_path), view_img)

            # Save ground truth labels
            np.save(sample_dir / "centroid.npy", wound.centroid)
            np.save(sample_dir / "radii.npy", wound.radii)
            np.save(sample_dir / "depth.npy", wound.depth_profile)
            
            layer_amounts = wound.get_layer_fill_pattern(self.num_layers)
            np.save(sample_dir / "layer_amounts.npy", layer_amounts)

            # Metadata
            if sample_idx < num_train:
                split_name = "train"
            elif sample_idx < num_train + num_val:
                split_name = "val"
            else:
                split_name = "test"
            all_metadata.append({
                "sample_name": sample_name,
                "centroid": wound.centroid.tolist(),
                "mean_radius": float(wound.radii.mean()),
                "mean_depth": float(wound.depth_profile.mean()),
                "split": split_name,
            })

            if (sample_idx + 1) % 10 == 0:
                print(f"Generated {sample_idx + 1}/{self.num_samples} samples")

        print(f"\n✓ Dataset saved to {self.output_dir}")
        print(f"  Train: {num_train} samples")
        print(f"  Val:   {num_val} samples")
        print(f"  Test:  {self.num_samples - num_train - num_val} samples")
        print(f"  Views per sample: {self.num_views}")
        print(f"  Image size: {self.image_size}×{self.image_size}")

    def _render_view(self, wound: Wound3DModel, camera_angle_deg: float) -> np.ndarray:
        """
        Render wound from a specific camera angle.
        
        Camera rotates around the wound, looking at it from the side.
        
        Args:
            wound: Wound3DModel instance
            camera_angle_deg: angle in degrees (0, 45, 90, ...)
        
        Returns:
            (256, 256, 3) RGB image
        """
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Generate background (skin texture)
        image = self._generate_skin_background(self.image_size)

        # Get 3D surface points
        X, Y, Z = wound.get_3d_surface_points()

        # Project 3D points to 2D based on camera angle
        camera_angle_rad = np.radians(camera_angle_deg)

        # Simple orthographic projection with rotation
        # Rotate point cloud around Z axis
        cos_a = np.cos(camera_angle_rad)
        sin_a = np.sin(camera_angle_rad)

        X_rot = X * cos_a - Z * sin_a
        Y_rot = Y
        Z_rot = X * sin_a + Z * cos_a

        # Project to 2D (just use X_rot, Y_rot)
        x_2d = X_rot
        y_2d = Y_rot

        # Normalize to [0, image_size]
        x_px = ((x_2d / self.image_size) * self.image_size).astype(np.int32)
        y_px = ((y_2d / self.image_size) * self.image_size).astype(np.int32)

        # Clip to image bounds
        valid = (x_px >= 0) & (x_px < self.image_size) & (y_px >= 0) & (y_px < self.image_size)
        x_px = x_px[valid]
        y_px = y_px[valid]

        # Draw wound boundary (green line)
        if len(x_px) > 1:
            pts = np.column_stack([x_px, y_px])
            pts_closed = np.vstack([pts, pts[0]])
            cv2.polylines(image, [pts_closed], isClosed=False, color=(0, 255, 0), thickness=2)

        # Fill wound interior with wound color
        if len(x_px) > 2:
            pts_int = np.column_stack([x_px, y_px]).reshape((-1, 1, 2)).astype(np.int32)
            
            # Create mask
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_int], 255)

            # Apply wound coloring
            wound_color = self._get_wound_color()
            mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
            
            # Blend
            image = (image * (1 - mask_3ch * 0.7) + wound_color * mask_3ch * 0.7).astype(np.uint8)

        # Add noise/texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def _generate_skin_background(self, size: int) -> np.ndarray:
        """Generate realistic skin-tone background."""
        # Random skin tone in HSV
        hue = np.random.randint(8, 25)
        sat = np.random.randint(80, 180)
        val = np.random.randint(140, 230)

        base = np.full((size, size, 3), [hue, sat, val], dtype=np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)

        # Add texture
        noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
        noise = cv2.GaussianBlur(noise, (21, 21), 5)
        image = cv2.add(base, noise)

        return image

    def _get_wound_color(self) -> np.ndarray:
        """Generate realistic wound color (reddish)."""
        hue = np.random.randint(0, 12)
        sat = np.random.randint(120, 220)
        val = np.random.randint(80, 180)

        wound_color = np.full((256, 256, 3), [hue, sat, val], dtype=np.uint8)
        wound_color = cv2.cvtColor(wound_color, cv2.COLOR_HSV2BGR)

        return wound_color


class MultiViewWoundLoader:
    """
    PyTorch-compatible data loader for multi-view wounds.
    
    Loads images and labels from disk.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        num_views: int = 8,
    ):
        self.dataset_dir = Path(dataset_dir) / split
        self.num_views = num_views

        # Find all sample directories
        self.samples = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        print(f"Loaded {len(self.samples)} samples from {split} split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single multi-view sample.
        
        Returns:
            dict with:
                - views: (num_views, 3, 256, 256) torch tensor
                - centroid: (2,) torch tensor
                - radii: (64,) torch tensor
                - depth: (64,) torch tensor
                - layer_amounts: (64, 4) torch tensor
        """
        import torch

        sample_dir = self.samples[idx]

        # Load views
        views_list = []
        for angle_idx in range(self.num_views):
            angle_deg = int(360 * angle_idx / self.num_views)
            view_path = sample_dir / f"view_{angle_deg:04d}.png"

            if not view_path.exists():
                # Fallback: try to find closest angle
                view_files = sorted(sample_dir.glob("view_*.png"))
                if view_files:
                    view_path = view_files[angle_idx % len(view_files)]
                else:
                    raise FileNotFoundError(f"No views found in {sample_dir}")

            # Load and convert to tensor
            img = cv2.imread(str(view_path))  # BGR, (H, W, 3)
            if img is None:
                raise ValueError(f"Could not load {view_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            views_list.append(img_tensor)

        views = torch.stack(views_list, dim=0)  # (num_views, 3, H, W)

        # Load labels
        centroid = torch.tensor(np.load(sample_dir / "centroid.npy"), dtype=torch.float32)
        radii = torch.tensor(np.load(sample_dir / "radii.npy"), dtype=torch.float32)
        depth = torch.tensor(np.load(sample_dir / "depth.npy"), dtype=torch.float32)
        layer_amounts = torch.tensor(np.load(sample_dir / "layer_amounts.npy"), dtype=torch.float32)

        return {
            "views": views,
            "centroid": centroid,
            "radii": radii,
            "depth": depth,
            "layer_amounts": layer_amounts,
        }


# ============================================================
# Quick Test / Generation Script
# ============================================================

if __name__ == "__main__":
    print("Generating synthetic multi-view wound dataset...")
    print("=" * 60)

    # Generate dataset
    dataset_gen = MultiViewWoundDataset(
        output_dir="data/synthetic_multiview",
        num_samples=10,  # Small for testing
        image_size=256,
        num_views=8,
        num_radii=64,
        num_layers=4,
    )

    dataset_gen.generate_dataset(seed=42)

    # Test loader
    print("\nTesting MultiViewWoundLoader...")
    loader = MultiViewWoundLoader(
        dataset_dir="data/synthetic_multiview",
        split="train",
        num_views=8,
    )

    # Load one sample
    sample = loader[0]
    print(f"Sample shapes:")
    print(f"  Views: {sample['views'].shape}")
    print(f"  Centroid: {sample['centroid'].shape}")
    print(f"  Radii: {sample['radii'].shape}")
    print(f"  Depth: {sample['depth'].shape}")
    print(f"  Layer amounts: {sample['layer_amounts'].shape}")

    print("\n✓ Dataset generation and loading test passed!")
