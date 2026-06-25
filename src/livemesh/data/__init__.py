from livemesh.data.dataset import WoundBoundaryDataset, create_dataloaders
from livemesh.data.multiview_dataset import Wound3DModel, MultiViewWoundDataset, MultiViewWoundLoader
from livemesh.data.polar_conversion import mask_to_polar, polar_to_cartesian, polar_to_mask
from livemesh.data.synthetic_2d import generate_dataset, generate_star_convex_wound

# Lazy imports for modules requiring trimesh (not available on Kaggle)
def __getattr__(name):
    _synthetic_names = {
        "add_noise", "add_occlusion", "cylinder_patch",
        "flat_plane", "saddle_surface", "sphere_cap", "wound_crater",
    }
    if name in _synthetic_names:
        from livemesh.data import synthetic
        return getattr(synthetic, name)
    raise AttributeError(f"module 'livemesh.data' has no attribute {name!r}")

__all__ = [
    "WoundBoundaryDataset",
    "create_dataloaders",
    "Wound3DModel",
    "MultiViewWoundDataset",
    "MultiViewWoundLoader",
    "mask_to_polar",
    "polar_to_cartesian",
    "polar_to_mask",
    "sphere_cap",
    "saddle_surface",
    "wound_crater",
    "cylinder_patch",
    "flat_plane",
    "add_noise",
    "add_occlusion",
    "generate_star_convex_wound",
    "generate_dataset",
]
