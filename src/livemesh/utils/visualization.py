"""Visualization utilities for thesis figures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Optional, List, Tuple


THESIS_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def set_thesis_style():
    """Apply consistent matplotlib style for thesis figures."""
    plt.rcParams.update(THESIS_STYLE)


def plot_boundary_comparison(
    image: np.ndarray,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot predicted vs ground truth boundary overlaid on image.

    Args:
        image: (H, W, 3) RGB image
        gt_points: (N, 2) ground truth boundary in [0, 1]
        pred_points: (M, 2) predicted boundary in [0, 1]
        title: Figure title
        save_path: If provided, save figure to this path
    """
    set_thesis_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    h, w = image.shape[:2]
    ax.imshow(image)

    # Ground truth
    gt_px = gt_points * np.array([w, h])
    gt_closed = np.vstack([gt_px, gt_px[0]])
    ax.plot(gt_closed[:, 0], gt_closed[:, 1], "g-", linewidth=2, label="Ground Truth")

    # Prediction
    pred_px = pred_points * np.array([w, h])
    pred_closed = np.vstack([pred_px, pred_px[0]])
    ax.plot(pred_closed[:, 0], pred_closed[:, 1], "r--", linewidth=2, label="Predicted")

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    components: Optional[dict] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves.

    Args:
        train_losses: Per-epoch training loss
        val_losses: Per-epoch validation loss
        components: Optional dict of component losses (e.g. centroid, radii, points)
        title: Figure title
        save_path: Save path
    """
    set_thesis_style()

    if components:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train")
    ax1.plot(epochs, val_losses, "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if components:
        for name, values in components.items():
            ax2.plot(epochs[: len(values)], values, label=name)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Component Loss")
        ax2.set_title("Loss Components")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_ablation_grid(
    images: List[np.ndarray],
    gt_boundaries: List[np.ndarray],
    predictions: dict,
    save_path: Optional[str] = None,
):
    """Grid comparison of all 3 decoder variants on sample images.

    Args:
        images: List of (H, W, 3) images
        gt_boundaries: List of (N, 2) GT boundaries
        predictions: Dict mapping decoder_name -> list of (N, 2) predictions
        save_path: Save path
    """
    set_thesis_style()
    n_images = len(images)
    n_decoders = len(predictions)

    fig, axes = plt.subplots(
        n_images, n_decoders + 1, figsize=(3 * (n_decoders + 1), 3 * n_images)
    )

    decoder_names = list(predictions.keys())

    for i in range(n_images):
        h, w = images[i].shape[:2]

        # GT column
        axes[i, 0].imshow(images[i])
        gt_px = gt_boundaries[i] * np.array([w, h])
        gt_closed = np.vstack([gt_px, gt_px[0]])
        axes[i, 0].plot(gt_closed[:, 0], gt_closed[:, 1], "g-", linewidth=2)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Ground Truth")

        # Decoder columns
        for j, name in enumerate(decoder_names):
            axes[i, j + 1].imshow(images[i])
            pred_px = predictions[name][i] * np.array([w, h])
            pred_closed = np.vstack([pred_px, pred_px[0]])
            axes[i, j + 1].plot(pred_closed[:, 0], pred_closed[:, 1], "r-", linewidth=2)
            axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(name)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig
