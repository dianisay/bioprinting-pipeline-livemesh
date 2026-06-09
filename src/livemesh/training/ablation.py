"""Ablation study runner: trains all 3 decoder variants with identical settings.

Produces comparative results for the thesis (Section 4.2).
Designed to be run on Kaggle GPU.
"""

import logging

import json

logger = logging.getLogger(__name__)
import time
from pathlib import Path

from livemesh.training.train import Trainer
from livemesh.training.evaluate import evaluate
from livemesh.data.dataset import create_dataloaders


def run_ablation(
    fuseg_dir: str = None,
    synthetic_dir: str = None,
    output_dir: str = "results/ablation",
    batch_size: int = 8,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-4,
    num_points: int = 64,
    pretrained: bool = True,
):
    """Run complete ablation study: train + evaluate all 3 decoders.

    All variants use identical:
    - Encoder (ResNet-50 + 6-layer Transformer)
    - Data (same train/val/test splits)
    - Hyperparameters (Adam, lr=1e-4, batch=8, 100 epochs, patience=10)
    - Evaluation metrics

    Only the decoder differs.
    """
    decoder_types = ["polar", "detr", "autoregressive"]
    all_results = {}

    logger.info(
        f"Ablation study starting: decoders={decoder_types}, epochs={max_epochs}, "
        f"patience={patience}, lr={lr}, batch_size={batch_size}, "
        f"num_points={num_points}, pretrained={pretrained}, output={output_dir}"
    )
    # Create dataloaders once (shared across all variants)
    train_loader, val_loader, test_loader = create_dataloaders(
        fuseg_dir=fuseg_dir,
        synthetic_dir=synthetic_dir,
        batch_size=batch_size,
        num_radii=num_points,
    )

    total_start = time.time()

    for i, decoder_type in enumerate(decoder_types):
        logger.info(
            f"Ablation progress [{i + 1}/{len(decoder_types)}]: "
            f"training {decoder_type.upper()} decoder"
        )
        trainer = Trainer(
            decoder_type=decoder_type,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            num_points=num_points,
            output_dir=output_dir,
            pretrained_backbone=pretrained,
        )

        history = trainer.train(train_loader, val_loader)

        # Evaluate on test set
        checkpoint_path = str(Path(output_dir) / decoder_type / "best.pth")
        eval_results = evaluate(
            checkpoint_path=checkpoint_path,
            fuseg_dir=fuseg_dir,
            synthetic_dir=synthetic_dir,
            batch_size=batch_size,
        )

        all_results[decoder_type] = {
            "eval": eval_results,
            "best_val_loss": min(history["val_loss"]),
            "convergence_epoch": history["val_loss"].index(min(history["val_loss"])) + 1,
            "total_epochs": len(history["val_loss"]),
        }
        logger.info(
            f"Ablation [{decoder_type}] complete: best_val_loss="
            f"{all_results[decoder_type]['best_val_loss']:.4f}, "
            f"convergence_epoch={all_results[decoder_type]['convergence_epoch']}"
        )

    total_time = time.time() - total_start

    logger.info("Ablation comparison:")
    for metric in ["chamfer", "hausdorff", "iou", "closure", "ordering"]:
        values = [all_results[dt]["eval"][metric]["mean"] for dt in decoder_types]
        logger.info(
            f"  {metric}: polar={values[0]:.4f}, detr={values[1]:.4f}, "
            f"autoregressive={values[2]:.4f}"
        )

    conv_epochs = [all_results[dt]["convergence_epoch"] for dt in decoder_types]
    logger.info(
        f"Convergence epochs: polar={conv_epochs[0]}, detr={conv_epochs[1]}, "
        f"autoregressive={conv_epochs[2]}"
    )
    logger.info(f"Ablation total time: {total_time / 60:.1f} min")
    # Save comparison
    comparison_path = Path(output_dir) / "ablation_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Ablation comparison saved to {comparison_path}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--fuseg-dir", type=str, default=None)
    parser.add_argument("--synthetic-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    run_ablation(
        fuseg_dir=args.fuseg_dir,
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        num_points=args.num_points,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
