"""Ablation study runner: trains all 3 decoder variants with identical settings.

Produces comparative results for the thesis (Section 4.2).
Designed to be run on Kaggle GPU.
"""

import json
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

    print("=" * 70)
    print("ABLATION STUDY: Polar vs DETR vs Autoregressive Decoder")
    print("=" * 70)
    print(f"  Epochs: {max_epochs}, Patience: {patience}, LR: {lr}")
    print(f"  Batch size: {batch_size}, Num points: {num_points}")
    print(f"  Pretrained backbone: {pretrained}")
    print(f"  Output: {output_dir}")
    print()

    # Create dataloaders once (shared across all variants)
    train_loader, val_loader, test_loader = create_dataloaders(
        fuseg_dir=fuseg_dir,
        synthetic_dir=synthetic_dir,
        batch_size=batch_size,
        num_radii=num_points,
    )

    total_start = time.time()

    for decoder_type in decoder_types:
        print(f"\n{'='*70}")
        print(f"  Training: {decoder_type.upper()} decoder")
        print(f"{'='*70}")

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

    total_time = time.time() - total_start

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<15} {'Polar':>12} {'DETR':>12} {'Autoregr.':>12}")
    print(f"  {'-'*51}")

    for metric in ["chamfer", "hausdorff", "iou", "closure", "ordering"]:
        values = []
        for dt in decoder_types:
            v = all_results[dt]["eval"][metric]["mean"]
            values.append(v)
        print(f"  {metric:<15} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f}")

    print(f"\n  {'Convergence':<15} ", end="")
    for dt in decoder_types:
        print(f"{all_results[dt]['convergence_epoch']:>12d}", end="")
    print(" epochs")

    print(f"\n  Total training time: {total_time/60:.1f} minutes")

    # Save comparison
    comparison_path = Path(output_dir) / "ablation_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Comparison saved to: {comparison_path}")

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
