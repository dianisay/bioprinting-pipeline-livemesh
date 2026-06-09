"""Evaluation script: compute all metrics on test set from a saved checkpoint."""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from livemesh.perception.encoder import CNNTransformerEncoder
from livemesh.perception.polar_decoder import PolarDecoder
from livemesh.perception.detr_decoder import DETRDecoder
from livemesh.perception.autoregressive_decoder import AutoregressiveDecoder
from livemesh.data.dataset import create_dataloaders
from livemesh.utils.metrics import chamfer_distance, hausdorff_distance, boundary_iou, closure_error, ordering_consistency


def load_model(checkpoint_path: str, device: torch.device):
    """Load encoder + decoder from checkpoint using saved config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder_type = ckpt["decoder_type"]
    cfg = ckpt.get("config", {})

    d_model = cfg.get("d_model", 256)
    num_heads = cfg.get("num_heads", 8)
    num_encoder_layers = cfg.get("num_encoder_layers", 6)
    num_decoder_layers = cfg.get("num_decoder_layers", 6)
    num_points = cfg.get("num_points", 64)

    encoder = CNNTransformerEncoder(
        d_model=d_model, num_heads=num_heads,
        num_layers=num_encoder_layers, pretrained=False,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])

    if decoder_type == "polar":
        decoder = PolarDecoder(d_model=d_model, num_radii=num_points).to(device)
    elif decoder_type == "detr":
        decoder = DETRDecoder(
            d_model=d_model, num_heads=num_heads,
            num_layers=num_decoder_layers, num_queries=num_points,
        ).to(device)
    elif decoder_type == "autoregressive":
        decoder = AutoregressiveDecoder(
            d_model=d_model, num_heads=num_heads,
            num_layers=num_decoder_layers, num_points=num_points,
        ).to(device)
    else:
        raise ValueError(f"Unknown decoder: {decoder_type}")

    decoder.load_state_dict(ckpt["decoder_state"])
    return encoder, decoder, decoder_type


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    fuseg_dir: str = None,
    synthetic_dir: str = None,
    batch_size: int = 8,
    output_path: str = None,
):
    """Run full evaluation on test set.

    Computes per-sample: Chamfer, Hausdorff, IoU, closure error, ordering.
    Reports mean, std, min, max for each metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, decoder_type = load_model(checkpoint_path, device)
    encoder.eval()
    decoder.eval()

    _, _, test_loader = create_dataloaders(
        fuseg_dir=fuseg_dir,
        synthetic_dir=synthetic_dir,
        batch_size=batch_size,
    )

    print(f"Evaluating {decoder_type} on {len(test_loader.dataset)} test samples...")

    all_metrics = {
        "chamfer": [],
        "hausdorff": [],
        "iou": [],
        "closure": [],
        "ordering": [],
    }

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        gt_points = batch["points"].numpy()

        features = encoder(images)

        if decoder_type == "autoregressive":
            pred = decoder(features)
        else:
            pred = decoder(features)

        pred_points = pred["points"].cpu().numpy()

        for i in range(len(images)):
            p = pred_points[i]
            g = gt_points[i]

            all_metrics["chamfer"].append(chamfer_distance(p, g))
            all_metrics["hausdorff"].append(hausdorff_distance(p, g))
            all_metrics["iou"].append(boundary_iou(p, g))
            all_metrics["closure"].append(closure_error(p))
            all_metrics["ordering"].append(ordering_consistency(p, g))

    # Compute summary statistics
    results = {"decoder_type": decoder_type, "num_samples": len(all_metrics["chamfer"])}
    for metric_name, values in all_metrics.items():
        arr = np.array(values)
        results[metric_name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {decoder_type} decoder ({results['num_samples']} samples)")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*47}")
    for m in ["chamfer", "hausdorff", "iou", "closure", "ordering"]:
        r = results[m]
        print(f"  {m:<15} {r['mean']:>8.4f} {r['std']:>8.4f} {r['min']:>8.4f} {r['max']:>8.4f}")

    # Save
    if output_path is None:
        output_path = str(Path(checkpoint_path).parent / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fuseg-dir", type=str, default=None)
    parser.add_argument("--synthetic-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.fuseg_dir, args.synthetic_dir, args.batch_size, args.output)


if __name__ == "__main__":
    main()
