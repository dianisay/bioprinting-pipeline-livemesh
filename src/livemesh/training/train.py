"""Main training loop for wound boundary detection models.

Trains encoder + decoder with early stopping, logging, and checkpointing.
Designed to run on Kaggle GPU or locally on CPU (slower).
"""

import logging

import torch

logger = logging.getLogger(__name__)
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional

from livemesh.perception.encoder import CNNTransformerEncoder
from livemesh.perception.polar_decoder import PolarDecoder, PolarBoundaryLoss
from livemesh.perception.detr_decoder import DETRDecoder, HungarianLoss
from livemesh.perception.autoregressive_decoder import AutoregressiveDecoder, AutoregressiveLoss
from livemesh.data.dataset import create_dataloaders


class Trainer:
    """Training engine for CNN-Transformer boundary detection."""

    def __init__(
        self,
        decoder_type: str = "polar",
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_points: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 8,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "auto",
        output_dir: str = "results",
        pretrained_backbone: bool = True,
    ):
        self.decoder_type = decoder_type
        self.max_epochs = max_epochs
        self.patience = patience
        self.output_dir = Path(output_dir) / decoder_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Training device: {self.device}")

        # Build model
        self.encoder = CNNTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            pretrained=pretrained_backbone,
        ).to(self.device)

        self.decoder, self.criterion = self._build_decoder(
            decoder_type, d_model, num_heads, num_decoder_layers, num_points
        )
        self.decoder = self.decoder.to(self.device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Scheduler: cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        # Logging
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch_time": [],
        }
        if decoder_type == "polar":
            self.history.update({"loss_centroid": [], "loss_radii": [], "loss_points": []})

    def _build_decoder(self, decoder_type, d_model, num_heads, num_layers, num_points):
        if decoder_type == "polar":
            decoder = PolarDecoder(d_model=d_model, num_radii=num_points)
            criterion = PolarBoundaryLoss()
        elif decoder_type == "detr":
            decoder = DETRDecoder(
                d_model=d_model, num_heads=num_heads,
                num_layers=num_layers, num_queries=num_points,
            )
            criterion = HungarianLoss()
        elif decoder_type == "autoregressive":
            decoder = AutoregressiveDecoder(
                d_model=d_model, num_heads=num_heads,
                num_layers=num_layers, num_points=num_points,
            )
            criterion = AutoregressiveLoss()
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        return decoder, criterion

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"Training {self.decoder_type} decoder: epochs={self.max_epochs}, "
            f"patience={self.patience}, train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}"
        )
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            self.scheduler.step()
            epoch_time = time.time() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            self.history["epoch_time"].append(epoch_time)

            logger.info(
                f"Epoch {epoch}/{self.max_epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={self.history['lr'][-1]:.2e}, "
                f"time={epoch_time:.1f} s"
            )
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best.pth", epoch, val_loss)
                logger.info(
                    f"New best model saved: val_loss={val_loss:.4f} at epoch {epoch}"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (patience={self.patience})"
                    )
                    break

        # Save final checkpoint and history
        self._save_checkpoint("final.pth", epoch, val_loss)
        self._save_history()

        logger.info(
            f"Training complete: best_val_loss={best_val_loss:.4f}, "
            f"results_dir={self.output_dir}"
        )
        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = {
                "centroid": batch["centroid"].to(self.device),
                "radii": batch["radii"].to(self.device),
                "points": batch["points"].to(self.device),
            }

            self.optimizer.zero_grad()
            features = self.encoder(images)

            if self.decoder_type == "autoregressive":
                pred = self.decoder(features, target_points=targets["points"])
            else:
                pred = self.decoder(features)

            losses = self.criterion(pred, targets)
            loss = losses["total"]
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            total_loss += loss.item()

            # Log components for polar
            if self.decoder_type == "polar" and len(self.history["loss_centroid"]) < len(self.history["train_loss"]) + 1:
                pass  # logged at epoch level below

        avg_loss = total_loss / len(loader)

        # Log polar components (last batch)
        if self.decoder_type == "polar":
            self.history["loss_centroid"].append(losses["centroid"].item())
            self.history["loss_radii"].append(losses["radii"].item())
            self.history["loss_points"].append(losses["points"].item())

        return avg_loss

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = {
                "centroid": batch["centroid"].to(self.device),
                "radii": batch["radii"].to(self.device),
                "points": batch["points"].to(self.device),
            }

            features = self.encoder(images)

            if self.decoder_type == "autoregressive":
                pred = self.decoder(features, target_points=targets["points"])
            else:
                pred = self.decoder(features)

            losses = self.criterion(pred, targets)
            total_loss += losses["total"].item()

        return total_loss / len(loader)

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "decoder_type": self.decoder_type,
            "config": {
                "d_model": self.encoder.d_model,
                "num_heads": self.encoder.transformer.layers[0].mha.num_heads,
                "num_encoder_layers": len(self.encoder.transformer.layers),
                "num_decoder_layers": len(self.decoder.layers) if hasattr(self.decoder, "layers") else 0,
                "num_points": self._get_num_points(),
            },
        }, self.output_dir / filename)
        logger.debug(f"Checkpoint saved: {self.output_dir / filename} (epoch={epoch})")
    def _get_num_points(self) -> int:
        if self.decoder_type == "polar":
            return self.decoder.num_radii
        elif self.decoder_type == "detr":
            return self.decoder.num_queries
        else:
            return self.decoder.num_points

    def _save_history(self):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train wound boundary model")
    parser.add_argument("--decoder", type=str, default="polar", choices=["polar", "detr", "autoregressive"])
    parser.add_argument("--fuseg-dir", type=str, default=None)
    parser.add_argument("--synthetic-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        fuseg_dir=args.fuseg_dir,
        synthetic_dir=args.synthetic_dir,
        batch_size=args.batch_size,
        num_radii=args.num_points,
    )

    # Train
    trainer = Trainer(
        decoder_type=args.decoder,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        num_points=args.num_points,
        output_dir=args.output_dir,
        pretrained_backbone=not args.no_pretrained,
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
