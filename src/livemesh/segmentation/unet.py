"""
U-Net for wound segmentation.

Ported from your MATLAB WoundSegmentation_Training.m.
Architecture matches MATLAB's unetLayers defaults:
  - Encoder depth: 4
  - Base filters: 32
  - 2x (Conv3x3 + ReLU) per stage + MaxPool
  - Skip connections via concatenation
  - TransposedConv 2x2 decoder
  - 2-class softmax head (background, wound)
  - Input: 512x512x3 RGB
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net matching your MATLAB architecture exactly."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_filters: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = in_channels
        encoder_channels = []

        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.encoders.append(_double_conv(ch, out_ch))
            encoder_channels.append(out_ch)
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        bottleneck_ch = base_filters * (2**depth)
        self.bottleneck = _double_conv(ch, bottleneck_ch)

        for i in range(depth - 1, -1, -1):
            out_ch = encoder_channels[i]
            self.upconvs.append(nn.ConvTranspose2d(bottleneck_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(_double_conv(out_ch * 2, out_ch))
            bottleneck_ch = out_ch

        self.head = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.head(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
