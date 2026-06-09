"""Tests for U-Net architecture (no trained weights required)."""

import numpy as np
import pytest
import torch

from livemesh.segmentation.unet import UNet


class TestUNet:
    def test_output_shape(self):
        model = UNet(in_channels=3, num_classes=2, base_filters=32, depth=4)
        x = torch.randn(1, 3, 512, 512)
        y = model(x)
        assert y.shape == (1, 2, 512, 512)

    def test_predict_returns_class_indices(self):
        model = UNet(in_channels=3, num_classes=2)
        x = torch.randn(1, 3, 512, 512)
        pred = model.predict(x)
        assert pred.shape == (1, 512, 512)
        assert pred.dtype == torch.int64
        assert set(pred.unique().tolist()).issubset({0, 1})

    def test_smaller_input(self):
        model = UNet(in_channels=3, num_classes=2, base_filters=16, depth=3)
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        assert y.shape == (1, 2, 256, 256)

    def test_batch_processing(self):
        model = UNet(in_channels=3, num_classes=2, base_filters=16, depth=3)
        x = torch.randn(4, 3, 256, 256)
        y = model(x)
        assert y.shape[0] == 4
