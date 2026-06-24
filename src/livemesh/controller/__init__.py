"""Closed-loop bioprinting controller.

Modules:
    depth_sensor    - Intel RealSense D405 model (simulated + hardware)
    depth_fusion    - Confidence-weighted depth blending (predicted + measured)
    closed_loop     - Scan-deposit-verify-correct printing loop
    wound_bridge    - Bridge: decoder output -> trajectory planner input
"""

from livemesh.controller.depth_sensor import DepthSensorModel, RealSenseD405
from livemesh.controller.depth_fusion import fuse_depth, fuse_depth_polar, DepthFusionConfig
from livemesh.controller.closed_loop import PrintingLoopController, PrintingState, LayerResult
from livemesh.controller.wound_bridge import bridge_decoder_to_planner, apply_depth_correction

__all__ = [
    "DepthSensorModel",
    "RealSenseD405",
    "fuse_depth",
    "fuse_depth_polar",
    "DepthFusionConfig",
    "PrintingLoopController",
    "PrintingState",
    "LayerResult",
    "bridge_decoder_to_planner",
    "apply_depth_correction",
]
